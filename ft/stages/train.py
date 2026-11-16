"""train: upload, launch the LoRA job, poll (spec 9.7).

Never launches a second job for an attempt: if state.json already has a job ID, it polls that job
(each create call starts a new billed job).
"""

from __future__ import annotations

import time

from ft.config import ConfigError, stable_hash
from ft.io import read_json, write_json
from ft.providers.base import TrainParams
from ft.spend import estimate_training_usd
from ft.stages import StageResult, upstream

NAME = "train"


class TrainingFailed(Exception):
    pass


def input_hash(ctx) -> str:
    cfg = ctx.cfg_for(ctx.attempt)
    return stable_hash([upstream(ctx, "prepare", attempt=ctx.attempt), cfg.train.hash_view()])


def run(ctx) -> StageResult:
    n = ctx.attempt
    cfg = ctx.cfg_for(n)
    adir = ctx.attempt_dir(n)
    trainer = ctx.providers.trainer
    base = cfg.train.base_model

    info = ctx.models.get(base)
    if info is None:
        raise ConfigError(f"train.base_model '{base}' is not in models.yaml")
    if not info.deployable and not ctx.dry_run:
        raise ConfigError(f"'{base}' is not marked deployable in models.yaml; verify it can be served first")
    provider = cfg.train.provider
    if info.provider != provider:
        raise ConfigError(f"'{base}' is a {info.provider} model in models.yaml, but train.provider is {provider}")
    trainer.preflight(base)

    ext = ctx.state.external(n)
    record_path = adir / "training.json"
    record = read_json(record_path) if record_path.exists() else {"status_history": []}
    prep = read_json(adir / "prepare.json")
    per_mtok = ctx.prices.training_per_mtok(base)
    min_charge = ctx.prices.training_min_charge(base)

    params = TrainParams(
        base_model=base,
        epochs=cfg.train.epochs,
        learning_rate=cfg.train.learning_rate,
        lora=cfg.train.lora,
        suffix=f"{ctx.run_id}-a{n}"[-64:],
        seed=cfg.splits.seed,
    )
    if "job_id" not in ext:
        if "file_id" not in ext:  # uploading and validating the file is free
            ctx.log("uploading training file and waiting for validation")
            file_id = trainer.upload(adir / "train.together.jsonl")
            ctx.state.set_external(n, file_id=file_id)
            ctx.events.emit("external_id", stage=NAME, attempt=n, file_id=file_id)
        file_id = ctx.state.external(n)["file_id"]
        local = estimate_training_usd(prep["approx_tokens"], cfg.train.epochs, per_mtok, min_charge)
        provider_est = trainer.estimate(file_id, params)
        est = max(provider_est, min_charge) if provider_est is not None else local
        record.update(estimate_usd={"provider": provider_est, "local": round(local, 4)})
        src = "the provider's estimate" if provider_est is not None else "local estimate"
        ctx.spend.check(
            NAME, provider, est, f"LoRA {base}, {prep['rows']} rows × {cfg.train.epochs} epochs ({src})", attempt=n
        )
        job_id = trainer.start(file_id, params)
        ctx.state.set_external(n, job_id=job_id)  # saved before anything else can fail
        ctx.events.emit("external_id", stage=NAME, attempt=n, job_id=job_id)
    else:
        ctx.events.emit("resume_poll", stage=NAME, attempt=n, job_id=ext["job_id"])
    job_id = ctx.state.external(n)["job_id"]
    record.update(file_id=ctx.state.external(n).get("file_id"), job_id=job_id, base_model=base)

    last = None
    while True:
        status = trainer.poll(job_id)
        if status.status != last:
            ctx.log(f"training job {job_id}: {status.status}")
            record["status_history"].append({"status": status.status, "ts": time.time()})
            ctx.events.emit("train_status", stage=NAME, attempt=n, status=status.status)
            write_json(record_path, record)
            last = status.status
        if status.terminal:
            break
        time.sleep(trainer.poll_seconds)

    if status.status != "completed":
        record["error"] = status.error
        write_json(record_path, record)
        raise TrainingFailed(f"training job {job_id} ended as {status.status}: {status.error}")

    ref = trainer.result(job_id)
    ctx.state.set_external(n, model_object_id=ref.model_object_id)
    # Prefer the provider's billed amount; otherwise price the token count ourselves.
    est_tokens = prep["approx_tokens"] * cfg.train.epochs
    if status.billed_usd:
        usd, basis = status.billed_usd, "billed by provider"
    else:
        usd = estimate_training_usd(prep["approx_tokens"], cfg.train.epochs, per_mtok, min_charge)
        basis = "estimated"
    ctx.spend.record(NAME, provider, usd, f"LoRA {base} ({basis})", attempt=n)
    record.update(
        model_object_id=ref.model_object_id,
        loss_curve=status.loss_curve,
        provider_token_count=status.trained_tokens,
        estimated_billed_tokens=est_tokens,
        usd=round(usd, 4),
        usd_basis=basis,
    )
    write_json(record_path, record)
    return StageResult(f"job {job_id} completed → {ref.model_object_id}", [record_path])
