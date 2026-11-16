"""predict: deploy the tuned model, run val and test while it's up, tear down (spec 9.8, 9.9)."""

from __future__ import annotations

from ft.config import stable_hash
from ft.io import read_jsonl, write_json, write_jsonl
from ft.stages import StageResult, predict_split, prediction_row, upstream
from ft.stages.serve import deployed

NAME = "predict"


def input_hash(ctx) -> str:
    cfg = ctx.cfg_for(ctx.attempt)
    parts = [upstream(ctx, "train", attempt=ctx.attempt), upstream(ctx, "label"), cfg.task.instruction]
    if cfg.train.labels_in_prompt:  # appended only when on, so earlier attempts' hashes don't change
        parts.append(ctx.task_type(ctx.attempt).tuned_prompt().system)
    return stable_hash(parts)


def run(ctx) -> StageResult:
    n = ctx.attempt
    cfg = ctx.cfg_for(n)
    task = ctx.task_type(n)
    adir = ctx.attempt_dir(n)
    prompt = task.tuned_prompt()
    conc = cfg.eval.concurrency
    splits = {s: ctx.items(read_jsonl(ctx.data_dir / f"{s}.jsonl")) for s in ctx.eval_splits(n)}
    total = sum(len(v) for v in splits.values())

    outputs = []
    with deployed(ctx, ctx.tuned_model(n), n, task.eval_minutes(total, conc)) as ep:
        for split, items in splits.items():
            preds, run_meta = predict_split(
                ctx, lambda part: ctx.providers.tuned.predict(ep, part, prompt, conc), items, f"tuned {split}", conc
            )
            path = adir / f"{split}.predictions.jsonl"
            write_jsonl(
                path,
                [
                    prediction_row(task, p, structured=False, **({"logprobs": p.logprobs} if p.logprobs else {}))
                    for p in preds
                ],
            )
            write_json(adir / f"{split}.run.json", run_meta)
            outputs.append(path)
    for note in getattr(ctx.providers.tuned, "notes", []):
        ctx.events.emit("warning", stage=NAME, attempt=n, message=note)
        ctx.log(f"warning: {note}")
    return StageResult(f"{total} predictions from {ep.name} (torn down)", outputs)
