"""Together provider: LoRA training (SDK), dedicated endpoints (CLI), tuned inference (OpenAI SDK).

Facts checked against together==2.41.0 and docs.together.ai on 2026-10-16 (see NOTES.md):
- fine-tuned models are served only on dedicated endpoints, billed per minute while a replica is ready;
- `tg beta endpoints deploy` creates the endpoint + deployment in one step; in JSON mode it needs an
  explicit --project, and --traffic-weight 1 routes the endpoint's traffic to the new deployment;
- endpoints never stop on their own unless --inactive-timeout is set; we set 30 min as a backstop to our
  own teardown;
- the job's `api_model_object_id` is the deployable (merged) model; `total_price` is in nanodollars;
- inference: OpenAI SDK at https://api-inference.together.ai/v1 with model = "<project_slug>/<endpoint>".
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError
from together import APIStatusError, Together

from ft.images import data_url
from ft.providers.base import (
    LOGPROB_TOP_K,
    Endpoint,
    Item,
    JobStatus,
    ModelRef,
    Prediction,
    PromptSpec,
    ProviderError,
    TrainParams,
    Usage,
    normalize_logprobs,
)
from ft.spend import SpendError

INFERENCE_BASE_URL = "https://api-inference.together.ai/v1"
TERMINAL = {"completed", "error", "cancelled", "failed"}
NANODOLLAR = 1_000_000_000
INACTIVE_TIMEOUT_MIN = 30


class TogetherError(ProviderError):
    pass


# ---------------------------------------------------------------- CLI wrapper


def _tg_path() -> str:
    exe = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "tg"
    return str(exe) if exe.exists() else (shutil.which("tg") or "tg")


def tg(*args: str, project: str | None = None, timeout: float = 300) -> Any:
    """Run the Together CLI in JSON, non-interactive mode and return parsed JSON."""
    cmd = [_tg_path(), "--json", "--non-interactive"]
    if project:
        cmd += ["--project", project]
    cmd += list(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
    out = proc.stdout.strip()
    try:
        data = json.loads(out) if out else None
    except json.JSONDecodeError:
        data = None
    if proc.returncode != 0 or (isinstance(data, dict) and "error" in data and len(data) == 1):
        detail = (data or {}).get("error") if isinstance(data, dict) else None
        raise TogetherError(f"`tg {' '.join(args)}` failed: {detail or proc.stderr.strip() or out}"[:1500])
    return data


def _get(d: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Read a field that may be serialized in snake_case or camelCase."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


CONFLICT_RETRY_DELAYS_S = (5, 10, 20, 30, 45, 60, 60)


def _retry_conflicts(call):
    """Retry a CLI call that fails with 409 Conflict (etag races, deployments mid-transition)."""
    for delay in (*CONFLICT_RETRY_DELAYS_S, None):
        try:
            return call()
        except TogetherError as e:
            if delay is None or not re.search(r"Conflict|409|aborted", str(e)):
                raise
            time.sleep(delay)


# ---------------------------------------------------------------- training


class TogetherTrainer:
    poll_seconds = 60.0
    file_poll_seconds = 5.0

    def __init__(self) -> None:
        self.client = Together()

    def preflight(self, base_model: str) -> None:
        data = tg("beta", "models", "configs", base_model)
        if not (data or {}).get("data"):
            raise TogetherError(f"{base_model} has no deployable configs; a fine-tune of it couldn't be served")

    def limits(self, base_model: str) -> dict[str, float]:
        """Fine-tuning limits for the model (`fine_tuning.model_limits`, free). Qwen3.5-4B, checked
        2026-10-18: learning rate 1e-8 to 1.0, at most 20 epochs."""
        r = self.client.fine_tuning.model_limits(model_name=base_model)
        return {
            "min_learning_rate": float(r.min_learning_rate),
            "max_learning_rate": float(r.max_learning_rate),
            "max_epochs": float(r.max_num_epochs),
        }

    def upload(self, path: Path) -> str:
        f = self.client.files.upload(file=path, purpose="fine-tune", check=True)
        deadline = time.monotonic() + 30 * 60
        while True:
            f = self.client.files.retrieve(f.id)
            status = f.processing_status
            if status in (None, "COMPLETED"):
                return f.id
            if status in ("FAILED", "INVALID_FORMAT"):
                report = f.validation_report.model_dump() if f.validation_report else None
                raise TogetherError(f"training file {f.id} was rejected ({status}): {report}")
            if time.monotonic() > deadline:
                raise TogetherError(f"training file {f.id} still {status} after 30 min")
            time.sleep(self.file_poll_seconds)

    def estimate(self, file_id: str, params: TrainParams) -> float | None:
        """Together's own price estimate for the job (free); None when unavailable."""
        try:
            est = self.client.fine_tuning.estimate_price(
                training_file=file_id, model=params.base_model, n_epochs=params.epochs, n_evals=0
            )
        except APIError:
            return None
        if getattr(est, "estimation_available", False) and est.estimated_total_price is not None:
            return float(est.estimated_total_price)
        return None

    def start(self, file_id: str, params: TrainParams) -> str:
        try:
            job = self.client.fine_tuning.create(
                training_file=file_id,
                model=params.base_model,
                n_epochs=params.epochs,
                learning_rate=params.learning_rate,
                lora=params.lora,
                train_on_inputs="auto",
                n_checkpoints=1,
                suffix=params.suffix,
                random_seed=params.seed,
            )
        except APIStatusError as e:
            if e.status_code == 402:  # seen live 2026-10-17: "insufficient_balance", no job is created
                raise SpendError(
                    f"Together refused the training job: the account's balance plus credit limit is too low "
                    f"({_message(e)}). Add credits at together.ai, then resume the run."
                ) from e
            raise
        return job.id

    def poll(self, job_id: str) -> JobStatus:
        job = self.client.fine_tuning.retrieve(job_id)
        status = str(job.status)
        terminal = status in TERMINAL
        out = JobStatus(status=status, terminal=terminal, trained_tokens=job.token_count)
        if job.total_price is not None:
            out.billed_usd = job.total_price / NANODOLLAR
        if terminal and status != "completed":
            out.error = self._last_error(job_id)
        if status == "completed":
            out.loss_curve = self._loss_curve(job_id)
        return out

    def _last_error(self, job_id: str) -> str | None:
        try:
            events = self.client.fine_tuning.list_events(job_id).data or []
        except APIError:
            return None
        msgs = [e.message for e in events if getattr(e, "level", None) in ("error", "legacy_ierror")]
        return (msgs or [e.message for e in events[-3:]] or [None])[-1]

    def _loss_curve(self, job_id: str) -> list[float] | None:
        try:
            metrics = self.client.fine_tuning.list_metrics(job_id, resolution=100).metrics or []
        except (APIError, TypeError):
            return None
        # Keys are "train/loss", "train/global_step", ... (checked 2026-10-16); older names as fallbacks.
        losses = [m.get("train/loss", m.get("train_loss", m.get("loss"))) for m in metrics]
        losses = [round(float(x), 4) for x in losses if x is not None]
        return losses or None

    def result(self, job_id: str) -> ModelRef:
        job = self.client.fine_tuning.retrieve(job_id)
        model_id = job.api_model_object_id or job.api_model_object_name
        if not model_id:
            raise TogetherError(f"job {job_id} completed without a model object id")
        return ModelRef(model_object_id=model_id, base_model=job.model or "")


# ---------------------------------------------------------------- serving


class TogetherServer:
    poll_seconds = 15.0

    def __init__(self, deploy_configs: dict[str, str | None] | None = None):
        self._project: str | None = os.environ.get("TOGETHER_PROJECT_ID")
        self.deploy_configs = deploy_configs or {}

    @property
    def project(self) -> str:
        if not self._project:
            who = tg("whoami")
            self._project = (who.get("project") or {}).get("id")
            if not self._project:
                raise TogetherError("couldn't determine the Together project id (set TOGETHER_PROJECT_ID)")
        return self._project

    def create(self, model: ModelRef, name: str) -> Endpoint:
        args = [
            "beta",
            "endpoints",
            "deploy",
            model.model_object_id,
            "--endpoint",
            name,
            "--min-replicas",
            "1",
            "--max-replicas",
            "1",
            "--inactive-timeout",
            str(INACTIVE_TIMEOUT_MIN),
        ]
        # No --traffic-weight here: on a new endpoint the CLI applies it with the etag it read before
        # creating the deployment, which always conflicts (409, seen 2026-10-16). Routing is set below.
        config = self.deploy_configs.get(model.base_model)
        if config:
            args += ["--config", config]
        try:
            data = tg(*args, project=self.project, timeout=600)
        except TogetherError as e:
            if "is a LoRA adapter" not in str(e):
                raise
            data = tg(*args, "--merge", project=self.project, timeout=600)  # adapter: serve it merged
        ep = data.get("endpoint") or {}
        dep = data.get("deployment") or {}
        if dep.get("id"):
            self._route_traffic(ep["id"], dep["id"])
        hardware = _get(dep, "hardware") or {}
        return Endpoint(
            id=ep["id"],
            name=name,
            model=ep["name"],  # "<project_slug>/<endpoint name>", the inference model string
            deployment_id=dep.get("id"),
            hardware_config=hardware.get("name") if isinstance(hardware, dict) else None,
        )

    def is_ready(self, endpoint: Endpoint) -> bool:
        ep = tg("beta", "endpoints", "get", endpoint.id, project=self.project)
        deps = _get(ep, "deployments", default=[]) or []
        states = [_get(d, "state", default="") for d in deps]
        if any(s in ("DEPLOYMENT_STATE_FAILED",) for s in states):
            raise TogetherError(f"endpoint {endpoint.id} deployment failed: {deps}")
        return any(
            s == "DEPLOYMENT_STATE_READY" and (_get(d, "ready_replicas", "readyReplicas", default=1) or 0) >= 1
            for s, d in zip(states, deps, strict=True)
        )

    def _route_traffic(self, endpoint_id: str, deployment_id: str) -> None:
        """Send the endpoint's live traffic to this deployment, unless it already gets it.
        `update` re-reads the endpoint (fresh etag) on every call, so retrying a conflict is safe."""
        ep = tg("beta", "endpoints", "get", endpoint_id, project=self.project)
        split = _get(ep, "traffic_split", "trafficSplit", default=[]) or []
        if any(_get(e, "deployment_id", "deploymentId") == deployment_id for e in split):
            return
        _retry_conflicts(
            lambda: tg("beta", "endpoints", "update", deployment_id, "--traffic-weight", "1", project=self.project)
        )

    def teardown(self, endpoint_id: str) -> None:
        """Clear routing, scale down, delete deployments and the endpoint. Already gone counts as done.
        A deployment that is mid-transition answers 409 for a while, so conflicts are retried."""
        try:
            _retry_conflicts(
                lambda: tg("beta", "endpoints", "rm", endpoint_id, "--force", project=self.project, timeout=600)
            )
        except TogetherError as e:
            if re.search(r"not.?found|404|does not exist", str(e), re.I):
                return
            raise

    def list_ft_endpoints(self) -> list[Endpoint]:
        data = tg("beta", "endpoints", "ls", project=self.project) or {}
        out = []
        for ep in data.get("data", []):
            short = ep["name"].split("/")[-1]
            if short.startswith("ft-"):
                out.append(Endpoint(id=ep["id"], name=short, model=ep["name"]))
        return out


# ---------------------------------------------------------------- tuned inference

_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str | None) -> str | None:
    """Qwen3.5 is a reasoning model; drop any <think> block before the answer."""
    if text is None:
        return None
    text = _THINK.sub("", text)
    return text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def user_content(item: Item) -> str | list[dict[str, Any]]:
    """Text as a string; an image as an OpenAI-style image_url part (base64 data URL), the same shape as
    the training file, plus any text."""
    if not item.image:
        return item.input
    parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_url(item.image)}}]
    if item.input:
        parts.append({"type": "text", "text": item.input})
    return parts


class TogetherTuned:
    """Calls the tuned endpoint through the OpenAI SDK (proves the 'change base URL + model' claim)."""

    base_url = INFERENCE_BASE_URL
    key_env = "TOGETHER_API_KEY"
    thinking_off: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}  # request option

    def __init__(self, max_retries: int = 6, timeout_s: float = 120.0):
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.disable_thinking = True
        # Together's docs describe `logprobs` as an integer 0-20, but a dedicated endpoint rejected that live
        # (2026-10-18: "invalid type: integer `5`, expected a boolean"), so it gets the OpenAI form:
        # logprobs: true + top_logprobs. If that's rejected too, it's dropped for the run.
        self.logprobs_k: int | None = LOGPROB_TOP_K
        self.notes: list[str] = []  # request options the endpoint rejected (the predict stage logs them)

    def _extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.disable_thinking:
            extra.update(self.thinking_off)
        if self.logprobs_k:
            extra["logprobs"] = True
            extra["top_logprobs"] = self.logprobs_k
        return extra

    async def _one(self, client: AsyncOpenAI, endpoint: Endpoint, item: Item, prompt: PromptSpec) -> Prediction:
        kwargs: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [{"role": "system", "content": prompt.system}, {"role": "user", "content": user_content(item)}],
            "temperature": 0,
            "max_tokens": prompt.max_output_tokens,
        }
        t0 = time.monotonic()
        try:
            while True:
                extra = self._extra()
                try:
                    resp = await client.chat.completions.create(**kwargs, extra_body=extra)
                    break
                except BadRequestError as e:
                    # Drop only what *this* request sent. Calls run concurrently: deciding from the shared
                    # flags let a burst of logprobs rejections also switch thinking back on (seen live
                    # 2026-10-18: every answer came back empty, the token budget spent on thinking).
                    if "logprobs" in extra:
                        if self.logprobs_k:
                            self.logprobs_k = None
                            self.notes.append(f"logprobs rejected by the endpoint, not requested again: {e}"[:300])
                    elif any(k in extra for k in self.thinking_off):
                        if self.disable_thinking:
                            self.disable_thinking = False  # server doesn't take the option; rely on stripping
                            self.notes.append(f"thinking-off option rejected by the endpoint: {e}"[:300])
                    else:
                        raise
        except APIError as e:
            return Prediction(
                id=item.id,
                prediction_raw=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=f"{type(e).__name__}: {e}"[:500],
            )
        latency = (time.monotonic() - t0) * 1000
        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0, output_tokens=getattr(u, "completion_tokens", 0) or 0
        )
        choice = resp.choices[0] if resp.choices else None
        raw = strip_thinking(choice.message.content if choice else None)
        lps = normalize_logprobs(getattr(choice, "logprobs", None)) if choice else None
        return Prediction(id=item.id, prediction_raw=raw, latency_ms=latency, usage=usage, logprobs=lps)

    async def predict(self, endpoint: Endpoint, items: list[Item], prompt: PromptSpec, concurrency: int):
        sem = asyncio.Semaphore(concurrency)
        key = os.environ.get(self.key_env)
        async with AsyncOpenAI(
            base_url=self.base_url, api_key=key, max_retries=self.max_retries, timeout=self.timeout_s
        ) as client:

            async def guarded(it: Item) -> Prediction:
                async with sem:
                    return await self._one(client, endpoint, it, prompt)

            return list(await asyncio.gather(*(guarded(it) for it in items)))


def _message(e: APIStatusError) -> str:
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body)
    return str(err.get("message") if isinstance(err, dict) else e)[:300]
