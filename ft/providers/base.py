"""Provider interfaces (spec 10). Stages call these, never SDKs.

Deviation from the spec's sketch: `Server` exposes create / wait_ready / teardown instead of a
`deploy()` context manager. The context manager lives in `ft/stages/serve.py`, so the guarantees
(IDs saved the moment they exist, teardown in finally + atexit + signals) are written once for
every provider instead of once per provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ProviderError(RuntimeError):
    """A provider call failed in a way the run can't work around (rejected file, failed deployment, ...).
    The CLI reports it without a traceback; the run stays resumable."""


@dataclass
class Item:
    id: str
    input: str  # text ("" for image-only inputs)
    target: Any = None  # never sent to real providers; fakes use it to simulate accuracy
    image: str | None = None  # absolute path to a JPEG when task.input.kind == "image"


@dataclass
class PromptSpec:
    system: str
    json_schema: dict[str, Any] | None = None  # structured output schema (baselines)
    max_output_tokens: int = 64


@dataclass
class BaselineSettings:
    model: str
    reasoning_effort: str | None = None
    concurrency: int = 16
    cache_key: str | None = None  # groups requests that share a cached prompt prefix


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Prediction:
    id: str
    prediction_raw: str | None
    latency_ms: float
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    # Per output token, when the provider returns them: [{"t": token, "lp": logprob, "top": {token: logprob}}]
    # (first LOGPROB_TOKENS tokens). Saved for confidence-based routing later (NOTES.md, after-MVP ideas).
    logprobs: list[dict[str, Any]] | None = None


LOGPROB_TOKENS = 64
LOGPROB_TOP_K = 5


def normalize_logprobs(raw: Any) -> list[dict[str, Any]] | None:
    """A choice's logprobs in either shape we may get back, as [{"t", "lp", "top"}]:
    OpenAI's {content: [{token, logprob, top_logprobs: [{token, logprob}]}]}, or Together's documented
    {tokens: [...], token_logprobs: [...], top_logprobs: [{token: logprob}, ...]}."""
    if raw is None:
        return None
    d = raw.model_dump() if hasattr(raw, "model_dump") else raw
    if not isinstance(d, dict):
        return None
    out: list[dict[str, Any]] = []
    if d.get("content"):
        for c in d["content"][:LOGPROB_TOKENS]:
            top = {t["token"]: round(t["logprob"], 4) for t in c.get("top_logprobs") or [] if "token" in t}
            out.append({"t": c.get("token"), "lp": _round(c.get("logprob")), "top": top})
    elif d.get("tokens"):
        tops = d.get("top_logprobs")
        tops = tops if isinstance(tops, list) else [tops] * len(d["tokens"])
        for i, tok in enumerate(d["tokens"][:LOGPROB_TOKENS]):
            lps = d.get("token_logprobs") or []
            top = tops[i] if i < len(tops) and isinstance(tops[i], dict) else {}
            out.append({"t": tok, "lp": _round(lps[i] if i < len(lps) else None),
                        "top": {k: _round(v) for k, v in top.items()}})  # fmt: skip
    return out or None


def _round(x: Any) -> float | None:
    return None if x is None else round(float(x), 4)


@dataclass
class TrainParams:
    base_model: str
    epochs: int
    learning_rate: float
    lora: bool
    suffix: str
    seed: int | None = None


@dataclass
class JobStatus:
    status: str  # pending | queued | running | uploading | completed | error | cancelled
    terminal: bool
    trained_tokens: int | None = None
    loss_curve: list[float] | None = None
    error: str | None = None
    billed_usd: float | None = None  # provider-reported job cost, when available


@dataclass
class ModelRef:
    model_object_id: str
    base_model: str


@dataclass
class Endpoint:
    id: str
    name: str
    model: str  # the string to send as `model` on inference calls
    deployment_id: str | None = None
    hardware_config: str | None = None
    simulated: bool = False


class FrontierProvider(Protocol):
    async def predict(self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings) -> list[Prediction]: ...

    async def batch_predict(
        self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings
    ) -> list[Prediction]: ...


class Trainer(Protocol):
    poll_seconds: float

    def preflight(self, base_model: str) -> None: ...
    def limits(self, base_model: str) -> dict[str, float]: ...  # min/max learning_rate, max_epochs
    def upload(self, path: Path) -> str: ...
    def estimate(self, file_id: str, params: TrainParams) -> float | None: ...
    def start(self, file_id: str, params: TrainParams) -> str: ...
    def poll(self, job_id: str) -> JobStatus: ...
    def result(self, job_id: str) -> ModelRef: ...


class Server(Protocol):
    poll_seconds: float

    def create(self, model: ModelRef, name: str) -> Endpoint: ...
    def is_ready(self, endpoint: Endpoint) -> bool: ...
    def teardown(self, endpoint_id: str) -> None: ...
    def list_ft_endpoints(self) -> list[Endpoint]: ...


class TunedModel(Protocol):
    async def predict(
        self, endpoint: Endpoint, items: list[Item], prompt: PromptSpec, concurrency: int
    ) -> list[Prediction]: ...


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any]


@dataclass
class AgentTurn:
    calls: list[ToolCall]
    text: str = ""
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


class AgentLLM(Protocol):
    """A tool-calling model: the planning agent (ft/agent) and the retry loop's fixer (ft/fixer.py)."""

    model: str
    simulated: bool

    def start(self, system: str, user: str, tools: list[dict[str, Any]]) -> None: ...
    def step(self) -> AgentTurn: ...
    def add_results(self, results: list[tuple[str, str | list[str]]]) -> None: ...
    def nudge(self, text: str) -> None: ...


@dataclass
class Providers:
    frontier: FrontierProvider
    trainer: Trainer
    server: Server
    tuned: TunedModel
