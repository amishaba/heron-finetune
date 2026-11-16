"""Deterministic fake providers for --dry-run and tests (spec 10.3). No network, no sleeping.

Fakes see `Item.target` so they can be right with a set probability. Latency is simulated
(reported, not slept), so throughput math in the report still has numbers to work with.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ft.providers.base import (
    BaselineSettings,
    Endpoint,
    Item,
    JobStatus,
    ModelRef,
    Prediction,
    PromptSpec,
    ProviderError,
    TrainParams,
    Usage,
)


class FakeFailure(ProviderError):
    pass


def _rng(*parts: Any) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _corrupt(target: Any, labels: list[str], rng: random.Random) -> Any:
    if isinstance(target, str) and labels:
        others = [lbl for lbl in labels if lbl != target]
        return rng.choice(others) if others else target
    if isinstance(target, dict):  # extract (M2): blank one field
        out = dict(target)
        if out:
            out[rng.choice(sorted(out))] = ""
        return out
    return target


def _serialize(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


@dataclass
class FakeOptions:
    seed: int = 0
    frontier_p: float = 0.90
    tuned_q: float = 0.93  # tuned ahead of the frontier fake, as in the real runs; the retry loop has its own tests
    tuned_q_by_attempt: dict[int, float] = field(default_factory=dict)
    frontier_latency_ms: float = 700.0
    tuned_latency_ms: float = 120.0
    polls_to_complete: int = 2
    fail_on: set[str] = field(default_factory=set)  # e.g. {"train.poll", "tuned.predict"}
    tuned_constant: str | None = None  # simulate a collapsed fine-tune: this exact output for every input
    frontier_cache: bool = True  # False simulates prompt caching not working (every call pays full input)
    # Tuned accuracy follows the training config (learning rate, epochs, rows), so the retry loop's fixes
    # visibly help or hurt in dry runs (spec 13 done-when). Off: tuned_q / tuned_q_by_attempt.
    tuned_q_from_config: bool = False
    full_rows: int = 2000  # rows at which more training data stops helping (config_quality)
    trained: dict[int, float] = field(default_factory=dict)  # attempt -> quality; in-process only

    labels: list[str] = field(default_factory=list)

    # Saved in state.json so a resumed dry run behaves like the one that started it.
    SAVED = ("seed", "frontier_p", "tuned_q", "tuned_q_by_attempt", "polls_to_complete", "tuned_constant",
             "frontier_cache", "tuned_q_from_config", "full_rows")  # fmt: skip

    def to_state(self) -> dict[str, Any]:
        out = {k: getattr(self, k) for k in self.SAVED}
        out["tuned_q_by_attempt"] = {str(k): v for k, v in self.tuned_q_by_attempt.items()}  # JSON keys are strings
        return out

    @classmethod
    def from_state(cls, saved: dict[str, Any]) -> FakeOptions:
        saved = dict(saved)
        saved["tuned_q_by_attempt"] = {int(k): v for k, v in saved.get("tuned_q_by_attempt", {}).items()}
        return cls(**saved)


def config_quality(base_q: float, lr: float, epochs: int, rows: int, full_rows: int = 2000) -> float:
    """A made-up response surface: best near lr 1e-4 to 3e-4, 3+ epochs and full_rows+ rows."""
    f_lr = min(1.0, (lr / 5e-5) ** 0.5) if lr < 5e-5 else (1.0 if lr <= 5e-4 else max(0.2, (5e-4 / lr) ** 0.5))
    f_ep = min(1.0, epochs / 3) ** 0.5
    f_rows = min(1.0, rows / full_rows) ** 0.3
    return round(base_q * f_lr * f_ep * f_rows, 4)


class FakeFrontier:
    def __init__(self, opts: FakeOptions):
        self.opts = opts
        self.calls = 0

    async def predict(self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings) -> list[Prediction]:
        out = []
        sys_tokens = len(prompt.system) // 4
        for it in items:
            self.calls += 1
            rng = _rng(self.opts.seed, "frontier", settings.cache_key or settings.model, it.id)
            answer = it.target if rng.random() < self.opts.frontier_p else _corrupt(it.target, self.opts.labels, rng)
            raw = (
                json.dumps({"label": answer}) if prompt.json_schema and isinstance(answer, str) else _serialize(answer)
            )
            cached = sys_tokens if sys_tokens >= 1024 and self.opts.frontier_cache else 0
            out.append(
                Prediction(
                    id=it.id,
                    prediction_raw=raw,
                    latency_ms=max(50.0, rng.gauss(self.opts.frontier_latency_ms, self.opts.frontier_latency_ms * 0.2)),
                    usage=Usage(
                        input_tokens=sys_tokens + len(it.input) // 4 + 8 + (1300 if it.image else 0),
                        cache_read_tokens=cached,
                        output_tokens=len(raw) // 4 + 40,
                    ),
                )
            )
        return out

    async def batch_predict(self, items, prompt, settings):
        return await self.predict(items, prompt, settings)


class FakeTrainer:
    poll_seconds = 0.0

    def __init__(self, opts: FakeOptions, known_models: set[str]):
        self.opts = opts
        self.known_models = known_models
        self.uploads: list[str] = []
        self.started: list[str] = []
        self._polls: dict[str, int] = {}

    def preflight(self, base_model: str) -> None:
        # Real trainer also requires `deployable: true`; dry runs only need the model to be registered.
        if base_model not in self.known_models:
            raise FakeFailure(f"base model '{base_model}' is not in models.yaml")

    def limits(self, base_model: str) -> dict[str, float]:
        return {"min_learning_rate": 1e-8, "max_learning_rate": 1.0, "max_epochs": 20}

    def upload(self, path: Path) -> str:
        fid = "file-fake-" + hashlib.sha256(path.read_bytes()).hexdigest()[:10]
        self.uploads.append(fid)
        self._rows = getattr(self, "_rows", {})
        self._rows[fid] = sum(1 for line in path.read_text().splitlines() if line.strip())
        return fid

    def estimate(self, file_id: str, params: TrainParams) -> float | None:
        return None  # fall back to the local estimate

    def start(self, file_id: str, params: TrainParams) -> str:
        jid = f"ft-fake-{params.suffix}"
        self.started.append(jid)
        tail = params.suffix.rsplit("-a", 1)[-1]
        if self.opts.tuned_q_from_config and tail.isdigit():
            rows = getattr(self, "_rows", {}).get(file_id, 2000)
            self.opts.trained[int(tail)] = config_quality(
                self.opts.tuned_q, params.learning_rate, params.epochs, rows, self.opts.full_rows
            )
        return jid

    def poll(self, job_id: str) -> JobStatus:
        if "train.poll" in self.opts.fail_on:
            raise FakeFailure("simulated crash while polling the training job")
        n = self._polls[job_id] = self._polls.get(job_id, 0) + 1
        if n < self.opts.polls_to_complete:
            return JobStatus(status="running", terminal=False)
        curve = [round(2.5 * (0.6 ** (i / 3)) + 0.1, 4) for i in range(12)]
        return JobStatus(status="completed", terminal=True, loss_curve=curve)

    def result(self, job_id: str) -> ModelRef:
        return ModelRef(model_object_id=f"model-{job_id}", base_model="fake")


class FakeServer:
    poll_seconds = 0.0

    def __init__(self, opts: FakeOptions):
        self.opts = opts
        self.deployed: list[str] = []
        self.torn_down: list[str] = []

    def create(self, model: ModelRef, name: str) -> Endpoint:
        eid = f"endpoint-fake-{name}"
        self.deployed.append(eid)
        return Endpoint(
            id=eid,
            name=name,
            model=f"fake-project/{name}",
            deployment_id=f"dep-{name}",
            hardware_config="fake-1x-gpu",
            simulated=True,
        )

    def is_ready(self, endpoint: Endpoint) -> bool:
        return True

    def teardown(self, endpoint_id: str) -> None:
        if endpoint_id not in self.torn_down:
            self.torn_down.append(endpoint_id)

    def list_ft_endpoints(self) -> list[Endpoint]:
        return []


class FakeTuned:
    def __init__(self, opts: FakeOptions):
        self.opts = opts
        self.calls = 0

    @staticmethod
    def _attempt(endpoint: Endpoint) -> int:
        # Endpoint names end in "-a<n>" (serve stage), which lets accuracy vary per attempt.
        tail = endpoint.name.rsplit("-a", 1)[-1]
        return int(tail) if tail.isdigit() else 1

    async def predict(self, endpoint: Endpoint, items: list[Item], prompt: PromptSpec, concurrency: int):
        if "tuned.predict" in self.opts.fail_on:
            raise FakeFailure("simulated failure while calling the tuned endpoint")
        attempt = self._attempt(endpoint)
        q = self.opts.trained.get(attempt) or self.opts.tuned_q_by_attempt.get(attempt, self.opts.tuned_q)
        out = []
        for it in items:
            self.calls += 1
            rng = _rng(self.opts.seed, "tuned", attempt, it.id)
            answer = it.target if rng.random() < q else _corrupt(it.target, self.opts.labels, rng)
            raw = self.opts.tuned_constant if self.opts.tuned_constant is not None else _serialize(answer)
            out.append(
                Prediction(
                    id=it.id,
                    prediction_raw=raw,
                    latency_ms=max(10.0, rng.gauss(self.opts.tuned_latency_ms, self.opts.tuned_latency_ms * 0.2)),
                    usage=Usage(
                        input_tokens=len(prompt.system) // 4 + len(it.input) // 4 + 8, output_tokens=len(raw) // 4 + 1
                    ),
                    # one "token" for the whole answer; surer when right, like a real model on average
                    logprobs=[{"t": raw, "lp": round(math.log(0.9 if answer == it.target else 0.45), 4), "top": {}}],
                )
            )
        return out
