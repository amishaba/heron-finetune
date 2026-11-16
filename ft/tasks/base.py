"""TaskType interface: prompts, output schema, parsing, normalization, scoring.

All task-specific logic lives under ft/tasks/ (spec 17); stages stay generic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ft.config import TaskConfig
from ft.providers.base import PromptSpec
from ft.stats import MetricFn


@dataclass
class Parsed:
    value: Any
    valid: bool


@dataclass
class ScoreRow:
    id: str
    input: str
    target: Any
    parsed: Any
    valid: bool


class Scorer(ABC):
    """Holds encoded predictions so metrics can be recomputed on any subset of rows (bootstrap)."""

    n: int

    @abstractmethod
    def metric_fn(self, name: str) -> MetricFn: ...

    @abstractmethod
    def details(self) -> dict[str, Any]:
        """Secondary metrics and the 'where it goes wrong' tables."""


class TaskType(ABC):
    def __init__(self, cfg: TaskConfig, labels: list[str] | None):
        self.cfg = cfg
        self.labels = labels or []

    @property
    def instruction(self) -> str:
        return self.cfg.task.instruction.strip()

    @abstractmethod
    def baseline_prompt(self, context: str | None = None, examples: list[dict[str, Any]] | None = None) -> PromptSpec:
        """System prompt + structured-output schema for frontier baselines. `examples` ({input, target})
        go in the system prompt, the stable prefix that prompt caching covers."""

    def tuned_prompt(self) -> PromptSpec:
        """The tuned model sees only the instruction; it learned labels/schema in training (spec 9.5)."""
        return PromptSpec(system=self.instruction, max_output_tokens=self.tuned_max_tokens)

    tuned_max_tokens: int = 16
    # Seconds per tuned-endpoint call at the configured concurrency, for the endpoint-time estimate only.
    est_seconds_per_call: float = 0.4

    def eval_minutes(self, n_calls: int, concurrency: int) -> float:
        """Rough minutes a tuned endpoint needs to answer n_calls (for the endpoint spend estimate)."""
        return n_calls * self.est_seconds_per_call / concurrency / 60

    @abstractmethod
    def train_target(self, target: Any) -> str:
        """Assistant message content for a training row."""

    @abstractmethod
    def parse(self, raw: str | None, *, structured: bool) -> Parsed: ...

    @abstractmethod
    def scorer(self, rows: list[ScoreRow]) -> Scorer: ...
