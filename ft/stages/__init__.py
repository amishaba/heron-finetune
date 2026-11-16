"""Pipeline stages. Each module exposes NAME, input_hash(ctx) and run(ctx) -> StageResult.

Stages read task.yaml (via ctx) and the run folder, and write outputs atomically into the run folder.
They never import provider SDKs (spec 9, 17).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ft.context import RunContext
    from ft.providers.base import Prediction
    from ft.tasks.base import TaskType


@dataclass
class StageResult:
    summary: str
    outputs: list[Path] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


def upstream(ctx: RunContext, *stages: str, attempt: int | None = None) -> list[str | None]:
    """Output hashes of earlier stages; part of a stage's input hash so changes cascade."""
    return [ctx.state.stage(s, attempt).get("output_hash") for s in stages]


MAX_FAILED_FRACTION = 0.02
CHUNK = 400  # calls per progress line, and per mid-stage spend check


class TooManyFailures(RuntimeError):
    def __init__(self, message: str, predictions: list):
        super().__init__(message)
        self.predictions = predictions  # every call made, so the caller can still record what was paid for


def is_transient(error: str | None) -> bool:
    """API/network failures, as opposed to model outcomes (refusal, ran out of output budget)."""
    return bool(error) and not error.startswith(("refusal", "incomplete"))


def run_predictions(
    ctx: RunContext,
    call: Callable[[list], Awaitable[list]],
    items: list,
    what: str,
    after_chunk: Callable[[list], None] | None = None,
) -> list:
    """Run `call` over items in chunks (progress lines), retry transient failures once, and stop if more
    than 2% still fail, so an outage never shows up as a model being wrong. `after_chunk` sees every
    prediction so far after each chunk and may raise to stop (e.g. a spend projection over the cap)."""
    chunk = CHUNK
    by_id: dict[str, Any] = {}
    for start in range(0, len(items), chunk):
        part = items[start : start + chunk]
        for p in asyncio.run(call(part)):
            by_id[p.id] = p
        if len(items) > chunk:
            ctx.log(f"{what}: {min(start + chunk, len(items))}/{len(items)}")
        if after_chunk is not None:
            after_chunk(list(by_id.values()))
    failed = [it for it in items if is_transient(by_id[it.id].error)]
    if failed:
        ctx.log(f"{what}: retrying {len(failed)} failed calls")
        for p in asyncio.run(call(failed)):
            by_id[p.id] = p
    still = [by_id[it.id] for it in items if is_transient(by_id[it.id].error)]
    if len(still) > MAX_FAILED_FRACTION * len(items):
        sample = "; ".join({p.error for p in still[:5]})
        raise TooManyFailures(
            f"{what}: {len(still)}/{len(items)} calls failed after a retry ({sample})", list(by_id.values())
        )
    return [by_id[it.id] for it in items]


def predict_split(
    ctx: RunContext,
    call: Callable[[list], Awaitable[list]],
    items: list,
    what: str,
    concurrency: int,
    after_chunk: Callable[[list], None] | None = None,
) -> tuple[list, dict[str, Any]]:
    """`run_predictions` over one split, plus its run metadata (what `<split>.run.json` holds and the
    evaluate stage turns into throughput). Baselines and tuned models are both timed here."""
    t0 = time.monotonic()
    preds = run_predictions(ctx, call, items, what, after_chunk)
    wall = time.monotonic() - t0
    if ctx.dry_run:  # fakes don't sleep; derive wall time from simulated latency
        wall = sum(p.latency_ms for p in preds) / 1000 / concurrency
    return preds, {"wall_seconds": wall, "concurrency": concurrency, "simulated": ctx.dry_run}


def prediction_row(task: TaskType, p: Prediction, *, structured: bool, **extra: Any) -> dict[str, Any]:
    """One line of a `<split>.predictions.jsonl`: the shape `evaluate.score_rows` reads, for every model."""
    parsed = task.parse(p.prediction_raw, structured=structured)
    return {
        "id": p.id,
        "prediction_raw": p.prediction_raw,
        "prediction_parsed": parsed.value,
        "valid": parsed.valid,
        "latency_ms": round(p.latency_ms, 1),
        "usage": vars(p.usage),
        "error": p.error,
        **extra,
    }
