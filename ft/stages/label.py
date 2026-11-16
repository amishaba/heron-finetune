"""label: attach targets where they aren't already present (spec 9.4)."""

from __future__ import annotations

from ft.config import ConfigError, stable_hash
from ft.io import read_jsonl
from ft.stages import StageResult, upstream

NAME = "label"


def input_hash(ctx) -> str:
    return stable_hash([upstream(ctx, "split"), ctx.cfg.targets.model_dump(by_alias=True)])


def run(ctx) -> StageResult:
    source = ctx.cfg.targets.from_
    if source == "frontier":
        raise NotImplementedError("targets.from 'frontier' (frontier labeling) isn't built yet")
    paths = [ctx.data_dir / f"{s}.jsonl" for s in ("train", "val", "test")]
    for p in paths:
        empty = [r["id"] for r in read_jsonl(p) if r["target"] in (None, "", [], {})]
        if empty:
            raise ConfigError(f"{p.name}: {len(empty)} records have an empty target (e.g. {empty[:3]})")
    what = {"behavior": "behavior labels (from targets.rules)", "logs": "logged responses as targets"}.get(
        source, "dataset targets"
    )
    return StageResult(f"{what} present on every record", paths)
