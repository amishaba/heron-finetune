"""profile: summary of the loaded data for the agent (M3) and the report (spec 9.2)."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np

from ft.config import stable_hash
from ft.io import read_jsonl, write_json
from ft.stages import StageResult, upstream

NAME = "profile"


def input_hash(ctx) -> str:
    return stable_hash(upstream(ctx, "load"))


def _dtype(values: list[Any]) -> str:
    kinds = {type(v).__name__ for v in values if v not in (None, "")}
    if not kinds:
        return "empty"
    if kinds <= {"str"}:
        sample = [v for v in values if v not in (None, "")][:200]
        if all(_is_number(v) for v in sample):
            return "numeric-string"
        return "string"
    return "|".join(sorted(kinds))


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def field_profile(values: list[Any]) -> dict[str, Any]:
    n = len(values)
    nulls = sum(v in (None, "") for v in values)
    as_text = [v if isinstance(v, str) else repr(v) for v in values if v not in (None, "")]
    lengths = np.array([len(t) for t in as_text]) if as_text else np.array([0])
    counts = Counter(as_text)
    return {
        "dtype": _dtype(values),
        "null_rate": round(nulls / n, 4) if n else 0.0,
        "length": {
            "min": int(lengths.min()),
            "mean": round(float(lengths.mean()), 1),
            "p95": int(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        },
        "cardinality": len(counts),
        "top_values": [[v[:120], c] for v, c in counts.most_common(30)],
    }


def run(ctx) -> StageResult:
    records = read_jsonl(ctx.data_dir / "records.jsonl")
    fields: dict[str, list[Any]] = {"input": [r["input"] for r in records], "target": [r["target"] for r in records]}
    for r in records:
        for k, v in r["meta"].items():
            fields.setdefault(f"meta.{k}", []).append(v)
    rng = random.Random(ctx.cfg.splits.seed)
    samples = rng.sample(records, min(20, len(records)))
    profile = {
        "rows": len(records),
        "fields": {k: field_profile(v) for k, v in fields.items()},
        "split_hints": dict(Counter(r["meta"].get("split_hint", "none") for r in records)),
        "samples": [{"id": s["id"], "input": s["input"][:500], "target": s["target"]} for s in samples],
    }
    out = ctx.data_dir / "profile.json"
    write_json(out, profile)
    return StageResult(f"{len(records)} rows, {len(fields)} fields", [out])
