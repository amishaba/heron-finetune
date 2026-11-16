"""prepare: training JSONL in the trainer's chat format (spec 9.5)."""

from __future__ import annotations

import random
from collections import Counter
from statistics import median
from typing import Any

from ft.config import stable_hash
from ft.images import data_url, patches
from ft.io import read_jsonl, write_json, write_jsonl
from ft.stages import StageResult, upstream
from ft.stages.split import stratified_take

NAME = "prepare"


def input_hash(ctx) -> str:
    cfg = ctx.cfg_for(ctx.attempt)
    parts = [upstream(ctx, "label", "split"), cfg.train.max_examples, cfg.task.instruction, cfg.splits.seed]
    shaping = {k: v for k, v in cfg.train.hash_view().items() if k in SHAPING_OPTIONS}
    if shaping:  # only when set, so attempts made before these options existed keep their hash
        parts.append(shaping)
    if ctx.is_retrain(ctx.attempt):
        parts.append("train+val")
    return stable_hash(parts)


SHAPING_OPTIONS = ("labels_in_prompt", "class_weighting", "class_weight_cap")


def _example(ctx, system: str, record: dict[str, Any], answer: str, image: bool) -> dict[str, Any]:
    """One training row in Together's conversational format. With images, every message's content is a
    list of parts: Together rejects files that mix plain-string and multimodal messages (checked live
    2026-10-17). Images are base64 data URLs; only user messages may carry them."""
    if not image:
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": record["input"]},
                {"role": "assistant", "content": answer},
            ]
        }
    url = data_url(ctx.image_path(record))
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
    }


def class_weights(rows: list[dict[str, Any]], method: str, cap: float) -> dict[str, float]:
    """weight = (median class size / class size) ** p, clipped to [1, cap]; never down-weights."""
    counts = Counter(str(r["target"]) for r in rows)
    mid = median(counts.values())
    p = {"sqrt": 0.5, "balanced": 1.0}[method]
    return {lbl: min(cap, max(1.0, (mid / n) ** p)) for lbl, n in counts.items()}


def oversample(rows: list[dict[str, Any]], weights: dict[str, float], rng: random.Random) -> list[dict[str, Any]]:
    """Repeat each row floor(w) times, plus once more with probability frac(w); then shuffle so copies
    aren't adjacent."""
    out = []
    for r in rows:
        w = weights[str(r["target"])]
        copies = int(w) + (rng.random() < w - int(w))
        out += [r] * copies
    rng.shuffle(out)
    return out


def run(ctx) -> StageResult:
    n = ctx.attempt
    cfg = ctx.cfg_for(n)
    task = ctx.task_type(n)
    rows = read_jsonl(ctx.data_dir / "train.jsonl")
    if ctx.is_retrain(n):  # the shipped model learns from val too; class weights follow the combined rows
        rows += read_jsonl(ctx.data_dir / "val.jsonl")
    if cfg.train.max_examples and cfg.train.max_examples < len(rows):
        key = (lambda r: r["target"]) if cfg.task.type == "classify" else None
        rows, _rest = stratified_take(rows, cfg.train.max_examples, key, random.Random(cfg.splits.seed))
    weighting: dict[str, Any] | None = None
    if cfg.train.class_weighting != "none":
        weights = class_weights(rows, cfg.train.class_weighting, cfg.train.class_weight_cap)
        before = Counter(str(r["target"]) for r in rows)
        rows = oversample(rows, weights, random.Random(f"{cfg.splits.seed}-weighting"))
        after = Counter(str(r["target"]) for r in rows)
        weighting = {
            "method": cfg.train.class_weighting,
            "cap": cfg.train.class_weight_cap,
            "rows_before": sum(before.values()),
            "rows_after": len(rows),
            "boosted": {
                lbl: {"rows": before[lbl], "weight": round(w, 2), "rows_after": after[lbl]}
                for lbl, w in sorted(weights.items(), key=lambda kv: -kv[1])
                if w > 1
            },
        }
    system = task.tuned_prompt().system
    image = cfg.task.input.kind == "image"
    lines = [_example(ctx, system, r, task.train_target(r["target"]), image) for r in rows]
    out = ctx.attempt_dir(n) / "train.together.jsonl"
    write_jsonl(out, lines)
    # chars/4 for text plus ~5 template tokens per message (role markers, separators); images count one
    # token per 32x32 patch (Qwen3.5: 16px patches merged 2x2). Together can't price multimodal files
    # (estimate_price: unavailable, "multimodal_dataset"), so this estimate is what the spend guard sees.
    text_chars = [len(system) + len(task.train_target(r["target"])) + (0 if image else len(r["input"])) for r in rows]
    approx_tokens = sum(c // 4 + 15 for c in text_chars)
    if image:
        approx_tokens += sum(patches(ctx.image_path(r)) for r in rows)
    stats: dict[str, Any] = {"rows": len(lines), "approx_tokens": approx_tokens}
    if weighting:
        stats["class_weighting"] = weighting
    if cfg.train.labels_in_prompt:
        stats["labels_in_prompt"] = True
    write_json(ctx.attempt_dir(n) / "prepare.json", stats)
    extra = f" (rare classes repeated: {weighting['rows_before']} → {len(lines)} rows)" if weighting else ""
    return StageResult(f"{len(lines)} training rows, ~{approx_tokens:,} tokens{extra}", [out], stats)
