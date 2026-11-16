"""Whole-run cost and time estimate for a draft task.yaml, from the raw folder (the agent's estimate_cost
tool and the checkpoint).

Rougher than the per-stage estimates the spend guard uses later (those see the loaded data), but built
from the same prices.yaml and models.yaml, and it errs high. Times are fitted to measured runs (NOTES.md,
M3): stages run one after another, so the total is their sum.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from ft.agent.inspect import Folder, read_table
from ft.config import ConfigError, CsvSource, ImageFolderSource, JsonlSource, ListOf, TaskConfig
from ft.providers.openai_provider import REASONING_BUDGET, tpm_limit
from ft.registry import BaseModelInfo
from ft.spend import (
    ENDPOINT_PROVISION_MINUTES,
    OPENAI_TOKENS_PER_PATCH,
    REASONING_TOKENS_ESTIMATE,
    Prices,
    estimate_training_usd,
)
from ft.stages.load import SPLIT_NAMES, source_files

SECONDS_PER_TUNED_CALL = {"classify": 0.4, "extract": 2.0}
# Wall time per frontier call per concurrency slot when not throttled: measured ~1.5 s for Luna low on
# BANKING77 (3,080 calls in 290 s at 16), ~3.6 s for CORD receipts.
SECONDS_PER_FRONTIER_CALL = {"classify": 1.6, "extract": 4.0}
# Together LoRA jobs on Qwen3.5-4B, mostly fixed overhead (queueing, start-up, upload), then ~1M
# epoch-tokens/min. Measured end to end: BANKING77 1.3M epoch-tokens in 13.5 min, Enron 3.0M in 18.6 and
# 4.4M in 19.6, LEDGAR ~14.7M (our count) in 21. Errs high on large jobs.
TRAIN_FIXED_MINUTES = 12.0
TRAIN_TOKENS_PER_MINUTE = 1_000_000


class EstimateError(ValueError):
    pass


def split_rows(cfg: TaskConfig, folder: Folder) -> tuple[dict[str | None, list[dict]], list[dict]]:
    """Rows per split hint (None = unsplit), plus all rows."""
    src = cfg.source
    by_hint: dict[str | None, list[dict]] = {}
    if isinstance(src, CsvSource | JsonlSource):
        try:
            pairs = source_files(src, folder.repo_root / src.path)  # the files the load stage will read
        except ConfigError as e:
            raise EstimateError(str(e)) from e
        for p, hint in pairs:
            by_hint.setdefault(hint, []).extend(read_table(p))
    elif isinstance(src, ImageFolderSource):
        for r in read_table(folder.repo_root / src.labels_file):
            by_hint.setdefault(SPLIT_NAMES.get(str(r.get("split", "")).lower()), []).append(r)
    else:
        raise EstimateError(f"cost estimates from a raw folder support csv, jsonl and image_folder, not {src.type}")
    return by_hint, [r for rows in by_hint.values() for r in rows]


def split_sizes(cfg: TaskConfig, by_hint: dict[str | None, list[dict]]) -> dict[str, int]:
    """Mirrors the split stage's carving rules (no duplicate removal)."""
    s = cfg.splits
    total = sum(len(v) for v in by_hint.values())
    if s.by_time:
        test = min(s.test_max, max(1, round(total * s.test_fraction)))
        val = min(s.val_max, max(1, round((total - test) * s.val_fraction)))
        return {"train": total - test - val, "val": val, "test": test}
    hints = by_hint if s.use_predefined else {None: [r for v in by_hint.values() for r in v]}
    train = len(hints.get("train", [])) + len(hints.get(None, []))
    val, test = len(hints.get("val", [])), len(hints.get("test", []))
    if not test:
        test = min(s.test_max, max(1, round(train * s.test_fraction)))
        train -= test
    test = min(test, s.test_max)
    if not val:
        val = min(s.val_max, max(1, round(train * s.val_fraction)))
        train -= val
    return {"train": train, "val": min(val, s.val_max), "test": test}


def estimate_run(cfg: TaskConfig, folder: Folder, prices: Prices, models: dict[str, BaseModelInfo]) -> dict[str, Any]:
    by_hint, rows = split_rows(cfg, folder)
    if not rows:
        raise EstimateError("the source has no rows")
    sizes = split_sizes(cfg, by_hint)
    task = cfg.task
    sample = rows[:2000]

    # per-example input tokens
    if task.input.kind == "image":
        dims = folder.profile()["images"] or {}
        w, h = dims.get("median_size") or (1000, 1400)
        scale = min(1.0, task.input.max_side / max(w, h))
        patches = math.ceil(w * scale / 32) * math.ceil(h * scale / 32)
        input_tokens_tuned, input_tokens_frontier = patches, patches * OPENAI_TOKENS_PER_PATCH
    else:
        chars = statistics.mean(len(str(r.get(task.input.field, ""))) for r in sample)
        input_tokens_tuned = input_tokens_frontier = chars / 4 + 8

    # answer tokens and the frontier system prompt
    if task.type == "classify":
        labels = task.output.labels if isinstance(task.output.labels, list) else None
        if labels is None:
            labels = sorted({str(r.get(task.output.field, "")) for r in rows} - {""})
        answer = statistics.mean(len(lbl) for lbl in labels) / 3 + 4
        system = (len(task.instruction) + sum(len(lbl) + 1 for lbl in labels)) / 2
    else:
        fields = sum(len(v.list_of) if isinstance(v, ListOf) else 1 for v in (task.output.schema_ or {}).values())
        answer = 30 + fields * 12  # a compact JSON object; CORD-like receipts measured ~70-100 tokens
        system = len(task.instruction) / 4 + fields * 25

    lines = []
    n_eval = sizes["val"] + sizes["test"]
    for b in cfg.baselines:
        price = prices.token_price(b.model)
        sys_tokens = system
        if b.few_shot:
            sys_tokens += b.few_shot * (len(labels) if task.type == "classify" else 0) * input_tokens_frontier
        cached = sys_tokens >= 1024
        inp = n_eval * (sys_tokens + input_tokens_frontier)
        out = n_eval * (answer + 20 + REASONING_TOKENS_ESTIMATE.get(b.reasoning_effort, 500))
        usd = price.cost(inp, n_eval * sys_tokens * 0.95 if cached else 0, out)
        # The client throttle counts input plus the whole output allowance, cached input included, so
        # long few-shot prompts are bound by tokens per minute rather than by latency.
        counted = sys_tokens + input_tokens_frontier + answer + 32 + REASONING_BUDGET.get(b.reasoning_effort, 4096)
        minutes = max(
            n_eval * SECONDS_PER_FRONTIER_CALL[task.type] / cfg.eval.concurrency / 60, n_eval * counted / tpm_limit()
        )
        what = f"baseline {b.name or b.model} ({n_eval} calls"
        what += f", ~{sys_tokens:,.0f}-token cached prompt)" if b.few_shot else ")"
        lines.append({"item": what, "usd": round(usd, 3), "minutes": round(minutes)})

    base = models.get(cfg.train.base_model)
    if base is None:
        raise EstimateError(f"{cfg.train.base_model} is not in models.yaml")
    rows_train = min(sizes["train"], cfg.train.max_examples or sizes["train"])
    tokens = rows_train * (len(task.instruction) / 4 + input_tokens_tuned + answer + 15)
    train_usd = estimate_training_usd(
        int(tokens),
        cfg.train.epochs,
        prices.training_per_mtok(base.id),
        prices.training_min_charge(base.id),
    )
    lines.append(
        {
            "item": f"training {base.id}, ~{int(tokens):,} tokens x {cfg.train.epochs} epochs",
            "usd": round(train_usd, 2),
            "minutes": round(TRAIN_FIXED_MINUTES + tokens * cfg.train.epochs / TRAIN_TOKENS_PER_MINUTE),
        }
    )
    minutes = ENDPOINT_PROVISION_MINUTES + n_eval * SECONDS_PER_TUNED_CALL[task.type] / cfg.eval.concurrency / 60
    rate = prices.endpoint_per_hour(base.endpoint_config)
    lines.append(
        {"item": f"endpoint ~{minutes:.0f} min at ${rate}/h", "usd": round(minutes * rate / 60, 2),
         "minutes": round(minutes)}
    )  # fmt: skip
    total = round(sum(x["usd"] for x in lines), 2)
    return {
        "splits": sizes,
        "lines": lines,
        "total_usd": total,
        "total_minutes": sum(x["minutes"] for x in lines),
        "budget_max_spend_usd": cfg.budget.max_spend_usd,
        "fits_budget": total <= cfg.budget.max_spend_usd,
    }
