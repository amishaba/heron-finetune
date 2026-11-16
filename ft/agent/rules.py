"""The agent's fixed rules, enforced in code (spec 12, NOTES.md M3).

The system prompt states these rules, but the agent doesn't always follow them (on LEDGAR it kept a
10,000-row test set against the 5,000 cap). So every draft goes through `apply_rules` before it is
validated, estimated or submitted. A mechanical choice is corrected, and the correction is reported
back and logged. A plan that can't be corrected (over the per-run cost limit) is rejected.

Judgment stays with the agent: task type, input and label columns, time splits, instructions, training
size, and whether to decline.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ft.agent.estimate import EstimateError, estimate_run, split_rows
from ft.agent.inspect import Folder
from ft.config import ConfigError, TaskConfig, parse_task
from ft.registry import BaseModelInfo
from ft.spend import Prices, SpendError, budget_left_usd

TEST_CAP = 5000  # a given test set is kept whole up to this size, and capped above it
FRONTIER = {"model": "gpt-6-luna", "reasoning_effort": "low"}
FEW_SHOT_PROMPT_EXAMPLES = 100  # roughly how many examples the few-shot prompt holds in total
MACRO_F1_TOP_SHARE = 0.4
PASS_BAR_GAP = {"classify": 2.0, "extract": 3.0}
CONCURRENCY = {"text": 16, "image": 8}
MAX_ATTEMPTS = 3
MIN_BUDGET_USD = 10


def max_run_usd() -> float:
    """The most one agent-planned run may be estimated to cost."""
    return float(os.environ.get("FT_AGENT_MAX_RUN_USD", "15"))


def few_shot_per_label(n_labels: int) -> int:
    """Examples per label in the few-shot baseline: up to 3, fewer when there are many labels so the
    prompt stays around 100 examples (1 per label for BANKING77's 77 or LEDGAR's 100)."""
    return max(1, min(3, FEW_SHOT_PROMPT_EXAMPLES // max(n_labels, 1)))


def default_baselines(task_type: str, n_labels: int) -> list[dict[str, Any]]:
    """One zero-shot and one few-shot frontier baseline for classification (the comparison a customer
    makes); zero-shot only for extraction (few-shot isn't supported there)."""
    if task_type == "classify":
        return [dict(FRONTIER), {**FRONTIER, "few_shot": few_shot_per_label(n_labels)}]
    return [dict(FRONTIER)]


@dataclass
class RuleResult:
    raw: dict[str, Any]
    corrections: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    estimate: dict[str, Any] | None = None
    cfg: TaskConfig | None = None


def apply_rules(
    raw: dict[str, Any],
    folder: Folder,
    prices: Prices,
    models: dict[str, BaseModelInfo],
    runs_root: Path | None = None,
) -> RuleResult:
    """Correct the mechanical choices in a draft (a parsed task.yaml mapping) and check the cost limits.
    Returns the corrected mapping; `errors` are reasons to reject it."""
    res = RuleResult(raw=_copy(raw))
    try:
        cfg = parse_task(res.raw, where="draft task.yaml")
    except ConfigError as e:
        res.errors += str(e).splitlines()
        return res
    r, fix = res.raw, res.corrections
    kind, ttype = cfg.task.input.kind, cfg.task.type

    try:
        by_hint, rows = split_rows(cfg, folder)
    except (EstimateError, OSError, ValueError):
        by_hint, rows = {}, []  # sources the estimate can't read (mail): only the data-free rules apply

    # base model: the smallest deployable model for the input kind, the next one as fallback
    train = r.setdefault("train", {})
    provider = train.get("provider", "together")
    ok = sorted(
        (m for m in models.values() if m.provider == provider and m.deployable and m.accepts(kind)),
        key=lambda m: m.size_b,
    )
    if not ok:
        res.errors.append(f"no deployable model serves {kind} inputs: decline with fit='no_available_model'")
        return res
    want = [m.id for m in ok[:2]]
    if train.get("base_model") != want[0]:
        fix.append(f"train.base_model: {train.get('base_model')} -> {want[0]} (the smallest deployable {kind} model)")
        train["base_model"] = want[0]
    if list(train.get("fallback_models") or []) != want[1:]:
        fix.append(f"train.fallback_models: {train.get('fallback_models') or []} -> {want[1:]} (the next larger model)")
        train["fallback_models"] = want[1:]

    # baselines: zero-shot plus few-shot for classification
    labels = _labels(cfg, by_hint, rows)
    want_b = default_baselines(ttype, len(labels) if labels else 1)
    have_b = [{k: v for k, v in b.items() if v is not None} for b in r.get("baselines") or []]
    if have_b != want_b:
        fix.append(f"baselines: {_brief(have_b)} -> {_brief(want_b)} (the default comparison)")
        r["baselines"] = want_b

    # test size: keep a given test set whole up to TEST_CAP rows, cap it above
    splits = r.setdefault("splits", {})
    given = len(by_hint.get("test", [])) if cfg.splits.use_predefined and not cfg.splits.by_time else 0
    want_t = min(given, TEST_CAP) if given else min(cfg.splits.test_max, TEST_CAP)
    if cfg.splits.test_max != want_t:
        why = (
            f"the given test set has {given:,} rows" + (f", capped at {TEST_CAP:,}" if given > TEST_CAP else "")
            if given
            else f"carved test sets are capped at {TEST_CAP:,}"
        )
        fix.append(f"splits.test_max: {cfg.splits.test_max} -> {want_t} ({why})")
        splits["test_max"] = want_t

    # metric
    ev = r.setdefault("eval", {})
    metric = _metric(cfg, by_hint, rows)
    if metric and cfg.eval.primary_metric != metric[0]:
        fix.append(f"eval.primary_metric: {cfg.eval.primary_metric} -> {metric[0]} ({metric[1]})")
        ev["primary_metric"] = metric[0]
    bar = {"compare_to": "best_baseline", "max_gap_points": PASS_BAR_GAP[ttype]}
    if (ev.get("pass_bar") or {}) != bar:
        fix.append(f"eval.pass_bar -> within {bar['max_gap_points']} points of the best baseline")
        ev["pass_bar"] = bar
    if cfg.eval.concurrency != CONCURRENCY[kind]:
        fix.append(f"eval.concurrency: {cfg.eval.concurrency} -> {CONCURRENCY[kind]}")
        ev["concurrency"] = CONCURRENCY[kind]
    budget = r.setdefault("budget", {})
    if cfg.budget.max_attempts != MAX_ATTEMPTS:
        fix.append(f"budget.max_attempts: {cfg.budget.max_attempts} -> {MAX_ATTEMPTS}")
        budget["max_attempts"] = MAX_ATTEMPTS

    # cost: estimate the corrected plan; reject it over the limits, else set the budget to ~2x the estimate
    try:
        res.cfg = cfg = parse_task(r, where="corrected task.yaml")
    except ConfigError as e:
        res.errors += str(e).splitlines()
        return res
    try:
        est = res.estimate = estimate_run(cfg, folder, prices, models)
    except (EstimateError, SpendError) as e:
        res.errors.append(f"no cost estimate, so the cost limits can't be checked: {e}")
        return res
    total, limit = est["total_usd"], max_run_usd()
    if total > limit:
        biggest = max(est["lines"], key=lambda x: x["usd"])
        res.errors.append(
            f"estimated ${total:.2f} is over the ${limit:.0f} limit for one run; the largest part is "
            f"{biggest['item']} (${biggest['usd']}). Lower train.max_examples if training dominates."
        )
    if runs_root is not None:
        left = budget_left_usd(runs_root)
        if total > left:
            res.errors.append(f"estimated ${total:.2f} is more than the ${left:.2f} left in the all-runs budget")
    want_usd = max(MIN_BUDGET_USD, math.ceil(2 * total))
    if cfg.budget.max_spend_usd != want_usd:
        fix.append(
            f"budget.max_spend_usd: {cfg.budget.max_spend_usd:g} -> {want_usd} (about twice the ${total} estimate)"
        )
        budget["max_spend_usd"] = want_usd
        res.cfg = parse_task(r, where="corrected task.yaml")
        est.update(budget_max_spend_usd=want_usd, fits_budget=total <= want_usd)
    return res


def _labels(cfg: TaskConfig, by_hint: dict, rows: list[dict]) -> list[str]:
    if cfg.task.type != "classify":
        return []
    if isinstance(cfg.task.output.labels, list):
        return cfg.task.output.labels
    train = by_hint.get("train", []) + by_hint.get(None, []) or rows
    return sorted({str(x.get(cfg.task.output.field, "")) for x in train} - {""})


def _metric(cfg: TaskConfig, by_hint: dict, rows: list[dict]) -> tuple[str, str] | None:
    if cfg.task.type == "extract":
        return "field_accuracy", "extraction is scored per field"
    train = by_hint.get("train", []) + by_hint.get(None, []) or rows
    values = [str(x.get(cfg.task.output.field, "")) for x in train]
    values = [v for v in values if v]
    if not values:
        return None  # labels not in the raw rows (mail folders): the agent's choice stands
    share = Counter(values).most_common(1)[0][1] / len(values)
    if share > MACRO_F1_TOP_SHARE:
        return "macro_f1", f"the most common label is {share:.0%} of train rows, over {MACRO_F1_TOP_SHARE:.0%}"
    return "accuracy", f"the most common label is {share:.0%} of train rows, at most {MACRO_F1_TOP_SHARE:.0%}"


def _brief(baselines: list[dict[str, Any]]) -> str:
    def one(b: dict[str, Any]) -> str:
        return f"{b.get('model')} {b.get('reasoning_effort') or ''}".strip() + (
            f" {b['few_shot']}-shot" if b.get("few_shot") else " zero-shot"
        )

    return "[" + ", ".join(one(b) for b in baselines) + "]"


def _copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy(v) for v in obj]
    return obj
