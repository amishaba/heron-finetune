"""evaluate: one scorer for every predictions file (baselines and tuned), so comparisons are fair (spec 9.10)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ft.config import LOGGED_SELF_AGREEMENT, stable_hash
from ft.io import read_json, read_jsonl, write_json
from ft.spend import SpendError, tuned_cost_per_1k
from ft.stages import StageResult, upstream
from ft.stages.split import category_warnings
from ft.stats import bootstrap_ci, percentile
from ft.tasks.base import ScoreRow, TaskType

NAME = "evaluate"
RETEST_DIR = "logged-retest"  # under baselines/: the logged model re-asked on val (baseline.retest_logged)

COLLAPSE_MIN_SHARE = 0.5  # at least half the outputs are one identical string...
COLLAPSE_MARGIN = 0.25  # ...and that's far more often than the most common target occurs
EMPTY_SERVING_SHARE = 0.5  # more empty answers than this: the endpoint, not the model, is the problem
DIAG_CLASSES = 15
DIAG_CONFUSIONS = 8
DIAG_EXAMPLE_CHARS = 200
DIAG_LOSS_POINTS = 30


def score_rows(ctx, split: str, pred_path: Path, predictions: list[dict[str, Any]] | None = None) -> list[ScoreRow]:
    """Join a predictions file (or its rows, when already read) to its split. Missing predictions count as
    invalid (wrong)."""
    preds = {p["id"]: p for p in (predictions if predictions is not None else read_jsonl(pred_path))}
    rows = []
    for r in read_jsonl(ctx.data_dir / f"{split}.jsonl"):
        p = preds.get(r["id"])
        rows.append(
            ScoreRow(
                id=r["id"],
                input=r["input"],
                target=r["target"],
                parsed=p["prediction_parsed"] if p else None,
                valid=bool(p and p["valid"]),
            )
        )
    return rows


def collapse_check(outputs: list[str | None], targets: list[Any]) -> dict[str, Any]:
    """Flags a model that gives (nearly) the same output whatever the input, e.g. a fine-tune that
    ignores its image and emits an empty template. A skewed but honest classifier (most targets are one
    label) isn't flagged: the top output has to be much more common than the top target."""
    n = len(outputs)
    if n == 0:
        return {"collapsed": False}
    norm = [" ".join((o or "").split()) for o in outputs]
    top, count = Counter(norm).most_common(1)[0]
    keyed = [t if isinstance(t, str) else json.dumps(t, sort_keys=True) for t in targets]
    top_target = Counter(keyed).most_common(1)[0][1] / n if keyed else 0.0
    share = count / n
    collapsed = n >= 10 and share >= COLLAPSE_MIN_SHARE and share >= top_target + COLLAPSE_MARGIN
    out = {
        "collapsed": collapsed,
        "top_output_share": round(share, 3),
        "top_target_share": round(top_target, 3),
        "distinct_outputs": len(set(norm)),
        "top_output": top[:200],
    }
    if collapsed:
        out["message"] = (
            f"collapsed: the model is ignoring its input ({share:.0%} of {n} outputs are the same string, "
            f"while the most common correct answer is {top_target:.0%})"
        )
    return out


def compute_metrics(ctx, task: TaskType, split: str, pred_path: Path, run_meta: dict[str, Any]) -> dict[str, Any]:
    cfg = ctx.cfg
    predictions = read_jsonl(pred_path)
    rows = score_rows(ctx, split, pred_path, predictions)
    scorer = task.scorer(rows)
    primary = cfg.eval.primary_metric
    fn = scorer.metric_fn(primary)
    value = fn(np.arange(scorer.n))
    lo, hi = bootstrap_ci(fn, scorer.n, seed=cfg.splits.seed)
    latencies = [p["latency_ms"] for p in predictions]
    wall = run_meta.get("wall_seconds") or 0
    return {
        "split": split,
        "n": scorer.n,
        "primary_metric": primary,
        "primary": round(value, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        **scorer.details(),
        "collapse": collapse_check([p["prediction_raw"] for p in predictions], [r.target for r in rows]),
        "latency_ms": _latency(latencies),
        "throughput_rph": round(len(latencies) / wall * 3600, 1) if wall else None,
        "wall_seconds": round(wall, 2),
        "concurrency": run_meta.get("concurrency"),
        "simulated_timing": run_meta.get("simulated", False),
        **({"truth": truth_metrics(ctx, task, split, pred_path)} if cfg.targets.from_ == "logs" else {}),
    }


def truth_metrics(ctx, task: TaskType, split: str, pred_path: Path | None) -> dict[str, Any] | None:
    """Logs mode: accuracy against the truth sidecar, for the tuned model (pred_path) or, with pred_path None,
    for the logged model itself. None when the split has no truth labels. Evaluation only (spec 15)."""
    records = [r for r in read_jsonl(ctx.data_dir / f"{split}.jsonl") if "truth" in r.get("meta", {})]
    if not records:
        return None
    if pred_path is None:
        preds = {r["id"]: {"prediction_parsed": task.parse(r["target"], structured=False).value,
                           "valid": True} for r in records}  # fmt: skip
    else:
        preds = {p["id"]: p for p in read_jsonl(pred_path)}
    rows = [
        ScoreRow(id=r["id"], input=r["input"], target=r["meta"]["truth"],
                 parsed=(preds.get(r["id"]) or {}).get("prediction_parsed"),
                 valid=bool((preds.get(r["id"]) or {}).get("valid")))
        for r in records
    ]  # fmt: skip
    scorer = task.scorer(rows)
    fn = scorer.metric_fn("accuracy")
    lo, hi = bootstrap_ci(fn, scorer.n, seed=ctx.cfg.splits.seed)
    return {"n": scorer.n, "accuracy": round(fn(np.arange(scorer.n)), 2), "ci95": [round(lo, 2), round(hi, 2)]}


def logged_metrics(ctx, task: TaskType, split: str) -> dict[str, Any]:
    """The logged frontier model as a pseudo-baseline (spec 15): agreement 100 by definition, truth accuracy
    when there's a sidecar, latency and cost per 1,000 requests from the logged usage."""
    records = read_jsonl(ctx.data_dir / f"{split}.jsonl")
    lat = [r["meta"]["latency_ms"] for r in records if r.get("meta", {}).get("latency_ms") is not None]
    model = next((r["meta"].get("logged_model") for r in records if r.get("meta", {}).get("logged_model")), None)
    cost = None
    try:
        price = ctx.prices.token_price(model) if model else None
    except SpendError:  # a model missing from prices.yaml: show no cost rather than a wrong one
        price = None
    if price is not None and records:
        usd = sum(
            price.cost(u.get("input_tokens", 0), u.get("cache_read_tokens", 0), u.get("output_tokens", 0))
            for u in (r["meta"].get("usage") or {} for r in records)
        )
        cost = round(usd / len(records) * 1000, 4)
    return {
        "model": model,
        "n": len(records),
        "primary": 100.0,
        "ci95": [100.0, 100.0],
        "accuracy": 100.0,
        "macro_f1": 100.0,
        "invalid_rate": 0.0,
        "latency_ms": _latency(lat),
        "cost_per_1k": cost,
        "truth": truth_metrics(ctx, task, split, None),
    }


def _latency(ms: list[float]) -> dict[str, float | None]:
    return {"p50": _round(percentile(ms, 50), 1), "p95": _round(percentile(ms, 95), 1)}


def _round(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else round(x, digits)


def baseline_val_scores(ctx) -> dict[str, float]:
    out = {}
    for b in ctx.cfg.baselines:
        p = ctx.baseline_dir(b.name) / "metrics.val.json"
        if p.exists():
            out[b.name] = read_json(p)["primary"]
    return out


def pass_check(ctx, cfg, val_value: float) -> dict[str, Any]:
    """Tuned val score vs the pass bar (spec 13). Uses val only; test is never used for decisions."""
    bar = cfg.eval.pass_bar
    if not bar.is_relative:
        return {
            "passed": val_value >= bar.min_value,
            "threshold": bar.min_value,
            "rule": f"≥ {bar.min_value}",
            "value": val_value,
        }
    if bar.compare_to == LOGGED_SELF_AGREEMENT:
        ref_model = "the logged model's self-agreement"
        ref = read_json(ctx.baseline_dir(RETEST_DIR) / "metrics.val.json")["primary"]
    else:
        scores = baseline_val_scores(ctx)
        ref_model = max(scores, key=scores.get) if bar.compare_to == "best_baseline" else bar.compare_to
        ref = scores[ref_model]
    threshold = ref - bar.max_gap_points
    return {
        "passed": val_value >= threshold,
        "threshold": round(threshold, 2),
        "reference_model": ref_model,
        "reference_value": ref,
        "rule": f"within {bar.max_gap_points} points of {ref_model}",
        "value": val_value,
    }


def input_hash(ctx) -> str:
    n = ctx.attempt
    predicted, baselines = upstream(ctx, "predict", attempt=n), upstream(ctx, "baseline")
    return stable_hash([predicted, baselines, ctx.cfg.eval.model_dump(), upstream(ctx, "label")])


def run(ctx) -> StageResult:
    n = ctx.attempt
    cfg = ctx.cfg_for(n)
    task = ctx.task_type(n)
    adir = ctx.attempt_dir(n)
    endpoint = read_json(adir / "endpoint.json")
    outputs = []
    metrics = {}
    retrain = ctx.is_retrain(n)
    for split in ctx.eval_splits(n):
        meta = read_json(adir / f"{split}.run.json")
        m = compute_metrics(ctx, task, split, adir / f"{split}.predictions.jsonl", meta)
        rate = endpoint.get("hourly_rate_usd")
        m["cost"] = {
            "hourly_rate_usd": rate,
            "cost_per_1k_full_utilization": _round(tuned_cost_per_1k(rate, m["throughput_rph"] or 0)) if rate else None,
            "endpoint_minutes_billed": endpoint.get("minutes_billed"),
            "endpoint_usd": endpoint.get("usd"),
        }
        metrics[split] = m
    if retrain:  # no val of its own: the pass/fail decision stays with the attempt it retrains
        p = adir / "metrics.test.json"
        write_json(p, metrics["test"])
        ctx.events.emit("retrain_result", stage=NAME, attempt=n, test=metrics["test"]["primary"])
        return StageResult(f"test {metrics['test']['primary']} (retrained on train + val)", [p])
    metrics["val"]["pass"] = pass_check(ctx, cfg, metrics["val"]["primary"])
    for split, m in metrics.items():
        p = adir / f"metrics.{split}.json"
        write_json(p, m)
        outputs.append(p)
    diag = diagnosis(n, metrics["val"], details=diagnosis_details(ctx, n, cfg))
    write_json(adir / "diagnosis.json", diag)
    outputs.append(adir / "diagnosis.json")
    for flag in diag["flags"]:
        ctx.events.emit("warning", stage=NAME, attempt=n, message=flag)
    pc = metrics["val"]["pass"]
    ctx.events.emit(
        "attempt_result",
        stage=NAME,
        attempt=n,
        val=metrics["val"]["primary"],
        passed=pc["passed"],
        threshold=pc["threshold"],
    )
    verdict = "PASS" if pc["passed"] else "FAIL"
    return StageResult(
        f"val {metrics['val']['primary']} vs bar {pc['threshold']} → {verdict}",
        outputs,
        {"passed": pc["passed"], "val": metrics["val"]["primary"]},
    )


def diagnosis(attempt: int, val: dict[str, Any], details: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the M4 fixer sees: val only (spec 13: test is never shown to the fixer). Flags are
    plain-language problems that point at a cause rather than a tuning knob."""
    flags = []
    if val.get("collapse", {}).get("collapsed"):
        flags.append(val["collapse"]["message"])
    if (val.get("invalid_rate") or 0) > 20:
        flags.append(f"invalid outputs: {val['invalid_rate']:.0f}% of val predictions don't parse")
    empty = (details or {}).get("empty_share") or 0
    serving = empty > EMPTY_SERVING_SHARE
    if serving:
        flags.append(
            f"empty answers: {empty:.0%} of val calls returned no answer text (often hidden thinking or a "
            "serving setting using up the token budget). That's a serving problem; no training fix addresses it."
        )
    keep = ("primary_metric", "primary", "ci95", "accuracy", "macro_f1", "invalid_rate", "json_valid_rate",
            "collapse", "pass")  # fmt: skip
    out = {"attempt": attempt, "val": {k: val[k] for k in keep if k in val}, "flags": flags, "serving_problem": serving}
    # spec 13: per-class or per-field table, top confusions with examples, invalid-output examples
    if val.get("per_class"):
        out["worst_classes"] = val["per_class"][:DIAG_CLASSES]  # sorted worst first by the scorer
    if val.get("per_field"):
        out["per_field"] = val["per_field"]
    if val.get("confusions"):
        out["confusions"] = [
            {**c, "examples": [e[:DIAG_EXAMPLE_CHARS] for e in c.get("examples", [])[:2]]}
            for c in val["confusions"][:DIAG_CONFUSIONS]
        ]
    if val.get("field_errors"):
        out["field_errors"] = val["field_errors"][:DIAG_CONFUSIONS]
    return {**out, **(details or {})}


def diagnosis_details(ctx, n: int, cfg) -> dict[str, Any]:
    """The rest of spec 13's diagnosis: invalid-output examples, the training loss curve, and dataset
    stats (train vs val only; thin and late-arriving classes)."""
    adir = ctx.attempt_dir(n)
    preds = read_jsonl(adir / "val.predictions.jsonl")
    inputs = {r["id"]: r["input"] for r in read_jsonl(ctx.data_dir / "val.jsonl")}
    invalid = [
        {
            "input": str(inputs.get(p["id"], ""))[:DIAG_EXAMPLE_CHARS],
            "output": str(p["prediction_raw"])[:DIAG_EXAMPLE_CHARS],
        }
        for p in preds
        if not p["valid"]
    ][:5]
    training = read_json(adir / "training.json") if (adir / "training.json").exists() else {}
    curve = training.get("loss_curve") or []
    if len(curve) > DIAG_LOSS_POINTS:
        step = len(curve) / DIAG_LOSS_POINTS
        curve = [curve[int(i * step)] for i in range(DIAG_LOSS_POINTS)] + [curve[-1]]
    prep = read_json(adir / "prepare.json") if (adir / "prepare.json").exists() else {}
    train = read_jsonl(ctx.data_dir / "train.jsonl")
    data: dict[str, Any] = {"train_rows_available": len(train), "train_rows_used": prep.get("rows")}
    if cfg.task.type == "classify":
        counts = Counter(str(r["target"]) for r in train)
        val_counts = Counter(str(r["target"]) for r in read_jsonl(ctx.data_dir / "val.jsonl"))
        values = sorted(counts.values())
        data.update(
            labels=len(counts),
            train_per_label={"min": values[0], "median": float(np.median(values)), "max": values[-1]},
            warnings=category_warnings(counts, val_counts, by_time=bool(cfg.splits.by_time)),
        )
    answered = [p for p in preds if not p.get("error")]
    empty = sum(1 for p in answered if not (p.get("prediction_raw") or "").strip()) / max(len(answered), 1)
    return {"invalid_examples": invalid, "loss_curve": curve, "data": data, "empty_share": round(empty, 4)}
