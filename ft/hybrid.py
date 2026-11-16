"""Hybrid routing, simulated for free on a finished classification run's saved predictions (NOTES.md,
future work). No API calls.

Rule: the tuned model answers every input. When its answer is a label whose val F1 (the tuned model's,
on val) is below a threshold, or it gives no valid label, the input goes to a fallback baseline instead,
and that baseline's saved answer is used. The threshold is picked on val only (best val primary metric;
ties go to the lower threshold, i.e. less routing), then applied once to test.

Cost per 1k: the tuned model sees every input, so its cost per 1k at full utilization, plus what the
fallback's routed calls were actually billed. Routed traffic is a small share of the fallback's run, so
its prompt-cache hit rate in production could be lower than in the batch run; the report says so.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ft.io import read_json, read_jsonl
from ft.stats import paired_diff_ci
from ft.tasks.base import ScoreRow
from ft.tasks.classify import ClassifyScorer

NO_LABEL = "(invalid)"


class HybridError(ValueError):
    pass


@dataclass
class Joined:
    """One split, rows aligned by id: target, tuned answer, fallback answer and its billed cost."""

    ids: list[str]
    inputs: list[str]
    targets: list[str]
    tuned: list[tuple[str | None, bool]]
    fallback: list[tuple[str | None, bool]]
    fallback_usd: list[float]


def _join(run_dir: Path, attempt: int, fallback: str, split: str) -> Joined:
    records = read_jsonl(run_dir / "data" / f"{split}.jsonl")
    tuned = {p["id"]: p for p in read_jsonl(run_dir / "attempts" / str(attempt) / f"{split}.predictions.jsonl")}
    fb = {p["id"]: p for p in read_jsonl(run_dir / "baselines" / fallback / f"{split}.predictions.jsonl")}
    missing = [r["id"] for r in records if r["id"] not in tuned or r["id"] not in fb]
    if missing:
        raise HybridError(f"{split}: {len(missing)} records lack a tuned or {fallback} prediction")
    return Joined(
        ids=[r["id"] for r in records],
        inputs=[str(r["input"]) for r in records],
        targets=[str(r["target"]) for r in records],
        tuned=[(tuned[r["id"]]["prediction_parsed"], bool(tuned[r["id"]]["valid"])) for r in records],
        fallback=[(fb[r["id"]]["prediction_parsed"], bool(fb[r["id"]]["valid"])) for r in records],
        fallback_usd=[float(fb[r["id"]].get("usd") or 0.0) for r in records],
    )


def label_f1(targets: list[str], answers: list[tuple[str | None, bool]]) -> dict[str, float]:
    """F1 (points) per label the model predicted, from one split. Invalid answers count as NO_LABEL (F1 0)."""
    preds = [a if ok else NO_LABEL for a, ok in answers]
    tp = Counter(t for t, p in zip(targets, preds, strict=True) if t == p)
    true_n, pred_n = Counter(targets), Counter(preds)
    out = {}
    for label in pred_n:
        prec, rec = tp[label] / pred_n[label], (tp[label] / true_n[label] if true_n[label] else 0.0)
        out[label] = 0.0 if label == NO_LABEL or prec + rec == 0 else 200 * prec * rec / (prec + rec)
    return out


def route(answers: list[tuple[str | None, bool]], f1: dict[str, float], threshold: float) -> np.ndarray:
    """True where the input goes to the fallback. An invalid answer counts as a label with F1 0, and a label
    never predicted on val has no F1, so counts as 0 too; threshold 0 routes nothing."""
    return np.array([f1.get(str(a) if ok else NO_LABEL, 0.0) < threshold for a, ok in answers], dtype=bool)


def _scorer(j: Joined, answers: list[tuple[str | None, bool]], labels: list[str]) -> ClassifyScorer:
    rows = [
        ScoreRow(id=i, input=x, target=t, parsed=a if ok and a in labels else None, valid=bool(ok and a in labels))
        for i, x, t, (a, ok) in zip(j.ids, j.inputs, j.targets, answers, strict=True)
    ]
    return ClassifyScorer(rows, labels)


def _mix(j: Joined, routed: np.ndarray) -> list[tuple[str | None, bool]]:
    return [f if r else t for t, f, r in zip(j.tuned, j.fallback, routed, strict=True)]


def simulate(run_dir: Path, fallback: str | None = None, attempt: int | None = None) -> dict[str, Any]:
    state = read_json(run_dir / "state.json")
    attempt = attempt or state.get("final_attempt")
    if not attempt:
        raise HybridError("the run has no final attempt")
    adir = run_dir / "attempts" / str(attempt)
    tuned_test = read_json(adir / "metrics.test.json")
    metric = tuned_test["primary_metric"]
    if metric not in ("accuracy", "macro_f1"):
        raise HybridError(f"hybrid routing is simulated for classification runs, not {metric}")
    if fallback is None:  # the few-shot baseline if there is one (the customer's comparison), else the first
        names = sorted(p.name for p in (run_dir / "baselines").iterdir() if (p / "test.predictions.jsonl").exists())
        fallback = next((n for n in names if n.endswith("shot")), names[0] if names else None)
    if not fallback:
        raise HybridError("no baseline predictions to fall back to")
    labels = read_json(run_dir / "data" / "labels.json")
    val, test = (_join(run_dir, attempt, fallback, s) for s in ("val", "test"))

    # pick the threshold on val
    f1 = label_f1(val.targets, val.tuned)
    candidates = [0.0] + sorted({v + 1e-6 for v in f1.values()})
    curve = []
    for t in candidates:
        routed = route(val.tuned, f1, t)
        score = _scorer(val, _mix(val, routed), labels).metric_fn(metric)(np.arange(len(val.ids)))
        curve.append((t, score, float(routed.mean())))
    threshold, val_score, val_share = max(curve, key=lambda c: (round(c[1], 9), -c[0]))

    # apply it once to test
    routed = route(test.tuned, f1, threshold)
    hybrid_answers = _mix(test, routed)
    s_t, s_f, s_h = (_scorer(test, a, labels) for a in (test.tuned, test.fallback, hybrid_answers))
    n, idx = len(test.ids), np.arange(len(test.ids))
    tuned_per_1k = (tuned_test.get("cost") or {}).get("cost_per_1k_full_utilization")
    fb_per_1k = sum(test.fallback_usd) / n * 1000
    routed_per_1k = sum(u for u, r in zip(test.fallback_usd, routed, strict=True) if r) / n * 1000

    def row(s: ClassifyScorer, cost: float | None) -> dict[str, Any]:
        return {"accuracy": round(s._accuracy(idx), 2), "macro_f1": round(s._macro_f1(idx), 2),
                "cost_per_1k": None if cost is None else round(cost, 4)}  # fmt: skip

    def diff(a: ClassifyScorer, b: ClassifyScorer) -> list[float]:
        return [round(x, 2) for x in paired_diff_ci(a.metric_fn(metric), b.metric_fn(metric), n)]

    return {
        "run": run_dir.name,
        "attempt": attempt,
        "fallback": fallback,
        "primary_metric": metric,
        "threshold_val_f1": round(threshold, 2),
        "val": {"hybrid_primary": round(val_score, 2), "routed_share": round(val_share, 4),
                "tuned_primary": round(_scorer(val, val.tuned, labels).metric_fn(metric)(np.arange(len(val.ids))), 2)},
        "labels_routed": sorted(lbl for lbl, v in f1.items() if v < threshold),
        "test": {
            "n": n,
            "tuned": row(s_t, tuned_per_1k),
            "fallback": row(s_f, fb_per_1k),
            "hybrid": row(s_h, None if tuned_per_1k is None else tuned_per_1k + routed_per_1k),
            "routed_share": round(float(routed.mean()), 4),
            "hybrid_minus_tuned": diff(s_h, s_t),  # [point, lo, hi] in the primary metric
            "hybrid_minus_fallback": diff(s_h, s_f),
        },
        "val_curve": [{"threshold": round(t, 2), "primary": round(v, 2), "routed_share": round(sh, 4)}
                      for t, v, sh in curve],
        # For reading only, never used to choose: test at fixed thresholds.
        "test_at_thresholds": [_at(test, f1, t, labels, metric) for t in (25.0, 50.0, 75.0)],
        "cost_note": "tuned at full endpoint utilization (every input); routed inputs at the fallback's billed "
        "cost in the batch run (cache hit rate for sparse routed traffic may be lower)",
    }  # fmt: skip


def _at(j: Joined, f1: dict[str, float], t: float, labels: list[str], metric: str) -> dict[str, Any]:
    routed = route(j.tuned, f1, t)
    score = _scorer(j, _mix(j, routed), labels).metric_fn(metric)(np.arange(len(j.ids)))
    return {"threshold": t, "primary": round(score, 2), "routed_share": round(float(routed.mean()), 4)}
