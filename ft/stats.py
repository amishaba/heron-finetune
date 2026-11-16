"""Bootstrap confidence intervals, including the paired difference used for the report's verdict wording."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray], float]  # takes row indices, returns the metric in points


def bootstrap_ci(fn: MetricFn, n: int, *, resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(resamples)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_diff_ci(
    fn_a: MetricFn, fn_b: MetricFn, n: int, *, resamples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """(point, lo, hi) of metric_a − metric_b, resampling the same rows for both."""
    idx_all = np.arange(n)
    point = fn_a(idx_all) - fn_b(idx_all)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        diffs.append(fn_a(idx) - fn_b(idx))
    lo, hi = np.percentile(np.array(diffs), [2.5, 97.5])
    return (float(point), float(lo), float(hi))


def verdict_word(point: float, lo: float, hi: float) -> str:
    """Report wording rule (spec 9.11): "beat" / "fell short" only when the 95% CI of the paired
    difference excludes 0; otherwise "ahead" / "behind" (not statistically significant), or "tied"."""
    if lo > 0:
        return "beat"
    if hi < 0:
        return "fell short"
    if point > 0:
        return "ahead"
    if point < 0:
        return "behind"
    return "tied"


def verdict_phrase(word: str, point: float, other: str | None = None) -> str:
    """'beat X by 4.2 points', 'behind X by 5.3 points, not statistically significant', ..."""
    x = f"{abs(point):.1f} points"
    target = f" {other}" if other else ""
    return {
        "beat": f"beat{target} by {x}",
        "fell short": f"fell short of{target} by {x}",
        "ahead": f"ahead of{target} by {x}, not statistically significant",
        "behind": f"behind{target} by {x}, not statistically significant",
        "tied": f"tied with{target}",
    }[word]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.array(values), q)) if values else None
