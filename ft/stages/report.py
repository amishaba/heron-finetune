"""report: report.html (single file, inline CSS) + report.json with the same numbers (spec 9.11).

All numbers are test-set numbers except the attempts table, which shows val.
"""

from __future__ import annotations

import subprocess
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ft.context import REPO_ROOT
from ft.events import now_iso
from ft.io import atomic_write_text, read_json, write_json
from ft.spend import break_even_requests_per_day
from ft.stages import StageResult
from ft.stages.evaluate import logged_metrics, score_rows, truth_metrics
from ft.stats import paired_diff_ci, verdict_phrase, verdict_word

NAME = "report"


def input_hash(ctx) -> str:
    return uuid.uuid4().hex  # always regenerate; it's cheap (a timestamp can repeat within one second)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _short(model: str) -> str:
    return model.split("/")[-1]


def build(ctx) -> dict[str, Any]:
    cfg = ctx.cfg
    final = ctx.state.data.get("final_attempt") or ctx.best_attempt()
    fcfg = ctx.cfg_for(final) if final else cfg
    task = ctx.task_type(final) if final else None
    # The shipped model: the val-chosen attempt, or its retrain on train + val when one finished
    # (M4 design note 1). Pass/fail always comes from the val-chosen attempt.
    retrained = ctx.state.data.get("retrained_attempt")
    shown = final
    if (
        retrained
        and final
        and (ctx.attempt_dir(retrained) / "metrics.test.json").exists()
        and ctx.attempt_origin(retrained).get("base_attempt") == final
    ):
        shown = retrained
    tuned_label = (
        f"tuned {_short(fcfg.train.base_model)} (attempt {final}"
        + (f", retrained on train + val as attempt {shown})" if shown != final else ")")
    ) if final else None  # fmt: skip
    primary = cfg.eval.primary_metric
    splits_meta = read_json(ctx.data_dir / "splits.json")
    n_test = splits_meta["sizes"]["test"]

    # ---- comparison table (test)
    rows: list[dict[str, Any]] = []
    tuned_m = read_json(ctx.attempt_dir(shown) / "metrics.test.json") if final else None
    tuned_rate = tuned_m["cost"]["hourly_rate_usd"] if tuned_m else None
    for b in cfg.baselines:
        m = read_json(ctx.baseline_dir(b.name) / "metrics.test.json")
        cpr = m["cost"]["cost_per_request"]
        be = break_even_requests_per_day(tuned_rate, cpr) if tuned_rate and cpr else None
        rows.append(_row(_baseline_label(b), "baseline", m, m["cost"]["cost_per_1k"], be))
    logged = None
    if cfg.targets.from_ == "logs" and task is not None:
        logged = logged_metrics(ctx, task, "test")
        row = _row(
            f"logged {logged['model'] or 'model'} (pseudo-baseline)", "logged", logged, logged["cost_per_1k"], None
        )
        rows.append({**row, "truth_accuracy": (logged["truth"] or {}).get("accuracy")})
    if tuned_m:
        rows.append(
            _row(
                tuned_label,
                "tuned",
                tuned_m,
                tuned_m["cost"]["cost_per_1k_full_utilization"],
                None,
            )
        )
        rows[-1]["truth_accuracy"] = (tuned_m.get("truth") or {}).get("accuracy")

    # ---- collapsed models (same output whatever the input), test split
    collapsed = []
    for b in cfg.baselines:
        c = read_json(ctx.baseline_dir(b.name) / "metrics.test.json").get("collapse", {})
        if c.get("collapsed"):
            collapsed.append({"model": _baseline_label(b), **c})
    if tuned_m and tuned_m.get("collapse", {}).get("collapsed"):
        collapsed.append({"model": tuned_label, **tuned_m["collapse"]})

    # ---- verdict (paired bootstrap on test, wording rule in spec 9.11)
    verdict = _verdict(ctx, final, fcfg, task, primary, n_test, shown) if final else None

    # ---- attempts (val)
    attempts = []
    decisions = defaultdict(list)
    for e in ctx.events.of_type("decision"):
        decisions[e.get("attempt")].append(e.get("data", {}).get("rationale", ""))
    for n in ctx.evaluated_attempts():
        mv = ctx.val_metrics(n)
        mt_path = ctx.attempt_dir(n) / "metrics.test.json"
        origin = ctx.attempt_origin(n)
        patch_path = ctx.attempt_dir(n) / "task.patch.yaml"
        attempts.append(
            {
                "n": n,
                "patch": patch_path.read_text().strip() if patch_path.exists() else "(as written)",
                "base_model": ctx.cfg_for(n).train.base_model,
                "val": mv["primary"],
                "threshold": mv["pass"]["threshold"],
                "passed": mv["pass"]["passed"],
                "reasoning": " ".join(decisions.get(n, [])) or None,
                "final": n == final,
                "test": read_json(mt_path)["primary"] if mt_path.exists() else None,
                "by_hand": origin.get("by") == "human",
                "by_fixer": origin.get("by") == "fixer",
                "built_on": origin.get("base_attempt"),
                "changes": origin.get("changes"),
                "designed_after_test": bool(origin.get("designed_after_test")),
            }
        )
    stops = ctx.events.of_type("retry_stopped")
    retry_stopped = stops[-1]["data"].get("reason") if stops else None

    after_test = [a["n"] for a in attempts if a["designed_after_test"]]
    if verdict and final in after_test:
        verdict["caveat"] = (
            f"Attempt {final} was designed by hand after seeing test results, so its test numbers are not a "
            "clean held-out estimate."
        )

    # ---- spend
    by_stage: dict[str, float] = defaultdict(float)
    by_provider: dict[str, float] = defaultdict(float)
    for e in ctx.state.data["spend"]:
        by_stage[e["stage"]] += e["usd"]
        by_provider[e["provider"]] += e["usd"]
    total = sum(by_stage.values())

    interventions = ctx.events.of_type("human_intervention")
    agent_rationale = [
        e.get("data", {}).get("rationale") for e in ctx.events.of_type("decision") if e.get("attempt") is None
    ]
    training = read_json(ctx.attempt_dir(shown) / "training.json") if final else {}

    return {
        "generated_at": now_iso(),
        "dry_run": ctx.dry_run,
        "task": {
            "name": cfg.name,
            "description": cfg.description,
            "type": cfg.task.type,
            "input_kind": cfg.task.input.kind,
            "instruction": cfg.task.instruction,
            "n_labels": len(ctx.labels()) if cfg.task.type == "classify" else None,
        },
        "primary_metric": primary,
        "concurrency": cfg.eval.concurrency,
        "pass_bar": cfg.eval.pass_bar.model_dump(exclude_none=True),
        "verdict": verdict,
        "comparison": rows,
        "attempts": attempts,
        "retry_stopped": retry_stopped,
        "truth_mode": bool(logged and logged.get("truth")),
        "retrain": _retrain_info(ctx, final, shown),
        "collapsed": collapsed,
        "errors": {
            "per_class": (tuned_m or {}).get("per_class", [])[:25],
            "confusions": (tuned_m or {}).get("confusions", []),
            "per_field": (tuned_m or {}).get("per_field", []),
            "field_errors": (tuned_m or {}).get("field_errors", [])[:8],
        },
        "setup": {
            "label_source": cfg.targets.from_,
            "splits": splits_meta["sizes"],
            "split_method": splits_meta["method"],
            "duplicates_dropped": splits_meta.get("duplicates_dropped", {}),
            "label_variants_merged": splits_meta.get("label_variants_merged", {}),
            "rows_dropped_at_load": _load_drops(ctx),
            "warnings": splits_meta.get("warnings", []),
            "base_model": fcfg.train.base_model,
            "epochs": fcfg.train.epochs,
            "learning_rate": fcfg.train.learning_rate,
            "max_examples": fcfg.train.max_examples,
            "baselines": [b.model_dump(exclude_none=True) for b in cfg.baselines],
            "agent_rationale": [r for r in agent_rationale if r],
            "loss_curve": training.get("loss_curve"),
        },
        "spend": {
            "total_usd": round(total, 4),
            "budget_usd": ctx.spend.cap_usd,
            "by_stage": {k: round(v, 4) for k, v in by_stage.items()},
            "by_provider": {k: round(v, 4) for k, v in by_provider.items()},
            "simulated": ctx.dry_run,
        },
        "privacy": _privacy(ctx, splits_meta),
        "interventions": {
            "count": len(interventions),
            "items": [{"ts": e["ts"], **e.get("data", {})} for e in interventions],
        },
        "metadata": {
            "run_id": ctx.run_id,
            "git_commit": _git_commit(),
            "created_at": ctx.state.data["created_at"],
            "final_attempt": final,
            "task_updates": [{"ts": e["ts"], **e.get("data", {})} for e in ctx.events.of_type("task_updated")],
            "external_ids": {n: a.get("external", {}) for n, a in ctx.state.data["attempts"].items()},
            "endpoints": ctx.state.data["endpoints"],
        },
    }


def _baseline_label(b) -> str:
    """e.g. 'gpt-6-luna (low, 3-shot)'."""
    bits = [b.reasoning_effort or "default effort"] + ([f"{b.few_shot}-shot"] if b.few_shot else [])
    return f"{b.model} ({', '.join(bits)})"


def _privacy(ctx, splits_meta) -> dict[str, Any] | None:
    """What left this machine and where (spec 14.5). Only for runs whose data went through redaction."""
    p = ctx.data_dir / "redaction.json"
    if not p.exists():
        return None
    red = read_json(p)
    sizes = splits_meta["sizes"]
    sent = []
    for b in ctx.cfg.baselines:
        extra = " + your context file" if b.context_file else ""
        shots = f" + {b.few_shot} train examples per label" if b.few_shot else ""
        sent.append(f"OpenAI ({b.model}): {sizes['val'] + sizes['test']:,} val/test messages{shots}{extra}")
    trainer = ctx.cfg.train.provider.capitalize()
    sent.append(f"{trainer}: training file with {sizes['train']:,} messages (attempt files under attempts/<n>/)")
    sent.append(f"{trainer} endpoint: val/test messages at prediction time")
    return {"redaction": red, "sent": sent}


def _load_drops(ctx) -> dict[str, int]:
    p = ctx.data_dir / "load_stats.json"
    return {k: v for k, v in read_json(p).get("dropped", {}).items() if v} if p.exists() else {}


def _row(model: str, kind: str, m: dict[str, Any], cost_per_1k: float | None, break_even: float | None):
    return {
        "model": model,
        "kind": kind,
        "primary": m["primary"],
        "ci95": m["ci95"],
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "invalid_rate": m.get("invalid_rate"),
        "p50_ms": m["latency_ms"]["p50"],
        "p95_ms": m["latency_ms"]["p95"],
        "throughput_rph": m.get("throughput_rph"),
        "cost_per_1k": cost_per_1k,
        "break_even_per_day": None if break_even is None else round(break_even),
    }


def _headline(val_pass: dict[str, Any], primary: str, labels: dict[str, str]) -> str:
    """Pass/fail against the bar comes first (val; the retry loop's decision), then the test comparison."""
    word = "Passed" if val_pass["passed"] else "Failed"
    ref = val_pass.get("reference_model")
    why = (
        f"within {val_pass['rule'].split('within ', 1)[-1].split(' of ')[0]} of {labels.get(ref, ref)}, "
        f"which scored {val_pass['reference_value']:.1f} on val"
        if ref
        else val_pass["rule"]
    )
    return f"{word} the bar: val {primary} {val_pass['value']:.1f}, needed {val_pass['threshold']:.1f} ({why})."


def _verdict(ctx, final: int, fcfg, task, primary: str, n_test: int, shown: int | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    shown = shown or final
    labels = {b.name: _baseline_label(b) for b in cfg.baselines}
    val_pass = ctx.val_metrics(final)["pass"]
    tuned_rows = score_rows(ctx, "test", ctx.attempt_dir(shown) / "test.predictions.jsonl")
    fn_tuned = task.scorer(tuned_rows).metric_fn(primary)
    tuned_val = fn_tuned(np.arange(len(tuned_rows)))
    tuned_name = f"Tuned {_short(fcfg.train.base_model)}" + (" (retrained on train + val)" if shown != final else "")
    unit = "examples"
    out: dict[str, Any] = {
        "passed_val": val_pass["passed"],
        "val_rule": val_pass["rule"],
        "val_value": val_pass["value"],
        "val_threshold": val_pass["threshold"],
        "tuned_test": round(tuned_val, 2),
        "headline": _headline(val_pass, primary, labels),
    }
    if not cfg.baselines:
        out["sentence"] = f"{tuned_name} scored {tuned_val:.1f} {primary} on {n_test:,} test {unit}."
        if cfg.targets.from_ == "logs":
            t_tuned = truth_metrics(ctx, task, "test", ctx.attempt_dir(shown) / "test.predictions.jsonl")
            t_logged = logged_metrics(ctx, task, "test")["truth"]
            if t_tuned and t_logged:
                out["truth"] = {"tuned": t_tuned, "logged": t_logged}
                out["sentence"] = (
                    f"{tuned_name} agreed with the logged model's answers on {tuned_val:.1f}% of {n_test:,} test "
                    f"{unit}. Against the truth labels it scored {t_tuned['accuracy']:.1f}% accuracy, vs "
                    f"{t_logged['accuracy']:.1f}% for the logged model."
                )
        return out

    # Paired bootstrap against every baseline; the sentence uses the pass bar's reference.
    comparisons = []
    for b in cfg.baselines:
        base_rows = score_rows(ctx, "test", ctx.baseline_dir(b.name) / "test.predictions.jsonl")
        fn_base = task.scorer(base_rows).metric_fn(primary)
        point, lo, hi = paired_diff_ci(fn_tuned, fn_base, len(tuned_rows), seed=cfg.splits.seed)
        comparisons.append(
            {
                "baseline": b.name,
                "label": labels[b.name],
                "baseline_test": round(fn_base(np.arange(len(base_rows))), 2),
                "diff": round(point, 2),
                "diff_ci95": [round(lo, 2), round(hi, 2)],
                "word": verdict_word(point, lo, hi),
                "phrase": verdict_phrase(verdict_word(point, lo, hi), point),
            }
        )
    by_name = {c["baseline"]: c for c in comparisons}
    ref = val_pass.get("reference_model") or max(by_name, key=lambda n: by_name[n]["baseline_test"])
    c = by_name[ref]
    point, (lo, hi), word, ref_label = c["diff"], c["diff_ci95"], c["word"], labels[ref]
    tail = verdict_phrase(word, point, ref_label)
    tail = tail if word in ("beat", "fell short", "tied") else f"was {tail}"
    out.update(
        word=word,
        reference_model=ref,
        reference_test=c["baseline_test"],
        diff=point,
        diff_ci95=[lo, hi],
        comparisons=comparisons,
        sentence=(
            f"{tuned_name} scored {tuned_val:.1f} vs {ref_label}'s {c['baseline_test']:.1f} {primary} on "
            f"{n_test:,} test {unit}: it {tail} (difference {point:+.2f} points, "
            f"95% CI [{lo:+.2f}, {hi:+.2f}])."
        ),
    )
    return out


def render_html(data: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(REPO_ROOT / "ft" / "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["usd"] = lambda v: "—" if v is None else (f"${v:,.4f}" if abs(v) < 1 else f"${v:,.2f}")
    env.filters["num"] = lambda v, d=1: "—" if v is None else f"{v:,.{d}f}"
    return env.get_template("report.html.j2").render(r=data)


def run(ctx) -> StageResult:
    data = build(ctx)
    write_json(ctx.run_dir / "report.json", data)
    out = ctx.run_dir / "report.html"
    atomic_write_text(out, render_html(data))
    return StageResult(f"report at {out}", [out, ctx.run_dir / "report.json"])


def _retrain_info(ctx, final: int | None, shown: int | None) -> dict[str, Any] | None:
    if not final:
        return None
    if shown != final:
        prep = read_json(ctx.attempt_dir(shown) / "prepare.json")
        return {"attempt": shown, "base_attempt": final, "rows": prep.get("rows"),
                "test": read_json(ctx.attempt_dir(shown) / "metrics.test.json")["primary"],
                "base_test": read_json(ctx.attempt_dir(final) / "metrics.test.json")["primary"]}  # fmt: skip
    skips = ctx.events.of_type("retrain_skipped")
    return {"skipped": skips[-1]["data"]["reason"]} if skips else None
