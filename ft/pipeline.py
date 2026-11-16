"""Runs stages in order, skipping any whose outputs exist and whose inputs haven't changed (spec 4, 8)."""

from __future__ import annotations

import time
from types import ModuleType
from typing import Any

from rich.console import Console

from ft import fixer
from ft.config import ConfigError, apply_patch, retrain_with_val, stable_hash
from ft.context import RunContext
from ft.io import file_hash, read_json
from ft.stages import baseline, evaluate, label, load, predict, prepare, profile, report, split, train

FREE_STAGES: list[ModuleType] = [load, profile, split, label]  # local only: nothing is paid for yet
RUN_STAGES: list[ModuleType] = [*FREE_STAGES, baseline]
ATTEMPT_STAGES: list[ModuleType] = [prepare, train, predict, evaluate]
STAGE_NAMES = [m.NAME for m in RUN_STAGES + ATTEMPT_STAGES] + [report.NAME]

TERMS_REMINDER = (
    "Reminder: this run trains on a frontier model's outputs. Check the provider's terms before using "
    "the result beyond personal research (OpenAI restricts using outputs to develop competing models)."
)


def run_stage(ctx: RunContext, mod: ModuleType, attempt: int | None, console: Console) -> None:
    ctx.attempt = attempt
    ctx.log = lambda msg: console.print(f"    [dim]{msg}[/]")
    label_ = mod.NAME if attempt is None else f"{mod.NAME} · a{attempt}"
    h = mod.input_hash(ctx)
    if ctx.state.stage_done(mod.NAME, attempt, h):
        ctx.events.emit("stage_skip", stage=mod.NAME, attempt=attempt)
        console.print(f"  [dim]·[/] {label_:<16} [dim]done, inputs unchanged[/]")
        return
    ctx.events.emit("stage_start", stage=mod.NAME, attempt=attempt)
    ctx.state.mark_stage(mod.NAME, attempt, "running", input_hash=h)
    t0 = time.monotonic()
    try:
        res = mod.run(ctx)
    except BaseException as e:  # also Ctrl-C and SIGTERM: the stage is recorded as failed, then it propagates
        error = f"{type(e).__name__}: {e}"
        ctx.state.mark_stage(mod.NAME, attempt, "failed", error=error)
        ctx.events.emit("stage_error", stage=mod.NAME, attempt=attempt, error=error)
        console.print(f"  [bold]✗[/] {label_:<16} {error}")
        raise
    existing = [p for p in res.outputs if p.exists()]
    out_hash = file_hash(*existing) if existing else stable_hash(res.summary)
    secs = time.monotonic() - t0
    ctx.state.mark_stage(mod.NAME, attempt, "done", input_hash=h, output_hash=out_hash, seconds=round(secs, 2))
    ctx.events.emit("stage_end", stage=mod.NAME, attempt=attempt, summary=res.summary, seconds=round(secs, 2))
    console.print(f"  [bold]✓[/] {label_:<16} {res.summary}")


def run_free_stages(ctx: RunContext, console: Console) -> None:
    """Load through label: what the human checkpoint shows before any paid stage."""
    for mod in FREE_STAGES:
        run_stage(ctx, mod, None, console)


def run_attempt(ctx: RunContext, n: int, console: Console) -> None:
    for mod in ATTEMPT_STAGES:
        run_stage(ctx, mod, n, console)


def _note(console: Console, what: str, text: str, *, done: bool = False) -> None:
    """A pipeline line that isn't a stage (fix, retry, retrain), aligned with the stage lines."""
    console.print(f"  [green]✓[/] {what:<16} {text}" if done else f"  [dim]·[/] {what:<16} [dim]{text}[/]")


def invalidate_from(ctx: RunContext, stage: str) -> None:
    """--from STAGE: re-run that stage and everything after it. Training job IDs are kept, so
    re-running `train` polls the existing job instead of paying for a new one."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown stage '{stage}'; choose from {', '.join(STAGE_NAMES)}")
    later = set(STAGE_NAMES[STAGE_NAMES.index(stage) :])
    keys = [k for k in ctx.state.data["stages"] if k.split("/")[-1] in later]
    ctx.state.invalidate_from(keys)
    ctx.events.emit("invalidate", stage=stage, keys=keys)


def run_pipeline(ctx: RunContext, console: Console, from_stage: str | None = None) -> None:
    if from_stage:
        invalidate_from(ctx, from_stage)
    if ctx.cfg.targets.from_ in ("logs", "frontier"):
        console.print(f"[dim]{TERMS_REMINDER}[/]")

    for mod in RUN_STAGES:
        run_stage(ctx, mod, None, console)

    max_attempts = ctx.cfg.budget.max_attempts
    n = 1
    while True:
        ctx.state.attempt(n)
        ctx.state.save()
        run_attempt(ctx, n, console)
        if ctx.val_metrics(n)["pass"]["passed"]:
            break
        nxt = n + 1
        if ctx.is_retrain(nxt):  # the run already reached its final retrain
            break
        if (ctx.attempt_dir(nxt) / "task.patch.yaml").exists():  # already planned (resume, or by hand)
            n = nxt
            continue
        if nxt > max_attempts:
            _retry_end(
                ctx, console, n, f"attempt {n} missed the bar and budget.max_attempts ({max_attempts}) is used up"
            )
            break
        if not _plan_retry(ctx, console, n, nxt):
            break
        n = nxt

    # Attempts added by hand (`ft attempt`) after the loop ended; finishes them if a run was interrupted.
    for m in sorted(int(k) for k in ctx.state.data["attempts"]):
        if m > n and not ctx.is_retrain(m):
            run_attempt(ctx, m, console)

    best = _pick_final(ctx)
    if retrain_with_val(ctx.cfg):
        _retrain(ctx, console, best)
    run_stage(ctx, report, None, console)


def _pick_final(ctx: RunContext) -> int | None:
    best = ctx.state.data["final_attempt"] = ctx.best_attempt()
    ctx.state.save()
    return best


def _retrain(ctx: RunContext, console: Console, best: int | None) -> None:
    """M4 design note 1: retrain the val-chosen config on train + val; the shipped model and its test
    numbers come from the retrain, the pass/fail decision stays with `best`. Only for a passing pick, and
    only if it fits the budget. Runs once per chosen attempt (resumable)."""
    if best is None:
        return
    done = ctx.state.data.get("retrained_attempt")
    if done and ctx.attempt_origin(done).get("base_attempt") == best:
        run_attempt(ctx, done, console)
        return
    if not ctx.val_metrics(best)["pass"]["passed"]:
        ctx.events.emit("retrain_skipped", attempt=best, reason=f"attempt {best} didn't pass, so nothing ships")
        return
    patch = ctx.attempt_patch(best)
    est = fixer.estimate_attempt_usd(ctx, best, patch)
    left = ctx.spend.remaining()
    if est > left:
        reason = f"retraining (~${est:.2f}) won't fit the ${left:.2f} left in this run's budget"
        ctx.events.emit("retrain_skipped", attempt=best, reason=reason)
        _note(console, "retrain", f"skipped: {reason}")
        return
    r = ctx.next_attempt()
    ctx.write_attempt(
        r, patch, by="retrain", base_attempt=best, data="train+val", estimate_usd=est, designed_after_test=False
    )
    ctx.events.emit(
        "decision", attempt=r, rationale=f"retrain attempt {best}'s config on train + val for the shipped model"
    )
    ctx.state.data["retrained_attempt"] = r
    ctx.state.attempt(r)
    ctx.state.save()
    _note(console, "retrain", f"attempt {r}: attempt {best}'s config on train + val (~${est:.2f})", done=True)
    run_attempt(ctx, r, console)


def _retry_end(ctx: RunContext, console: Console, n: int, reason: str) -> None:
    ctx.events.emit("retry_stopped", attempt=n, reason=reason)
    _note(console, "retry", f"stopped: {reason}")


def _plan_retry(ctx: RunContext, console: Console, n: int, nxt: int) -> bool:
    """Spec 13: the fixer picks fixes from the closed menu after failed attempt n; code writes attempt
    nxt's patch, unless the fixer stops or the next attempt wouldn't fit the run's budget."""
    diag = read_json(ctx.attempt_dir(n) / "diagnosis.json")
    if diag.get("serving_problem"):  # no menu fix can help; don't pay for attempts that can't work
        _retry_end(ctx, console, n, f"attempt {n} hit a serving problem, not a training one: {diag['flags'][-1]}")
        return False
    decision = fixer.decide(ctx, n)
    if decision.stop:
        _retry_end(ctx, console, n, f"the fixer stopped: {decision.reason}")
        return False
    est = fixer.estimate_attempt_usd(ctx, decision.base_attempt, decision.patch)
    decision.estimate_usd = est
    left = ctx.spend.remaining()
    if est > left:
        _retry_end(ctx, console, n, f"attempt {nxt} (~${est:.2f}) won't fit the ${left:.2f} left in this run's budget")
        return False
    fixer.write_attempt(ctx, nxt, decision)
    changes = "; ".join(decision.changes)
    _note(console, "fix", f"attempt {nxt} (from attempt {decision.base_attempt}, ~${est:.2f}): {changes}", done=True)
    return True


def add_attempt(ctx: RunContext, console: Console, patch: dict[str, Any], note: str, after_test: bool) -> int:
    """A hand-designed attempt on an existing run: same data, splits and baselines, config patched.
    Logged as a human intervention; `after_test` marks it in the report as designed after seeing test
    results, so its test numbers aren't a clean holdout. (The M4 fixer will write these patches itself.)"""
    n = ctx.next_attempt()
    if n > ctx.cfg.budget.max_attempts:
        raise ConfigError(f"budget.max_attempts is {ctx.cfg.budget.max_attempts}; attempt {n} would exceed it")
    if not patch:
        raise ConfigError("the patch is empty; an attempt needs at least one config change")
    apply_patch(ctx.cfg, patch)  # validate before touching the run folder
    ctx.write_attempt(n, patch, by="human", note=note, designed_after_test=after_test)
    when = " after seeing test results" if after_test else ""
    ctx.events.emit("human_intervention", attempt=n, what=f"attempt {n} designed by hand{when}: {note}")
    ctx.events.emit("decision", attempt=n, rationale=note)
    ctx.state.attempt(n)
    ctx.state.save()
    run_attempt(ctx, n, console)
    _pick_final(ctx)
    run_stage(ctx, report, None, console)
    return n
