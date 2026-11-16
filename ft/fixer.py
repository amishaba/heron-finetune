"""The retry loop's fixer (spec 13): after a failed attempt, pick fixes from a closed menu; code turns them
into a config patch for the next attempt.

Same agent model as the planner (gpt-6-luna, medium effort), separate prompt. It sees val only: the failed
attempt's diagnosis.json, every earlier attempt's config, fixes and val result, and the limits. It never
sees test numbers, and its patch can only touch training options (never the test set, metric, pass bar
or baselines). The next attempt builds on the best attempt so far by val, not necessarily the last one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ft.config import ConfigError, TaskConfig, apply_patch
from ft.io import read_json
from ft.providers.base import AgentLLM, AgentTurn, ToolCall
from ft.spend import estimate_endpoint_usd, estimate_training_usd
from ft.stages.serve import hourly_rate

MAX_EPOCHS = 6  # spec 13
# A practical range on top of the provider's own (Together allows 1e-8 to 1.0 for Qwen3.5-4B).
LR_FLOOR, LR_CEILING = 1e-6, 1e-3
MAX_TURNS = 6

MENU = {
    "more_epochs": "train.epochs += value (a whole number >= 1), at most {max_epochs} epochs in total",
    "change_learning_rate": "train.learning_rate = value, between {lr_lo:g} and {lr_hi:g}",
    "use_more_data": "raise train.max_examples to value, or remove the cap with value null (only when capped)",
    "bigger_model": "train on the next model in train.fallback_models (value null)",
    "labels_in_prompt": "classify only: list the labels in the tuned model's prompt (value null)",
    "class_weighting": "classify only: repeat rare classes in training; value 'sqrt' or 'balanced'",
    "add_examples": "not available: needs frontier labeling (targets.frontier_model), which isn't built yet",
    "drop_suspect_labels": "not available: needs frontier second opinions on train labels, not built yet",
}
UNAVAILABLE = {"add_examples", "drop_suspect_labels"}

TOOLS: list[dict[str, Any]] = [
    {"name": "submit_fixes", "description": "Fixes for the next attempt, from the menu, each with a one-line "
     "reason tied to the diagnosis. Code validates them; errors come back to you.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["fixes"], "properties": {
         "fixes": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                              "required": ["fix", "value", "reason"], "properties": {
             "fix": {"type": "string", "enum": list(MENU)},
             "value": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]},
             "reason": {"type": "string"}}}}}}},
    {"name": "stop_retrying", "description": "End the retry loop: no fix on the menu is likely to reach the bar "
     "(say why). The best attempt so far becomes the final model.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["reason"],
                    "properties": {"reason": {"type": "string"}}}},
]  # fmt: skip

SYSTEM = """\
You are the fixer in an automatic fine-tuning pipeline. A small model was fine-tuned and missed the pass bar
on the validation set. You get its diagnosis (validation only) and the history of earlier attempts, and you
choose what to change for the next attempt from a fixed menu. Code applies your choice; you can't change the
test set, the metric, the pass bar or the baselines.

Read the evidence before choosing:
- The learning rate and loss curve: a loss that barely falls points at a learning rate far too low or too
  few epochs (LoRA on these models usually works at 1e-4 to 3e-4). A loss near zero with poor validation
  points at overfitting or a data problem, not more epochs.
- Training rows used vs available: a capped, small training set is the first thing to lift.
- Worst classes and confusions: thin or late-arriving classes (data warnings) can't be learned from a few
  examples; labels_in_prompt or class_weighting may help them, more epochs won't.
- Flags such as "collapsed" or a high invalid-output rate point at a broken setup rather than tuning.
- Earlier attempts: don't repeat a fix that already failed; the next attempt builds on the best one so far.
Combine fixes when the evidence supports several (e.g. a learning rate far too low and a capped training
set). Prefer the cheapest fixes that address the cause. Call stop_retrying when no menu item is likely to
reach the bar, for example when the gap is large and every relevant fix has been tried.

MENU (fix: what code does)
{menu}

Finish with exactly one call: submit_fixes or stop_retrying."""


class FixError(ValueError):
    pass


@dataclass
class FixDecision:
    stop: bool
    reason: str = ""
    fixes: list[dict[str, Any]] = field(default_factory=list)
    patch: dict[str, Any] = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)
    base_attempt: int | None = None
    estimate_usd: float | None = None
    usd: float = 0.0


def limits_for(ctx, base_model: str) -> dict[str, float]:
    try:
        lim = ctx.providers.trainer.limits(base_model)
    except Exception:  # limits are a convenience; fall back to the practical range
        lim = {}
    return {
        "lr_lo": max(LR_FLOOR, lim.get("min_learning_rate", LR_FLOOR)),
        "lr_hi": min(LR_CEILING, lim.get("max_learning_rate", LR_CEILING)),
        "max_epochs": int(min(MAX_EPOCHS, lim.get("max_epochs", MAX_EPOCHS))),
    }


def build_patch(
    cfg: TaskConfig, base: TaskConfig, base_patch: dict[str, Any], fixes: list[dict[str, Any]],
    limits: dict[str, float], train_rows: int,
) -> tuple[dict[str, Any], list[str]]:  # fmt: skip
    """Turn menu fixes into the next attempt's patch: the base attempt's patch plus the changes. Only
    `train.*` can change. Raises FixError with every problem, so the fixer can correct itself."""
    if not fixes:
        raise FixError("give at least one fix, or call stop_retrying")
    errors, changes = [], []
    train = dict(base_patch.get("train") or {})
    t = base.train
    seen = set()
    for f in fixes:
        name, value = f.get("fix"), f.get("value")
        if name in seen:
            errors.append(f"{name} given twice")
            continue
        seen.add(name)
        if name in UNAVAILABLE:
            errors.append(f"{name} is not available in this build")
        elif name == "more_epochs":
            if not isinstance(value, (int, float)) or value < 1 or int(value) != value:
                errors.append("more_epochs needs a whole number >= 1")
            elif t.epochs + int(value) > limits["max_epochs"]:
                errors.append(f"more_epochs: {t.epochs} + {int(value)} is over the {limits['max_epochs']}-epoch limit")
            else:
                train["epochs"] = t.epochs + int(value)
                changes.append(f"epochs {t.epochs} -> {train['epochs']}")
        elif name == "change_learning_rate":
            if not isinstance(value, (int, float)) or not limits["lr_lo"] <= value <= limits["lr_hi"]:
                errors.append(
                    f"change_learning_rate needs a number between {limits['lr_lo']:g} and {limits['lr_hi']:g}"
                )
            elif math.isclose(value, t.learning_rate):
                errors.append(f"the learning rate is already {t.learning_rate:g}")
            else:
                train["learning_rate"] = float(value)
                changes.append(f"learning rate {t.learning_rate:g} -> {value:g}")
        elif name == "use_more_data":
            cap = t.max_examples
            if not cap or cap >= train_rows:
                errors.append(f"use_more_data: training already uses every available row ({train_rows:,})")
            elif value is not None and (not isinstance(value, (int, float)) or value <= cap):
                errors.append(f"use_more_data needs a number above the current cap {cap:,}, or null for no cap")
            else:
                new = None if value is None or value >= train_rows else int(value)
                train["max_examples"] = new
                changes.append(
                    f"max_examples {cap:,} -> {'all ' + format(train_rows, ',') if new is None else f'{new:,}'}"
                )
        elif name == "bigger_model":
            chain = [cfg.train.base_model, *cfg.train.fallback_models]
            i = chain.index(t.base_model) if t.base_model in chain else len(chain)
            if i + 1 >= len(chain):
                errors.append("bigger_model: no larger model left in train.fallback_models")
            else:
                train["base_model"] = chain[i + 1]
                changes.append(f"base model {t.base_model} -> {chain[i + 1]}")
        elif name == "labels_in_prompt":
            if cfg.task.type != "classify":
                errors.append("labels_in_prompt is for classify tasks")
            elif t.labels_in_prompt:
                errors.append("labels_in_prompt is already on")
            else:
                train["labels_in_prompt"] = True
                changes.append("label list in the tuned prompt")
        elif name == "class_weighting":
            if cfg.task.type != "classify":
                errors.append("class_weighting is for classify tasks")
            elif value not in ("sqrt", "balanced"):
                errors.append("class_weighting needs 'sqrt' or 'balanced'")
            elif t.class_weighting == value:
                errors.append(f"class_weighting is already {value}")
            else:
                train["class_weighting"] = value
                changes.append(f"class weighting {t.class_weighting} -> {value}")
        else:
            errors.append(f"unknown fix {name!r}")
    if errors:
        raise FixError("; ".join(errors))
    patch = {**{k: v for k, v in base_patch.items() if k != "train"}, "train": train}
    try:
        apply_patch(cfg, patch)
    except ConfigError as e:
        raise FixError(f"the patch doesn't validate: {e}") from e
    return patch, changes


def payload(ctx, n: int, base: int, limits: dict[str, float]) -> dict[str, Any]:
    """Everything the fixer reads. Val only."""
    diag = read_json(ctx.attempt_dir(n) / "diagnosis.json")
    history = []
    for m in ctx.evaluated_attempts():
        val = ctx.val_metrics(m)
        origin = ctx.attempt_origin(m) or {"by": "initial config"}
        tcfg = ctx.cfg_for(m).train
        history.append({
            "attempt": m,
            "by": origin.get("by"),
            "built_on": origin.get("base_attempt"),
            "fixes": origin.get("fixes"),
            "train": {"base_model": tcfg.base_model, "epochs": tcfg.epochs, "learning_rate": tcfg.learning_rate,
                      "max_examples": tcfg.max_examples, "labels_in_prompt": tcfg.labels_in_prompt,
                      "class_weighting": tcfg.class_weighting},
            "val": val["primary"],
            "passed": val["pass"]["passed"],
        })  # fmt: skip
    cfg = ctx.cfg
    return {
        "goal": {"metric": cfg.eval.primary_metric, **{k: diag["val"]["pass"].get(k) for k in ("threshold", "rule")}},
        "failed_attempt": n,
        "next_attempt_builds_on": base,
        "attempts_left": cfg.budget.max_attempts - n,
        "budget_left_usd": round(ctx.spend.remaining(), 2),
        "diagnosis": diag,
        "history": history,
        "limits": {
            "learning_rate": [limits["lr_lo"], limits["lr_hi"]],
            "max_epochs": limits["max_epochs"],
            "fallback_models": cfg.train.fallback_models,
        },  # fmt: skip
    }


def fixer_llm(dry_run: bool) -> AgentLLM:
    if dry_run:
        return FakeFixer()
    from ft.providers.openai_provider import OpenAIAgent

    return OpenAIAgent()


def decide(ctx, n: int, llm: AgentLLM | None = None) -> FixDecision:
    """Ask the fixer what to change after failed attempt n. Records its spend; returns a validated patch
    or a stop. A fixer that can't produce a valid choice in MAX_TURNS stops the loop."""
    llm = llm or fixer_llm(ctx.dry_run)
    base = ctx.best_attempt() or n
    base_cfg = ctx.cfg_for(base)
    limits = limits_for(ctx, base_cfg.train.base_model)
    data = payload(ctx, n, base, limits)
    train_rows = data["diagnosis"]["data"]["train_rows_available"]
    menu = "\n".join(f"- {k}: {v.format(**limits)}" for k, v in MENU.items())
    price = None if llm.simulated else ctx.prices.token_price(llm.model)
    if not llm.simulated:
        ctx.spend.check("fix", "openai", 0.05, f"fixer ({llm.model}) after attempt {n}", attempt=n)
    llm.start(SYSTEM.replace("{menu}", menu), json.dumps(data, default=str), TOOLS)
    usd = 0.0
    try:
        for turn_no in range(1, MAX_TURNS + 1):
            turn = llm.step()
            if price is not None:
                usd += price.cost(turn.input_tokens, turn.cached_tokens, turn.output_tokens)
            if not turn.calls:
                llm.nudge("Call submit_fixes or stop_retrying.")
                continue
            results = []
            for call in turn.calls:
                if call.name == "stop_retrying":
                    return FixDecision(stop=True, reason=str(call.args.get("reason", "")), base_attempt=base, usd=usd)
                if call.name != "submit_fixes":
                    results.append((call.call_id, json.dumps({"error": f"unknown tool {call.name}"})))
                    continue
                fixes = call.args.get("fixes") or []
                try:
                    patch, changes = build_patch(ctx.cfg, base_cfg, ctx.attempt_patch(base), fixes, limits, train_rows)
                except FixError as e:
                    results.append((call.call_id, json.dumps({"accepted": False, "error": str(e)})))
                    continue
                return FixDecision(stop=False, fixes=fixes, patch=patch, changes=changes, base_attempt=base, usd=usd)
            llm.add_results(results)
            ctx.log(f"fixer turn {turn_no}: choice rejected, asking again")
        return FixDecision(stop=True, reason=f"the fixer gave no valid choice in {MAX_TURNS} turns", base_attempt=base,
                           usd=usd)  # fmt: skip
    finally:
        if usd:
            ctx.spend.record("fix", "openai", usd, f"fixer after attempt {n}", attempt=n, model=llm.model)


def estimate_attempt_usd(ctx, base: int, patch: dict[str, Any]) -> float:
    """Training plus endpoint for the next attempt, the way the train and serve guards will estimate them
    (errs high): training from the base attempt's tokens scaled to the new row count and epochs, at least
    the provider's minimum and the base job's billed cost scaled the same way; the endpoint as `deployed`
    estimates it."""
    cfg = apply_patch(ctx.cfg, patch)
    base_cfg = ctx.cfg_for(base)
    bdir = ctx.attempt_dir(base)
    prep = read_json(bdir / "prepare.json")
    rows_avail = sum(1 for _ in (ctx.data_dir / "train.jsonl").open())
    rows_base = prep.get("rows") or rows_avail
    rows_next = min(rows_avail, cfg.train.max_examples or rows_avail)
    scale = rows_next / max(rows_base, 1)
    model = ctx.models[cfg.train.base_model]
    train = estimate_training_usd(
        int((prep.get("approx_tokens") or 0) * scale), cfg.train.epochs, ctx.prices.training_per_mtok(model.id),
        ctx.prices.training_min_charge(model.id),
    )  # fmt: skip
    billed = (read_json(bdir / "training.json").get("usd") or 0) if (bdir / "training.json").exists() else 0
    # A bill at the provider's minimum says nothing about usage, so only a bill above it is scaled.
    if billed > ctx.prices.training_min_charge(model.id) and cfg.train.base_model == base_cfg.train.base_model:
        train = max(train, billed * scale * cfg.train.epochs / base_cfg.train.epochs)
    n_eval = sum(1 for s in ("val", "test") for _ in (ctx.data_dir / f"{s}.jsonl").open())
    eval_minutes = ctx.task_type().eval_minutes(n_eval, cfg.eval.concurrency)
    return round(train + estimate_endpoint_usd(eval_minutes, hourly_rate(ctx, cfg.train.base_model)), 2)


def write_attempt(ctx, nxt: int, decision: FixDecision) -> None:
    ctx.write_attempt(
        nxt, decision.patch, by="fixer", base_attempt=decision.base_attempt, fixes=decision.fixes,
        changes=decision.changes, estimate_usd=decision.estimate_usd, designed_after_test=False,
    )  # fmt: skip
    for f in decision.fixes:
        ctx.events.emit("decision", attempt=nxt, rationale=f"fixer: {f['fix']} ({f.get('value')}): {f['reason']}")


class FakeFixer:
    """Deterministic stand-in for dry runs: fixes the obvious cause first (learning rate far off, too few
    epochs, capped data), then a bigger model, then stops."""

    model = "fake-fixer"
    simulated = True

    def start(self, system: str, user: str, tools: list[dict[str, Any]]) -> None:
        self.data = json.loads(user)
        self.tried = 0

    def step(self) -> AgentTurn:
        d = self.data
        cur = next(h for h in d["history"] if h["attempt"] == d["next_attempt_builds_on"])["train"]
        lo, hi = d["limits"]["learning_rate"]
        fixes = []
        if cur["learning_rate"] < 5e-5 or cur["learning_rate"] > 5e-4:
            fixes.append(
                {"fix": "change_learning_rate", "value": min(max(1e-4, lo), hi), "reason": "learning rate far off"}
            )
        if cur["epochs"] < 3:
            fixes.append({"fix": "more_epochs", "value": 3 - cur["epochs"], "reason": "too few epochs"})
        used, avail = d["diagnosis"]["data"]["train_rows_used"], d["diagnosis"]["data"]["train_rows_available"]
        if cur["max_examples"] and (used or 0) < avail:
            fixes.append({"fix": "use_more_data", "value": None, "reason": f"only {used} of {avail} rows used"})
        if not fixes:
            fixes.append({"fix": "bigger_model", "value": None, "reason": "config looks sane; try more capacity"})
        self.tried += 1
        if self.tried > 1:  # the first choice was rejected (e.g. no bigger model): stop
            return AgentTurn(calls=[ToolCall("s", "stop_retrying", {"reason": "no menu fix left to try"})])
        return AgentTurn(calls=[ToolCall("f", "submit_fixes", {"fixes": fixes})])

    def add_results(self, results) -> None:
        pass

    def nudge(self, text: str) -> None:
        pass
