"""The planning agent (spec 12): look at a data folder, decide whether fine-tuning fits, write task.yaml.

Agent decides, code executes: the agent's only outputs are a validated task.yaml (with a one-line
rationale per decision) or a decline with a reason and a suggestion. It runs in the run folder it will
belong to: transcript in agent/, decisions in events.jsonl, spend in plan.json (moved into state.json
when the run starts).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ft.agent.inspect import Folder
from ft.agent.tools import RATIONALE_KEYS, TOOLS, AgentTools, Outcome
from ft.events import EventLog, now_iso
from ft.providers.base import AgentLLM, AgentTurn, ToolCall  # noqa: F401  (re-exported)
from ft.registry import BaseModelInfo
from ft.spend import Prices, spend_entry, total_budget_usd, total_real_spend

MAX_TURNS = 30
NUDGES = 2


def max_agent_usd() -> float:
    return float(os.environ.get("FT_AGENT_MAX_USD", "1.0"))


@dataclass
class PlanResult:
    outcome: str  # submitted | declined | gave_up
    task_yaml: str | None = None
    rationale: dict[str, str] = field(default_factory=dict)
    reason: str | None = None
    suggestion: str | None = None
    fit: str | None = None  # declines: "not_a_fit" | "no_available_model" (tools.DECLINE_FITS)
    fit_label: str | None = None  # the same, in words
    corrections: list[str] = field(default_factory=list)  # fixed rules code applied to the submitted draft
    turns: int = 0
    usd: float = 0.0


SYSTEM = """\
You are the planning agent of an automatic fine-tuning pipeline. You look at one data folder and decide
whether a small fine-tuned model can do the task as well as a frontier model, more cheaply. If yes, you
write the task.yaml the pipeline will run with no human edits. You never train anything yourself; code
does that from your task.yaml.

How to work: call get_profile first. Look at real rows with sample_rows and value_counts (and view_images
for image data). Draft a task.yaml, then call validate_task and estimate_cost and fix every error. Finish
with exactly one of submit_task (with a one-line rationale per decision) or decline.

FIXED RULES (code enforces these; validate_task and estimate_cost apply them and report each correction,
and submit_task applies them again, so you can't override them, only reject the plan by declining)
- base_model: the smallest deployable model for the input kind; fallback_models: the next larger one.
- baselines: for classify, gpt-6-luna at low effort zero-shot, plus the same with few_shot: N examples per
  label drawn from train (N = 3, fewer when there are many labels, so the prompt holds about 100 examples);
  for extract, the zero-shot one only.
- splits.test_max: a given test set is kept whole up to 5,000 rows and capped at 5,000 above that; carved
  test sets are capped at 5,000.
- primary_metric: accuracy when the most common label is at most 40% of train rows, macro_f1 when it's more;
  field_accuracy for extract.
- pass_bar: {compare_to: best_baseline, max_gap_points: 2.0} for classify, 3.0 for extract. concurrency: 16
  for text, 8 for images. max_attempts: 3.
- cost: the estimate (after corrections) must be at most $15 and fit in what's left of the all-runs budget,
  or the plan is rejected. budget.max_spend_usd is set to about twice the estimate, at least 10.
Write your rationale for the plan as corrected (the validate_task result shows the corrected task_yaml).

FIT RULES
Good fit: classify, route, extract, structured output, fixed format; text or images; the right answers are
already recorded in the data (labels, outcomes, folders); outputs can be graded automatically by comparing
with those answers; enough examples: at least 500 labeled rows in total.
Poor fit (decline, and say what would work instead):
- fewer than 500 labeled rows: suggest few-shot prompting a frontier model, or collecting more labels;
- open-ended outputs (long free-form answers, no consistent target that can be checked automatically):
  suggest keeping a frontier model and defining a grading rubric first;
- answers that depend on facts that change: suggest retrieval;
- no recorded answers at all: suggest labeling a sample first (frontier labeling isn't available yet).
All of these are decline(fit="not_a_fit").
A different kind of decline: the data IS a good fit, but list_base_models returns no model for its input kind
(e.g. images). Then decline(fit="no_available_model"): say the task itself suits fine-tuning, that this
provider has no deployable model that serves that input kind, and suggest trying another fine-tuning provider
(meanwhile a frontier model can do the task). Check the data is otherwise a good fit before saying so.

DECISIONS
- task.type: classify when the answer is one label from a fixed set; extract when it's a fixed set of fields.
- targets: {from: dataset} (labels in the data). For mail folders (maildir/mbox) the label is the folder.
- labels: auto (taken from the train split) unless the data comes with a fixed list.
- splits: use_predefined: true when files are named train/val/test; otherwise val/test are carved out of
  the data. Use by_time: <timestamp column> for behavior labels or when the data is a time series of one
  person's or system's choices.
- train.max_examples: training usually dominates the cost on large data. If the estimate is over $15, set
  max_examples (a stratified sample of train) to bring it under, keeping at least 50 examples per label
  where the data has them.
- Paths in task.yaml are relative to the repository root; the data folder is {folder}.

TASK.YAML SHAPE (unknown keys are errors)
name: short-kebab-name              # a-z, 0-9, dashes; at most 40 chars
description: One sentence.
source:                             # pick the one that matches the data
  type: csv                         # or jsonl
  path: {folder}                    # a folder (train.csv / test.csv picked up by name) or one file
  # files: {train: a.csv, val: b.csv, test: c.csv}   # only if names aren't train/val/test
  # type: image_folder, path: <images dir>, labels_file: <csv/jsonl with a file-name column and an
  #   optional split column>
  # type: maildir (or mbox), path, exclude_folders: [inbox, sent, ...], min_per_folder: 20
  # target_map: {column: <json column>, root: <dotted path>, fields: {out_name: dotted.path,
  #   items: {path: list_path, as_list: true, item: {name: nm}}}}   # extract from a JSON column
task:
  type: classify                    # or extract
  input: {kind: text, field: <input column>}   # kind: image for images (field = file-name column)
  output: {field: <label column>, labels: auto}
  # extract: output: {field: target, schema: {total: money, items: {list_of: {name: text, price: money}}}}
  #   field types: text, money, number, date, list_of (at most 15 leaf fields)
  instruction: One or two sentences saying what to predict.
targets: {from: dataset}
splits: {use_predefined: true, val_fraction: 0.1, val_max: 1000, test_fraction: 0.2, test_max: 2000, seed: 42}
baselines:                          # classify; extract has only the first
  - {model: gpt-6-luna, reasoning_effort: low}
  - {model: gpt-6-luna, reasoning_effort: low, few_shot: 1}
train: {provider: together, base_model: <id>, fallback_models: [<id>], lora: true, epochs: 3, learning_rate: 1.0e-4,
        max_examples: null}
eval:
  primary_metric: accuracy
  pass_bar: {compare_to: best_baseline, max_gap_points: 2.0}
  concurrency: 16
budget: {max_spend_usd: 15, max_attempts: 3}
"""


def system_prompt(folder: Folder) -> str:
    return SYSTEM.replace("{folder}", folder.repo_path())


class PlanLog:
    """Everything the plan leaves in the run folder."""

    def __init__(self, run_dir: Path, model: str, simulated: bool):
        self.run_dir = run_dir
        (run_dir / "agent").mkdir(parents=True, exist_ok=True)
        self.events = EventLog(run_dir / "events.jsonl")
        self.transcript = run_dir / "agent" / "transcript.jsonl"
        self.plan_path = run_dir / "plan.json"
        self.model = model
        self.simulated = simulated
        self.spend: list[dict[str, Any]] = []

    def note(self, kind: str, **data: Any) -> None:
        with self.transcript.open("a") as f:
            f.write(json.dumps({"ts": now_iso(), "kind": kind, **data}, default=str) + "\n")

    def add_spend(self, usd: float, detail: str) -> None:
        entry = spend_entry("agent", "openai", usd, detail, simulated=self.simulated, model=self.model)
        self.spend.append({"ts": now_iso(), **entry})
        self.save()

    def save(self, **extra: Any) -> None:
        data = json.loads(self.plan_path.read_text()) if self.plan_path.exists() else {}
        data.update(spend=self.spend, model=self.model, simulated=self.simulated, **extra)
        self.plan_path.write_text(json.dumps(data, indent=2, default=str))


def plan(
    folder: Folder,
    llm: AgentLLM,
    prices: Prices,
    models: dict[str, BaseModelInfo],
    run_dir: Path,
    runs_root: Path,
    *,
    max_turns: int = MAX_TURNS,
    max_usd: float | None = None,
    log: Callable[[str], None] = print,
) -> PlanResult:
    max_usd = max_agent_usd() if max_usd is None else max_usd
    record = PlanLog(run_dir, llm.model, llm.simulated)
    tools = AgentTools(folder, prices, models, runs_root=None if llm.simulated else runs_root)
    price = prices.token_price(llm.model) if not llm.simulated else None
    user = (
        f"Data folder: {folder.repo_path()}. Decide whether fine-tuning fits this data. If it does, submit a "
        "task.yaml; if not, decline with a reason and a suggestion."
    )
    llm.start(system_prompt(folder), user, TOOLS)
    record.events.emit("agent_start", model=llm.model, folder=folder.repo_path(), max_turns=max_turns, max_usd=max_usd)
    usd, nudges, turns = 0.0, 0, 0
    result: PlanResult | None = None
    while turns < max_turns:
        if not llm.simulated:
            overall = total_real_spend(runs_root) + usd
            if usd >= max_usd or overall >= total_budget_usd():
                result = PlanResult("gave_up", reason=f"agent spend limit reached (${usd:.2f} of ${max_usd:.2f})")
                break
        turns += 1
        turn = llm.step()
        cost = 0.0 if price is None else price.cost(turn.input_tokens, turn.cached_tokens, turn.output_tokens)
        usd += cost
        record.add_spend(cost, f"agent turn {turns}: {turn.input_tokens} in / {turn.output_tokens} out")
        if turn.text:
            record.note("assistant_text", turn=turns, text=turn.text)
        if not turn.calls:
            if nudges >= NUDGES:
                result = PlanResult("gave_up", reason="the agent stopped calling tools without submitting or declining")
                break
            nudges += 1
            llm.nudge("Use the tools, and finish with submit_task or decline.")
            continue
        results = []
        for call in turn.calls:
            out = tools.call(call.name, call.args)
            log(f"agent · {call.name}({_brief(call.args)})")
            record.events.emit("agent_tool", tool=call.name, args=_brief(call.args, 300))
            record.note(
                "tool",
                turn=turns,
                name=call.name,
                args=call.args,
                result=out if isinstance(out, str) else {"images": out},
            )
            results.append((call.call_id, out))
        llm.add_results(results)
        if tools.outcome:
            result = _finish(tools.outcome, record, tools.fit_label(tools.outcome.fit))
            break
    else:
        result = PlanResult("gave_up", reason=f"no decision after {max_turns} turns")
    result.turns, result.usd = turns, round(usd, 6)
    if result.outcome == "gave_up":
        record.events.emit("agent_gave_up", reason=result.reason)
    record.save(
        outcome=result.outcome,
        turns=turns,
        usd=round(usd, 6),
        reason=result.reason,
        suggestion=result.suggestion,
        fit=result.fit,
        fit_label=result.fit_label,
    )
    if result.task_yaml:
        (run_dir / "agent" / "task.draft.yaml").write_text(result.task_yaml)
    return result


def _finish(outcome: Outcome, record: PlanLog, fit_label: str) -> PlanResult:
    if outcome.kind == "declined":
        record.events.emit(
            "decision",
            rationale=f"declined ({fit_label}): {outcome.reason}",
            suggestion=outcome.suggestion,
            fit=outcome.fit,
        )
        return PlanResult(
            "declined", reason=outcome.reason, suggestion=outcome.suggestion, fit=outcome.fit, fit_label=fit_label
        )
    for key in RATIONALE_KEYS:
        line = (outcome.rationale.get(key) or "").strip()
        if line:
            record.events.emit("decision", topic=key, rationale=f"{key.replace('_', ' ')}: {line}")
    for fix in outcome.corrections:  # after the agent's lines, so the report shows what code overrode
        record.events.emit("decision", topic="rule", rationale=f"rule applied: {fix}")
    return PlanResult(
        "submitted", task_yaml=outcome.task_yaml, rationale=outcome.rationale, corrections=outcome.corrections
    )


def _brief(args: dict[str, Any], limit: int = 80) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in args.items() if k not in ("task_yaml", "rationale"))
    if "task_yaml" in args:
        s = (s + ", " if s else "") + "task_yaml=…"
    return s[:limit]
