"""The planning agent (M3) without network: folder inspection, tools, the loop's limits, the heuristic fake
agent's decisions, spend accounting and the CLI flow with the human checkpoint."""

import csv
import json
import random
from pathlib import Path

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from ft.agent.agent import AgentTurn, ToolCall, plan
from ft.agent.fake import FakeAgent
from ft.agent.inspect import Folder
from ft.agent.tools import AgentTools
from ft.cli import app
from ft.context import REPO_ROOT, RunContext
from ft.registry import load_models
from ft.spend import Prices, total_real_spend

PRICES = Prices.load(REPO_ROOT / "prices.yaml", dry_run=True)
MODELS = load_models(REPO_ROOT / "models.yaml")
FIXTURES = REPO_ROOT / "examples" / "fixtures"
INTENTS = ["lost_card", "refund", "change_address", "close_account", "card_arrival"]


def intents_folder(root: Path, n_train: int = 600, n_test: int = 150) -> Path:
    """A raw folder like BANKING77's: train.csv + test.csv, text + category columns, no task.yaml."""
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for name, n in (("train", n_train), ("test", n_test)):
        with (root / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text", "category"])
            w.writeheader()
            for i in range(n):
                label = INTENTS[i % len(INTENTS)]
                w.writerow({"text": f"customer says {label.replace('_', ' ')} please help, ref {rng.randint(1, 9999)}",
                            "category": label})  # fmt: skip
    return root


def tools_for(folder: Path) -> AgentTools:
    return AgentTools(Folder(folder, REPO_ROOT), PRICES, MODELS)


def run_plan(folder: Path, run_dir: Path, llm=None, **kw):
    return plan(Folder(folder, REPO_ROOT), llm or FakeAgent(), PRICES, MODELS, run_dir, run_dir.parent,
                log=lambda m: None, **kw)  # fmt: skip


# ---------------------------------------------------------------- inspection and tools


def test_profile_sees_tables_splits_json_columns_and_images(tmp_path):
    d = intents_folder(tmp_path / "raw")
    (d / "extra.jsonl").write_text(
        "".join(json.dumps({"img": f"i{i}.png", "gt": json.dumps({"a": i})}) + "\n" for i in range(5))
    )
    Image.new("RGB", (64, 48)).save(d / "i0.png")
    p = Folder(d, tmp_path).profile()
    assert p["tables"]["train.csv"]["rows"] == 600 and p["tables"]["train.csv"]["split_hint"] == "train"
    assert p["tables"]["test.csv"]["split_hint"] == "test"
    assert p["tables"]["extra.jsonl"]["columns"]["gt"]["json"] is True
    assert p["tables"]["train.csv"]["columns"]["category"]["cardinality"] == 5
    assert p["images"]["count"] == 1 and p["images"]["median_size"] == [64, 48]


def test_test_rows_are_held_out(tmp_path):
    t = tools_for(intents_folder(tmp_path / "raw"))
    assert "held out" in t.call("sample_rows", {"file": "test.csv", "n": 3, "where": None})
    assert "held out" in t.call("value_counts", {"file": "test.csv", "column": "category", "top": 5})
    rows = json.loads(
        t.call("sample_rows", {"file": "train.csv", "n": 3, "where": {"column": "category", "equals": "refund"}})
    )
    assert len(rows) == 3 and {r["category"] for r in rows} == {"refund"}
    assert "error" in json.loads(t.call("sample_rows", {"file": "../../etc/passwd", "n": 1, "where": None}))


def _draft(folder: Path, **over) -> str:
    cfg = {
        "name": "intents", "source": {"type": "csv", "path": str(folder)},
        "task": {"type": "classify", "input": {"kind": "text", "field": "text"},
                 "output": {"field": "category", "labels": "auto"}, "instruction": "Classify the message."},
        "targets": {"from": "dataset"}, "splits": {"use_predefined": True, "seed": 42},
        "baselines": [{"model": "gpt-6-luna", "reasoning_effort": "low"}],
        "train": {"base_model": "Qwen/Qwen3.5-4B"},
        "eval": {"primary_metric": "accuracy", "pass_bar": {"compare_to": "best_baseline", "max_gap_points": 2.0}},
        "budget": {"max_spend_usd": 15},
    }  # fmt: skip
    for dotted, v in over.items():
        node = cfg
        *parents, leaf = dotted.split("__")
        for k in parents:
            node = node[k]
        node[leaf] = v
    return yaml.safe_dump(cfg)


def test_validate_checks_the_data_not_just_the_schema(tmp_path):
    d = intents_folder(tmp_path / "raw")
    t = tools_for(d)
    assert t.validate(_draft(d))["valid"]
    bad = t.validate(_draft(d, task__input__field="message"))
    assert not bad["valid"] and "missing columns ['message']" in " ".join(bad["errors"])
    labels = t.validate(_draft(d, task__output__labels=["refund", "lost_card"]))
    assert "not in task.output.labels" in " ".join(labels["errors"])
    model = t.validate(_draft(d, train__base_model="Qwen/Qwen3-VL-8B-Instruct"))  # a rule corrects it
    assert (
        model["valid"] and "train.base_model: Qwen/Qwen3-VL-8B-Instruct -> Qwen/Qwen3.5-4B" in model["corrections"][0]
    )
    broken = t.validate("name: [unclosed")
    assert not broken["valid"] and "not valid YAML" in broken["errors"][0]  # comes back as text to fix


def test_estimate_mirrors_the_split_rules_and_errs_high(tmp_path):
    d = intents_folder(tmp_path / "raw")
    est = json.loads(tools_for(d).call("estimate_cost", {"task_yaml": _draft(d)}))
    assert est["splits"] == {"train": 540, "val": 60, "test": 150}  # val carved (10%) from train, test kept
    assert est["total_usd"] >= 4.0 and est["fits_budget"]  # Together's $4 training minimum
    assert est["total_minutes"] == sum(x["minutes"] for x in est["lines"]) and est["total_minutes"] >= 19


# ---------------------------------------------------------------- fixed rules, enforced in code


def test_rules_correct_mechanical_choices(tmp_path):
    d = intents_folder(tmp_path / "raw")
    out = tools_for(d).validate(
        _draft(d, splits__test_max=100, eval__primary_metric="macro_f1", budget__max_spend_usd=1, eval__concurrency=4)
    )
    assert out["valid"], out
    fixed = yaml.safe_load(out["task_yaml"])
    assert fixed["baselines"] == [  # zero-shot plus few-shot; 5 labels -> 3 examples each
        {"model": "gpt-6-luna", "reasoning_effort": "low"},
        {"model": "gpt-6-luna", "reasoning_effort": "low", "few_shot": 3},
    ]
    assert fixed["splits"]["test_max"] == 150  # the given test set, kept whole
    assert fixed["eval"]["primary_metric"] == "accuracy" and fixed["eval"]["concurrency"] == 16
    assert fixed["budget"]["max_spend_usd"] >= 10 and fixed["train"]["fallback_models"] == ["Qwen/Qwen3.5-9B"]
    assert any("test_max: 100 -> 150" in c for c in out["corrections"])
    assert tools_for(d).validate(out["task_yaml"])["corrections"] == []  # the corrected plan is stable


def test_rules_cap_large_test_sets_and_pick_the_metric_from_the_data(tmp_path):
    d = intents_folder(tmp_path / "raw", n_test=6000)
    with (d / "train.csv").open("a", newline="") as f:  # skew: "refund" becomes ~half of train
        w = csv.writer(f)
        for i in range(500):
            w.writerow([f"refund now {i}", "refund"])
    out = tools_for(d).validate(_draft(d, splits__test_max=6000))
    fixed = yaml.safe_load(out["task_yaml"])
    assert fixed["splits"]["test_max"] == 5000 and fixed["eval"]["primary_metric"] == "macro_f1"
    assert any("capped at 5,000" in c for c in out["corrections"])


def test_rules_reject_plans_over_the_cost_limits(tmp_path, monkeypatch):
    d = intents_folder(tmp_path / "raw")
    monkeypatch.setenv("FT_AGENT_MAX_RUN_USD", "1")
    out = tools_for(d).validate(_draft(d))
    assert not out["valid"] and "over the $1 limit for one run" in " ".join(out["errors"])
    monkeypatch.setenv("FT_AGENT_MAX_RUN_USD", "15")
    runs = tmp_path / "runs"
    (runs / "old").mkdir(parents=True)
    (runs / "old" / "plan.json").write_text(json.dumps({"spend": [{"usd": 98.0, "simulated": False}]}))
    monkeypatch.setenv("FT_TOTAL_BUDGET_USD", "100")
    t = AgentTools(Folder(d, REPO_ROOT), PRICES, MODELS, runs_root=runs)
    out = t.validate(_draft(d))
    assert not out["valid"] and "left in the all-runs budget" in " ".join(out["errors"])


def test_submitted_plan_is_the_corrected_one_and_corrections_are_logged(tmp_path):
    d = intents_folder(tmp_path / "raw")
    run = tmp_path / "runs" / "p"
    res = run_plan(d, run)
    assert res.outcome == "submitted" and res.corrections  # the fake agent writes one baseline
    assert len(yaml.safe_load(res.task_yaml)["baselines"]) == 2
    events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    assert any(e["data"].get("rationale", "").startswith("rule applied: baselines") for e in events)


def test_image_tasks_need_a_model_that_serves_images(tmp_path):
    t = tools_for(intents_folder(tmp_path / "raw"))
    assert json.loads(t.call("list_base_models", {"kind": "image"}))["models"] == []
    text = json.loads(t.call("list_base_models", {"kind": "text"}))["models"]
    assert [m["id"] for m in text] == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]  # smallest first


# ---------------------------------------------------------------- the loop


class Scripted:
    """An LLM that plays back fixed turns and records what it was told."""

    simulated = False
    model = "gpt-6-luna"

    def __init__(self, turns):
        self.turns = list(turns)
        self.results, self.nudges = [], []

    def start(self, system, user, tools):
        self.system = system

    def step(self):
        return self.turns.pop(0) if self.turns else AgentTurn(calls=[], input_tokens=10, output_tokens=5)

    def add_results(self, results):
        self.results += results

    def nudge(self, text):
        self.nudges.append(text)


def test_tool_errors_go_back_to_the_agent_and_decisions_are_logged(tmp_path):
    d = intents_folder(tmp_path / "raw")
    rationale = {
        "task": "one intent per message",
        "label_source": "labels in data",
        "base_model": "smallest",
        "metric": "balanced classes",
        "split": "given test set",
        "baselines": "default",
    }
    bad_call = ToolCall("1", "sample_rows", {"file": "nope.csv", "n": 2, "where": None})
    submit = ToolCall("2", "submit_task", {"task_yaml": _draft(d), "rationale": rationale})
    llm = Scripted(
        [
            AgentTurn([bad_call], input_tokens=1000, output_tokens=100),
            AgentTurn([submit], input_tokens=2000, output_tokens=300),
        ]
    )
    res = run_plan(d, tmp_path / "runs" / "r1", llm)
    assert res.outcome == "submitted" and res.turns == 2
    assert "error" in llm.results[0][1]
    events = [json.loads(line) for line in (tmp_path / "runs" / "r1" / "events.jsonl").read_text().splitlines()]
    decisions = [e["data"]["rationale"] for e in events if e["type"] == "decision"]
    agent_lines = [x for x in decisions if not x.startswith("rule applied:")]
    assert "metric: balanced classes" in agent_lines and len(agent_lines) == 6
    assert decisions[:6] == agent_lines and any(x.startswith("rule applied: baselines") for x in decisions[6:])
    spend = json.loads((tmp_path / "runs" / "r1" / "plan.json").read_text())["spend"]
    assert len(spend) == 2 and sum(e["usd"] for e in spend) == pytest.approx(
        PRICES.token_price("gpt-6-luna").cost(3000, 0, 400)
    )


def test_loop_limits_turns_nudges_and_spend(tmp_path):
    d = intents_folder(tmp_path / "raw")
    looping = Scripted(
        [AgentTurn([ToolCall(str(i), "get_profile", {})], input_tokens=10, output_tokens=1) for i in range(40)]
    )
    assert run_plan(d, tmp_path / "runs" / "a", looping, max_turns=5).reason == "no decision after 5 turns"
    silent = Scripted([])
    res = run_plan(d, tmp_path / "runs" / "b", silent)
    assert res.outcome == "gave_up" and len(silent.nudges) == 2
    pricey = Scripted(
        [AgentTurn([ToolCall(str(i), "get_profile", {})], input_tokens=5_000_000, output_tokens=0) for i in range(10)]
    )
    res = run_plan(d, tmp_path / "runs" / "c", pricey, max_usd=1.0)
    assert res.outcome == "gave_up" and "spend limit" in res.reason and res.turns == 2


# ---------------------------------------------------------------- the fake agent's decisions


@pytest.mark.parametrize(
    ("fixture", "why"), [("too_small", "only 60 labeled rows"), ("open_ended", "open-ended free text")]
)
def test_fake_agent_declines_the_spec_fixtures(tmp_path, fixture, why):
    res = run_plan(FIXTURES / fixture, tmp_path / "runs" / fixture)
    assert res.outcome == "declined" and why in res.reason and res.suggestion


def test_fake_agent_declines_images_when_no_model_serves_them(tmp_path):
    d = tmp_path / "receipts"
    (d / "images").mkdir(parents=True)
    with (d / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "total"])
        for i in range(600):
            Image.new("RGB", (8, 8)).save(d / "images" / f"r{i}.png") if i < 3 else None
            w.writerow([f"images/r{i}.png", str(i)])
    res = run_plan(d, tmp_path / "runs" / "img")
    assert res.outcome == "declined" and res.fit == "no_available_model"
    assert "serves image inputs" in res.reason and "another fine-tuning provider" in res.suggestion
    assert res.fit_label == "good fit, but no available model on Together can serve it"


def test_spec_fixture_declines_are_not_a_fit(tmp_path):
    res = run_plan(FIXTURES / "too_small", tmp_path / "runs" / "small")
    assert res.fit == "not_a_fit" and res.fit_label == "not a good fit for fine-tuning"
    plan = json.loads((tmp_path / "runs" / "small" / "plan.json").read_text())
    assert plan["fit"] == "not_a_fit"


def test_no_available_model_decline_needs_an_empty_model_list(tmp_path):
    prices = Prices.load(REPO_ROOT / "prices.yaml", dry_run=True)
    tools = AgentTools(Folder(FIXTURES / "tiny_intents", REPO_ROOT), prices, load_models(REPO_ROOT / "models.yaml"))
    out = json.loads(tools.call("decline", {"fit": "no_available_model", "reason": "r", "suggestion": "s"}))
    assert "error" in out and tools.outcome is None
    assert json.loads(tools.call("list_base_models", {"kind": "image"}))["models"] == []
    tools.call("decline", {"fit": "no_available_model", "reason": "r", "suggestion": "Use a frontier model."})
    assert tools.outcome.fit == "no_available_model"
    assert tools.outcome.suggestion == (
        "Use a frontier model. Try another fine-tuning provider whose deployed models serve image inputs."
    )


# ---------------------------------------------------------------- CLI: plan, checkpoint, run


def test_dry_run_from_a_raw_folder_end_to_end(tmp_path, isolated_runs):
    d = intents_folder(tmp_path / "raw")
    r = CliRunner().invoke(app, ["run", str(d), "--dry-run", "--yes", "--run-id", "agentrun"])
    assert r.exit_code == 0, r.output
    assert (d / "task.yaml").exists()
    ctx = RunContext.open("agentrun", yes=True)
    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert any(x.startswith("metric:") for x in report["setup"]["agent_rationale"])
    assert any(e["stage"] == "agent" for e in ctx.state.data["spend"])  # plan spend moved into the run
    assert report["verdict"] is not None


def test_checkpoint_quit_stops_before_paid_stages(tmp_path, isolated_runs):
    d = intents_folder(tmp_path / "raw")
    r = CliRunner().invoke(app, ["run", str(d), "--dry-run", "--run-id", "cp"], input="q\n")
    assert r.exit_code == 0, r.output
    assert "10 random test examples" in r.output and "estimated cost $" in r.output
    ctx = RunContext.open("cp", yes=True)
    assert ctx.state.stage("split").get("status") == "done" and not ctx.state.stage("baseline")
    assert [e["data"]["choice"] for e in ctx.events.of_type("checkpoint")] == ["quit"]
    # resuming shows the checkpoint again instead of going on to the paid stages
    r = CliRunner().invoke(app, ["resume", "cp"], input="q\n")
    assert r.exit_code == 0, r.output
    assert "10 random test examples" in r.output and not RunContext.open("cp", yes=True).state.stage("baseline")
    assert [e["data"]["choice"] for e in RunContext.open("cp", yes=True).events.of_type("checkpoint")] == ["quit"] * 2


def test_checkpoint_edit_replaces_the_runs_task(tmp_path, isolated_runs, monkeypatch):
    d = intents_folder(tmp_path / "raw")

    def edit(filename):  # stands in for $EDITOR
        cfg = yaml.safe_load(Path(filename).read_text())
        cfg["task"]["instruction"] = "Pick the intent."
        Path(filename).write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr("click.edit", edit)
    r = CliRunner().invoke(app, ["run", str(d), "--dry-run", "--run-id", "ed"], input="e\nq\n")
    assert r.exit_code == 0, r.output
    ctx = RunContext.open("ed", yes=True)
    assert ctx.cfg.task.instruction == "Pick the intent."
    assert [e["data"]["choice"] for e in ctx.events.of_type("checkpoint")] == ["edit", "quit"]


def test_plan_refuses_to_overwrite_and_reports_declines(tmp_path, isolated_runs):
    r = CliRunner().invoke(app, ["plan", str(FIXTURES / "too_small"), "--dry-run"])
    assert r.exit_code == 2 and "declined" in r.output and not (FIXTURES / "too_small" / "task.yaml").exists()
    r = CliRunner().invoke(app, ["plan", str(FIXTURES / "tiny_intents"), "--dry-run"])
    assert r.exit_code == 1 and "pass --force" in r.output


def test_plan_spend_counts_until_the_run_takes_it_over(tmp_path):
    runs = tmp_path / "runs"
    (runs / "p").mkdir(parents=True)
    (runs / "p" / "plan.json").write_text(json.dumps({"spend": [{"usd": 0.25, "simulated": False}]}))
    assert total_real_spend(runs) == pytest.approx(0.25)
    (runs / "p" / "state.json").write_text(json.dumps({"spend": [{"usd": 0.25, "simulated": False}, {"usd": 1.0}]}))
    assert total_real_spend(runs) == pytest.approx(1.25)  # not double counted
