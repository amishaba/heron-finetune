"""M4: the retry loop (spec 13). Dry runs with a fake tuned model whose accuracy follows the training
config, so a right fix visibly helps."""

import json

import pytest
import yaml
from rich.console import Console

from ft.config import parse_task
from ft.context import RunContext
from ft.fixer import FixError, build_patch
from ft.pipeline import run_pipeline
from ft.providers.fake import FakeOptions
from tests.conftest import TINY

quiet = Console(quiet=True)


def _task(tmp_path, **train) -> object:
    cfg = yaml.safe_load(TINY.read_text())
    cfg["source"]["path"] = str(TINY.parent.resolve())
    cfg["train"].update(train)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(tmp_path, run_id, fake=None, **train):
    fake = fake or FakeOptions(tuned_q=0.97, tuned_q_from_config=True, full_rows=150, frontier_p=0.85)
    ctx = RunContext.create(_task(tmp_path, **train), dry_run=True, yes=True, run_id=run_id, fake_opts=fake)
    run_pipeline(ctx, quiet)
    return RunContext.open(run_id, yes=True)


WEAK = {"max_examples": 60, "epochs": 1, "learning_rate": 1.0e-6}  # the spec's deliberately weak config, scaled


def test_weak_config_reaches_a_pass_through_the_fixer(tmp_path, isolated_runs):
    ctx = _run(tmp_path, "weak", **WEAK)
    vals = {n: json.loads((ctx.attempt_dir(n) / "metrics.val.json").read_text()) for n in (1, 2)}
    assert not vals[1]["pass"]["passed"] and vals[2]["pass"]["passed"]  # the right fix helped
    assert vals[2]["primary"] > vals[1]["primary"] + 30
    origin = json.loads((ctx.attempt_dir(2) / "origin.json").read_text())
    assert origin["by"] == "fixer" and origin["base_attempt"] == 1
    assert {f["fix"] for f in origin["fixes"]} == {"change_learning_rate", "more_epochs", "use_more_data"}
    patch = yaml.safe_load((ctx.attempt_dir(2) / "task.patch.yaml").read_text())
    assert set(patch) == {"train"} and patch["train"] == {"epochs": 3, "learning_rate": 1e-4, "max_examples": None}
    assert ctx.state.data["final_attempt"] == 2 and not (ctx.attempt_dir(3)).exists()
    assert ctx.events.of_type("human_intervention") == []
    report = json.loads((ctx.run_dir / "report.json").read_text())
    a2 = next(a for a in report["attempts"] if a["n"] == 2)
    assert a2["by_fixer"] and a2["final"] and "epochs 1 -> 3" in a2["changes"]


def test_fixer_sees_val_only(tmp_path, isolated_runs):
    ctx = _run(tmp_path, "valonly", **WEAK)
    diag = json.loads((ctx.attempt_dir(1) / "diagnosis.json").read_text())
    test_ids = {json.loads(line)["id"] for line in (ctx.data_dir / "test.jsonl").read_text().splitlines()}
    text = json.dumps(diag)
    assert '"test"' not in text and not any(i in text for i in test_ids)
    assert diag["loss_curve"] and diag["data"]["train_rows_used"] == 60 and diag["worst_classes"]


def test_loop_stops_when_the_fixer_has_nothing_left(tmp_path, isolated_runs):
    # sane config, no fallback model: the fake fixer tries bigger_model, gets an error, and stops
    fake = FakeOptions(tuned_q=0.2, frontier_p=0.95)
    ctx = _run(tmp_path, "stuck", fake=fake, fallback_models=[])
    assert [int(k) for k in ctx.state.data["attempts"]] == [1]
    stops = ctx.events.of_type("retry_stopped")
    assert stops and "the fixer stopped" in stops[-1]["data"]["reason"]
    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert "fixer stopped" in report["retry_stopped"]


def test_loop_stops_before_an_attempt_the_budget_cant_cover(tmp_path, isolated_runs):
    cfg = yaml.safe_load(_task(tmp_path, **WEAK).read_text())
    cfg["budget"]["max_spend_usd"] = 7.0  # attempt 1 costs ~$5 ($4 training minimum + endpoint)
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(cfg))
    fake = FakeOptions(tuned_q=0.97, tuned_q_from_config=True, full_rows=150, frontier_p=0.85)
    ctx = RunContext.create(path, dry_run=True, yes=True, run_id="broke", fake_opts=fake)
    run_pipeline(ctx, quiet)
    stops = ctx.events.of_type("retry_stopped")
    assert stops and "won't fit" in stops[-1]["data"]["reason"] and not ctx.attempt_dir(2).exists()


def test_patch_limits():
    cfg = parse_task(yaml.safe_load(TINY.read_text()), where="t")
    lim = {"lr_lo": 1e-6, "lr_hi": 1e-3, "max_epochs": 6}
    patch, changes = build_patch(cfg, cfg, {}, [{"fix": "more_epochs", "value": 2, "reason": "r"}], lim, 240)
    assert patch == {"train": {"epochs": 5}} and changes == ["epochs 3 -> 5"]
    bad = [
        [{"fix": "more_epochs", "value": 4, "reason": "r"}],  # 3 + 4 > 6
        [{"fix": "change_learning_rate", "value": 0.5, "reason": "r"}],
        [{"fix": "add_examples", "value": None, "reason": "r"}],
        [{"fix": "use_more_data", "value": None, "reason": "r"}],  # not capped
        [{"fix": "more_epochs", "value": 1, "reason": "r"}, {"fix": "more_epochs", "value": 1, "reason": "r"}],
        [],
    ]
    for fixes in bad:
        with pytest.raises(FixError):
            build_patch(cfg, cfg, {}, fixes, lim, 240)
    # builds on the base attempt's patch; never touches anything outside train
    patch, _ = build_patch(cfg, cfg, {"train": {"max_examples": 50}}, [{"fix": "bigger_model", "value": None,
                                                                        "reason": "r"}], lim, 240)  # fmt: skip
    assert patch == {"train": {"max_examples": 50, "base_model": "Qwen/Qwen3.5-9B"}}


def _timed_task(tmp_path, **train):
    """The tiny fixture with a time column, so splits are by time (retrain on by default)."""
    import csv

    src = TINY.parent
    out = tmp_path / "timed"
    out.mkdir()
    rows = list(csv.DictReader((src / "train.csv").open())) + list(csv.DictReader((src / "test.csv").open()))
    with (out / "all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*rows[0].keys(), "ts"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}"})
    cfg = yaml.safe_load(TINY.read_text())
    cfg["source"] = {"type": "csv", "path": str(out / "all.csv")}
    cfg["splits"] = {"by_time": "ts", "seed": 42, "test_fraction": 0.2, "val_fraction": 0.15}
    cfg["train"].update(train)
    path = tmp_path / "timed.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_time_split_retrains_the_chosen_config_on_train_and_val(tmp_path, isolated_runs):
    fake = FakeOptions(tuned_q=0.97, tuned_q_by_attempt={1: 1.0, 2: 0.6}, frontier_p=0.85)
    ctx = RunContext.create(_timed_task(tmp_path), dry_run=True, yes=True, run_id="timed", fake_opts=fake)
    run_pipeline(ctx, quiet)
    ctx = RunContext.open("timed", yes=True)
    assert ctx.state.data["final_attempt"] == 1 and ctx.state.data["retrained_attempt"] == 2
    assert ctx.is_retrain(2) and not (ctx.attempt_dir(2) / "metrics.val.json").exists()
    sizes = json.loads((ctx.data_dir / "splits.json").read_text())["sizes"]
    rows = json.loads((ctx.attempt_dir(2) / "prepare.json").read_text())["rows"]
    assert rows == sizes["train"] + sizes["val"]
    report = json.loads((ctx.run_dir / "report.json").read_text())
    tuned = next(r for r in report["comparison"] if r["kind"] == "tuned")
    assert "retrained on train + val as attempt 2" in tuned["model"]
    assert report["retrain"]["attempt"] == 2 and report["verdict"]["passed_val"]  # decided by attempt 1's val
    test2 = json.loads((ctx.attempt_dir(2) / "metrics.test.json").read_text())["primary"]
    assert report["verdict"]["tuned_test"] == test2


def test_retrain_is_off_for_random_splits_and_can_be_forced(tmp_path, isolated_runs):
    fake = FakeOptions(tuned_q_by_attempt={1: 1.0})
    ctx = _run(tmp_path, "plain", fake=fake)
    assert "retrained_attempt" not in ctx.state.data
    ctx = _run(tmp_path, "forced", fake=FakeOptions(tuned_q_by_attempt={1: 1.0}), retrain_with_val=True)
    assert ctx.state.data["retrained_attempt"] == 2


def test_no_retrain_when_nothing_passed(tmp_path, isolated_runs):
    fake = FakeOptions(tuned_q=0.1, frontier_p=0.95)
    ctx = _run(tmp_path, "nopass", fake=fake, retrain_with_val=True, fallback_models=[])
    assert "retrained_attempt" not in ctx.state.data
    assert ctx.events.of_type("retrain_skipped")


def test_empty_answers_stop_the_loop_as_a_serving_problem(tmp_path, isolated_runs):
    fake = FakeOptions(tuned_constant="", frontier_p=0.9)  # every tuned answer comes back empty
    ctx = _run(tmp_path, "empty", fake=fake, **WEAK)
    diag = json.loads((ctx.attempt_dir(1) / "diagnosis.json").read_text())
    assert diag["serving_problem"] and diag["empty_share"] == 1.0
    stops = ctx.events.of_type("retry_stopped")
    assert "serving problem" in stops[-1]["data"]["reason"] and not ctx.attempt_dir(2).exists()
