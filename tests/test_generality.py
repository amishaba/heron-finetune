"""The pipeline must not be tailored to BANKING77. These run datasets that break its assumptions,
changing only task.yaml (and data), never code."""

import csv
import json
from collections import Counter

import pytest
import yaml
from rich.console import Console

from ft.config import ConfigError
from ft.context import RunContext
from ft.pipeline import run_pipeline
from ft.stages.split import stratified_take
from tests.conftest import REPO

quiet = Console(quiet=True)
ROUTING = REPO / "examples" / "fixtures" / "support_routing" / "task.yaml"


def _run(task_path, run_id="g"):
    ctx = RunContext.create(task_path, dry_run=True, yes=True, run_id=run_id)
    run_pipeline(ctx, quiet)
    return ctx


def _split(ctx, name):
    return [json.loads(line) for line in (ctx.data_dir / f"{name}.jsonl").read_text().splitlines()]


def test_lopsided_jsonl_without_predefined_splits_runs_end_to_end():
    ctx = _run(ROUTING)
    meta = json.loads((ctx.data_dir / "splits.json").read_text())
    report = json.loads((ctx.run_dir / "report.json").read_text())

    # JSONL, columns `body`/`route`, one file with no split names → test and val were carved
    assert meta["method"] == "carved"
    assert all("split_hint" not in r["meta"] for r in _split(ctx, "test"))
    # spelling variants are one class, not three
    assert ctx.labels() == [
        "account_access",
        "billing",
        "legal_and_compliance_escalation",
        "shipping_and_delivery_delays",
        "technical_support",
    ]
    assert meta["label_variants_merged"] == {
        "billing": [" billing ", "Billing"],
        "technical_support": ["Technical_Support"],
    }
    # blank inputs never reach training
    assert json.loads((ctx.data_dir / "load_stats.json").read_text())["dropped"]["blank_input"] == 2
    assert all(r["input"].strip() for name in ("train", "val", "test") for r in _split(ctx, name))
    # lopsided classes: every class keeps training examples; the metric is macro-F1
    train = Counter(r["target"] for r in _split(ctx, "train"))
    assert set(train) == set(ctx.labels()) and train.most_common(1)[0][1] / sum(train.values()) > 0.5
    assert report["primary_metric"] == "macro_f1"
    assert "merged" in (ctx.run_dir / "report.html").read_text()


def test_accuracy_on_lopsided_classes_is_flagged(tmp_path):
    raw = yaml.safe_load(ROUTING.read_text())
    raw["eval"]["primary_metric"] = "accuracy"
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(raw))
    ctx = _run(p)
    warnings = json.loads((ctx.data_dir / "splits.json").read_text())["warnings"]
    assert any("billing" in w and "macro_f1" in w for w in warnings)


def _write_task(tmp_path, data_file, **task_overrides):
    raw = yaml.safe_load(ROUTING.read_text())
    raw["source"] = {"type": data_file.suffix.lstrip("."), "path": str(data_file)}
    raw["task"]["input"]["field"] = "msg"
    raw["task"]["output"]["field"] = "y"
    raw["splits"] = {"val_fraction": 0.2, "test_fraction": 0.2, "seed": 1}
    for k, v in task_overrides.items():
        raw[k] = v
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_integer_labels_from_jsonl(tmp_path):
    data = tmp_path / "ints.jsonl"
    data.write_text("".join(json.dumps({"msg": f"message number {i}", "y": i % 2}) + "\n" for i in range(120)))
    ctx = _run(_write_task(tmp_path, data))
    assert ctx.labels() == ["0", "1"]
    m = json.loads((ctx.attempt_dir(1) / "metrics.test.json").read_text())
    assert m["invalid_rate"] == 0.0 and m["primary"] > 50  # labels parse and score as strings


def test_multi_label_targets_are_rejected_clearly(tmp_path):
    data = tmp_path / "multi.jsonl"
    data.write_text("".join(json.dumps({"msg": f"m {i}", "y": ["a", "b"]}) + "\n" for i in range(20)))
    with pytest.raises(ConfigError, match="exactly one label per row"):
        _run(_write_task(tmp_path, data))


def test_csv_with_bom_and_unlabeled_rows(tmp_path):
    data = tmp_path / "bom.csv"
    with data.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["msg", "y"])
        for i in range(120):
            w.writerow([f"message {i}", "" if i % 40 == 0 else ("x" if i % 3 else "y")])
    ctx = _run(_write_task(tmp_path, data))
    assert json.loads((ctx.data_dir / "load_stats.json").read_text())["dropped"]["missing_target"] == 3
    assert ctx.labels() == ["x", "y"]


def test_carving_keeps_one_example_of_every_class_in_train():
    import random

    rows = [{"id": str(i), "target": "big"} for i in range(50)] + [{"id": "r1", "target": "rare"}]
    taken, rest = stratified_take(rows, 25, lambda r: r["target"], random.Random(0), keep_one=True)
    assert any(r["target"] == "rare" for r in rest)


def test_output_budget_scales_with_label_length(make_cfg):
    from ft.tasks.classify import ClassifyTask

    short = ClassifyTask(make_cfg(), ["a", "b"])
    long = ClassifyTask(
        make_cfg(),
        ["escalate_to_the_legal_and_compliance_team_within_one_business_day_and_copy_the_account_owner", "b"],
    )
    assert short.tuned_prompt().max_output_tokens == 16
    assert long.tuned_prompt().max_output_tokens > 16
    assert long.baseline_prompt().max_output_tokens > 64
