"""Several baselines per model, few-shot prompts, per-baseline caching, and updating a run's task."""

import copy
import json

import pytest
import yaml
from rich.console import Console

from ft.config import ConfigError, parse_task
from ft.context import RunContext
from ft.pipeline import run_pipeline
from tests.conftest import TINY

quiet = Console(quiet=True)


def _cfg(tiny_raw, baselines, **bar):
    raw = copy.deepcopy(tiny_raw)
    raw["baselines"] = baselines
    if bar:
        raw["eval"]["pass_bar"] = bar
    return parse_task(raw)


def test_names_default_to_model_or_settings(tiny_raw):
    one = _cfg(tiny_raw, [{"model": "gpt-6-luna", "reasoning_effort": "low"}])
    assert one.baselines[0].name == "gpt-6-luna"
    three = _cfg(
        tiny_raw,
        [
            {"model": "gpt-6-luna", "reasoning_effort": "low"},
            {"model": "gpt-6-luna", "reasoning_effort": "medium"},
            {"model": "gpt-6-luna", "reasoning_effort": "low", "few_shot": 3},
        ],
    )
    assert [b.name for b in three.baselines] == ["gpt-6-luna-low", "gpt-6-luna-medium", "gpt-6-luna-low-3shot"]


def test_duplicate_names_and_unknown_compare_to_are_errors(tiny_raw):
    with pytest.raises(ConfigError, match="unique"):
        _cfg(tiny_raw, [{"model": "m", "reasoning_effort": "low"}, {"model": "m", "reasoning_effort": "low"}])
    with pytest.raises(ConfigError, match="compare_to"):
        _cfg(tiny_raw, [{"model": "m"}], compare_to="nope", max_gap_points=1)
    ok = _cfg(tiny_raw, [{"model": "m", "name": "luna-a"}], compare_to="luna-a", max_gap_points=1)
    assert ok.eval.pass_bar.compare_to == "luna-a"


def _write(tmp_path, tiny_raw, baselines, name="task.yaml"):
    raw = copy.deepcopy(tiny_raw)
    raw["baselines"] = baselines
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw))
    return p


def test_few_shot_examples_come_from_train_and_sit_in_the_system_prompt(tmp_path, tiny_raw):
    p = _write(tmp_path, tiny_raw, [{"model": "gpt-6-luna", "reasoning_effort": "low", "few_shot": 2}])
    ctx = RunContext.create(p, dry_run=True, yes=True, run_id="fs")
    run_pipeline(ctx, quiet)
    d = ctx.baseline_dir("gpt-6-luna")
    shots = json.loads((d / "few_shot.json").read_text())["examples"]
    train_ids = {json.loads(line)["id"] for line in (ctx.data_dir / "train.jsonl").read_text().splitlines()}
    assert {e["id"] for e in shots} <= train_ids  # never val/test
    per_label = {}
    for e in shots:
        per_label[e["target"]] = per_label.get(e["target"], 0) + 1
    assert set(per_label) == set(ctx.labels()) and set(per_label.values()) == {2}
    prompt = ctx.task_type().baseline_prompt(None, shots)
    assert "Labeled examples:" in prompt.system and shots[0]["input"].split()[0] in prompt.system


def test_adding_baselines_to_a_finished_run_only_runs_the_new_ones(tmp_path, tiny_raw):
    first = _write(tmp_path, tiny_raw, [{"model": "gpt-6-luna", "reasoning_effort": "low", "name": "luna-low"}])
    ctx = RunContext.create(first, dry_run=True, yes=True, run_id="add")
    run_pipeline(ctx, quiet)
    calls_before = ctx.providers.frontier.calls
    trainer_jobs = list(ctx.providers.trainer.started)

    second = _write(
        tmp_path,
        tiny_raw,
        [
            {"model": "gpt-6-luna", "reasoning_effort": "low", "name": "luna-low"},
            {"model": "gpt-6-luna", "reasoning_effort": "medium", "name": "luna-medium"},
            {"model": "gpt-6-luna", "reasoning_effort": "low", "few_shot": 3, "name": "luna-low-3shot"},
        ],
        name="task2.yaml",
    )
    changed = RunContext.replace_task("add", second)
    assert changed == ["baselines"]
    ctx2 = RunContext.open("add", yes=True)
    run_pipeline(ctx2, quiet)

    n_eval = sum(1 for s in ("val", "test") for _ in (ctx2.data_dir / f"{s}.jsonl").read_text().splitlines())
    assert ctx2.providers.frontier.calls == 2 * n_eval  # the two new baselines only
    assert calls_before == n_eval
    assert ctx2.providers.trainer.started == [] and trainer_jobs  # no retraining
    report = json.loads((ctx2.run_dir / "report.json").read_text())
    assert len(report["comparison"]) == 4  # 3 baselines + tuned
    assert {c["baseline"] for c in report["verdict"]["comparisons"]} == {"luna-low", "luna-medium", "luna-low-3shot"}
    assert report["metadata"]["task_updates"][0]["changed"] == ["baselines"]
    skipped = {e["stage"] for e in ctx2.events.of_type("stage_skip")}
    assert {"load", "split", "prepare", "train", "predict"} <= skipped


def test_replace_task_rejects_a_different_task(tmp_path, tiny_raw):
    RunContext.create(TINY, dry_run=True, yes=True, run_id="r")
    raw = copy.deepcopy(tiny_raw)
    raw["name"] = "something-else"
    p = tmp_path / "other.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="new run"):
        RunContext.replace_task("r", p)


def test_spend_is_recorded_when_a_baseline_stops_on_failures(tmp_path, tiny_raw, monkeypatch):
    from ft.providers.base import Prediction, Usage
    from ft.stages import TooManyFailures

    p = _write(tmp_path, tiny_raw, [{"model": "gpt-6-luna", "reasoning_effort": "low"}])
    ctx = RunContext.create(p, dry_run=True, yes=True, run_id="fail")

    async def flaky(items, prompt, settings):
        return [
            Prediction(
                id=it.id,
                prediction_raw=None if i % 4 == 0 else '{"label": "lost_card"}',
                latency_ms=1.0,
                usage=Usage(input_tokens=1000, output_tokens=10),
                error="RateLimitError: slow down" if i % 4 == 0 else None,
            )
            for i, it in enumerate(items)
        ]

    monkeypatch.setattr(ctx.providers.frontier, "predict", flaky)
    with pytest.raises(TooManyFailures):
        run_pipeline(ctx, quiet)
    partial = [e for e in ctx.state.data["spend"] if "partial" in e["detail"]]
    assert len(partial) == 1 and partial[0]["usd"] > 0
