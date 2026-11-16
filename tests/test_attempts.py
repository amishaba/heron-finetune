"""Hand-designed attempts (`ft attempt`), labels_in_prompt and class weighting."""

import json
import random

import pytest
from rich.console import Console

from ft.config import ConfigError
from ft.context import RunContext
from ft.pipeline import ATTEMPT_STAGES, RUN_STAGES, add_attempt, run_pipeline
from ft.providers.fake import FakeOptions
from ft.stages.prepare import class_weights, oversample
from tests.conftest import TINY

quiet = Console(quiet=True)


def test_class_weights_boost_only_rare_classes_and_respect_the_cap():
    rows = [{"target": "big"}] * 100 + [{"target": "mid"}] * 25 + [{"target": "rare"}] * 4
    w = class_weights(rows, "sqrt", 5.0)
    assert w["big"] == 1.0 and w["mid"] == 1.0  # median is 25: never down-weight
    assert w["rare"] == pytest.approx(2.5)  # sqrt(25 / 4)
    assert class_weights(rows, "balanced", 5.0)["rare"] == 5.0  # 6.25, capped
    out = oversample(rows, w, random.Random(0))
    n_rare = sum(r["target"] == "rare" for r in out)
    assert 8 <= n_rare <= 12 and sum(r["target"] == "big" for r in out) == 100


def test_labels_in_prompt_reaches_training_file_and_inference(tiny_task_path):
    ctx = RunContext.create(TINY, dry_run=True, yes=True, run_id="a", fake_opts=FakeOptions(seed=3))
    run_pipeline(ctx, quiet)
    plain = ctx.task_type(1).tuned_prompt().system
    patch = {"train": {"labels_in_prompt": True, "class_weighting": "balanced", "class_weight_cap": 3}}
    n = add_attempt(ctx, quiet, patch, "names help rare labels", after_test=True)
    assert n == 2
    system = ctx.task_type(2).tuned_prompt().system
    assert system.startswith(plain) and all(lbl in system for lbl in ctx.labels())
    first = json.loads((ctx.attempt_dir(2) / "train.together.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["content"] == system
    prep = json.loads((ctx.attempt_dir(2) / "prepare.json").read_text())
    assert prep["class_weighting"]["rows_after"] >= prep["class_weighting"]["rows_before"]

    report = json.loads((ctx.run_dir / "report.json").read_text())
    a2 = next(a for a in report["attempts"] if a["n"] == 2)
    assert a2["by_hand"] and a2["designed_after_test"] and a2["test"] is not None
    assert report["interventions"]["count"] == 1
    assert "after seeing test" in (ctx.run_dir / "report.html").read_text()
    assert ("caveat" in report["verdict"]) == a2["final"]  # the caveat follows the final attempt

    # Resuming changes nothing: attempt 1 isn't re-run and attempt 2 stays done.
    ctx = RunContext.open("a", yes=True)
    for mod in RUN_STAGES:
        ctx.attempt = None
        assert ctx.state.stage_done(mod.NAME, None, mod.input_hash(ctx))
    for m in (1, 2):
        ctx.attempt = m
        for mod in ATTEMPT_STAGES:
            assert ctx.state.stage_done(mod.NAME, m, mod.input_hash(ctx)), (m, mod.NAME)


def test_attempt_needs_a_valid_patch_within_max_attempts():
    ctx = RunContext.create(TINY, dry_run=True, yes=True, run_id="b", fake_opts=FakeOptions(seed=3))
    run_pipeline(ctx, quiet)
    with pytest.raises(ConfigError):
        add_attempt(ctx, quiet, {"train": {"class_weighting": "loud"}}, "bad", after_test=False)
    with pytest.raises(ConfigError, match="empty"):
        add_attempt(ctx, quiet, {}, "nothing", after_test=False)
    assert not ctx.attempt_dir(2).exists()
