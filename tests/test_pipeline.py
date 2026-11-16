"""Integration: full dry runs on the tiny fixture, crash/resume, teardown guarantees, spend cap."""

import json

import pytest
from markupsafe import escape as html_escape
from rich.console import Console

from ft.context import RunContext
from ft.pipeline import run_pipeline
from ft.providers.fake import FakeFailure, FakeOptions
from ft.spend import SpendError
from tests.conftest import TINY

quiet = Console(quiet=True)


def PASSING(**kw) -> FakeOptions:  # noqa: N802
    """Attempt 1 passes, so tests of resume and teardown see one attempt; the retry loop has its own tests."""
    return FakeOptions(tuned_q_by_attempt={1: 1.0}, **kw)  # a perfect tuned model clears any bar


def _run(task_path=TINY, run_id="t", fake=None, **kw):
    fake = fake or PASSING()
    ctx = RunContext.create(task_path, dry_run=True, yes=True, run_id=run_id, fake_opts=fake, **kw)
    run_pipeline(ctx, quiet)
    return ctx


def test_full_dry_run_produces_report():
    ctx = _run()
    report = json.loads((ctx.run_dir / "report.json").read_text())
    html = (ctx.run_dir / "report.html").read_text()
    assert html_escape(report["verdict"]["sentence"]) in html
    assert report["interventions"]["count"] == 0
    assert report["dry_run"] is True and "simulated" in html
    assert {r["kind"] for r in report["comparison"]} == {"baseline", "tuned"}
    assert ctx.state.live_endpoints() == {}
    assert ctx.providers.server.deployed == ctx.providers.server.torn_down
    for f in ("records.jsonl", "profile.json", "train.jsonl", "val.jsonl", "test.jsonl", "labels.json"):
        assert (ctx.data_dir / f).exists()
    a1 = ctx.attempt_dir(1)
    for f in (
        "train.together.jsonl",
        "training.json",
        "endpoint.json",
        "val.predictions.jsonl",
        "test.predictions.jsonl",
        "metrics.val.json",
        "metrics.test.json",
    ):
        assert (a1 / f).exists(), f
    # tuned training rows carry only the instruction as system prompt
    first = json.loads((a1 / "train.together.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["content"] == ctx.cfg.task.instruction
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    # every external id recorded
    assert {"file_id", "job_id", "model_object_id"} <= set(ctx.state.external(1))
    assert ctx.state.spent(include_simulated=False) == 0


def test_rerun_skips_unchanged_stages():
    ctx = _run()
    ctx2 = RunContext.open(ctx.run_id, yes=True)
    run_pipeline(ctx2, quiet)
    skipped = {e["stage"] for e in ctx2.events.of_type("stage_skip")}
    assert {"load", "split", "baseline", "train", "predict"} <= skipped
    assert ctx2.providers.trainer.started == [] and ctx2.providers.frontier.calls == 0


def test_from_stage_reruns_later_stages_only():
    ctx = _run()
    ctx2 = RunContext.open(ctx.run_id, yes=True)
    run_pipeline(ctx2, quiet, from_stage="predict")
    assert ctx2.providers.frontier.calls == 0  # baseline untouched
    assert ctx2.providers.trainer.started == []  # job kept; no second paid job
    assert len(ctx2.providers.server.deployed) == 1


def test_resume_after_crash_mid_train_polls_existing_job():
    with pytest.raises(FakeFailure):
        _run(fake=PASSING(seed=42, fail_on={"train.poll"}))
    first = RunContext.open("t", yes=True)
    job_id = first.state.external(1)["job_id"]
    assert first.state.stage("train", 1)["status"] == "failed"

    ctx = RunContext.open("t", yes=True, fake_opts=PASSING(seed=42))
    run_pipeline(ctx, quiet)
    assert ctx.providers.trainer.started == []  # never launched a second job
    assert ctx.state.external(1)["job_id"] == job_id
    assert ctx.events.of_type("resume_poll")
    assert (ctx.run_dir / "report.html").exists()


def test_teardown_runs_when_predict_raises():
    with pytest.raises(FakeFailure):
        _run(fake=FakeOptions(seed=42, fail_on={"tuned.predict"}))
    ctx = RunContext.open("t", yes=True)
    eps = ctx.state.data["endpoints"]
    assert len(eps) == 1
    (rec,) = eps.values()
    assert rec["torn_down_at"] and rec["minutes_billed"] > 0
    assert ctx.state.live_endpoints() == {}
    assert any(e["stage"] == "serve" for e in ctx.state.data["spend"])  # endpoint time still billed


def test_teardown_is_idempotent():
    from ft.providers.base import Endpoint
    from ft.stages import serve

    ctx = _run()
    ((name, rec),) = ctx.state.data["endpoints"].items()
    spend_before = len(ctx.state.data["spend"])
    serve.teardown(ctx, Endpoint(id=rec["id"], name=name, model=rec["model"]), 1)
    assert len(ctx.state.data["spend"]) == spend_before


def test_spend_guard_stops_over_budget_run():
    with pytest.raises(SpendError, match="cap"):
        _run(max_spend=0.000001)
    ctx = RunContext.open("t", yes=True)
    assert ctx.state.stage("baseline")["status"] == "failed"
    assert not ctx.baseline_dir("gpt-6-luna").exists()
    assert ctx.state.data["spend"] == []


def test_endpoint_names_are_valid_and_unique():
    from ft.stages.serve import endpoint_name

    a = endpoint_name("20261017-1430-Banking77_Extra.Long-Name-" + "x" * 80, 1)
    b = endpoint_name("20261017-1430-Banking77_Extra.Long-Name-" + "x" * 80, 2)
    assert a != b and a.endswith("-a1") and len(a) <= 60
    assert all(c.islower() or c.isdigit() or c == "-" for c in a)


def test_attempt_accuracy_tracks_fake_quality():
    hi = _run(run_id="hi", fake=FakeOptions(seed=1, tuned_q=0.99))
    lo = _run(run_id="lo", fake=FakeOptions(seed=1, tuned_q=0.50))
    v_hi = json.loads((hi.attempt_dir(1) / "metrics.val.json").read_text())
    v_lo = json.loads((lo.attempt_dir(1) / "metrics.val.json").read_text())
    assert v_hi["pass"]["passed"] and not v_lo["pass"]["passed"]


def test_collapsed_fine_tune_is_flagged_in_report_and_diagnosis():
    ctx = _run(run_id="collapsed", fake=FakeOptions(seed=1, tuned_constant="lost_card"))
    diag = json.loads((ctx.attempt_dir(1) / "diagnosis.json").read_text())
    assert diag["flags"] and diag["flags"][0].startswith("collapsed: the model is ignoring its input")
    assert set(diag["val"]) >= {"primary", "collapse", "pass"} and "test" not in json.dumps(diag)
    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert [c["model"].startswith("tuned") for c in report["collapsed"]] == [True]
    assert "ignoring its input" in (ctx.run_dir / "report.html").read_text()
    healthy = json.loads((_run(run_id="healthy").run_dir / "report.json").read_text())
    assert healthy["collapsed"] == []


@pytest.mark.parametrize(("cache", "stops"), [(True, False), (False, True)])
def test_cached_prompt_baseline_is_checked_against_billed_cost(tmp_path, monkeypatch, cache, stops):
    """A few-shot baseline is let through on its cached estimate, then stopped partway if the billed cost
    shows caching isn't working and the stage would pass the run cap."""
    import yaml

    import ft.stages
    from ft.stages import baseline

    monkeypatch.setattr(ft.stages, "CHUNK", 5)
    monkeypatch.setattr(baseline, "CHUNK", 5)
    cfg = yaml.safe_load(TINY.read_text())
    cfg["source"]["path"] = str(TINY.parent.resolve())
    cfg["baselines"] = [{"model": "gpt-6-luna", "reasoning_effort": "low", "few_shot": 20}]  # ~1,500-token prompt
    cfg["budget"]["max_spend_usd"] = 0.017  # cached estimate passes, the uncached projection (~$0.02) does not
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump(cfg))
    ctx = RunContext.create(
        task, dry_run=True, yes=True, run_id=f"cache-{cache}", fake_opts=FakeOptions(frontier_cache=cache)
    )
    from ft.pipeline import RUN_STAGES, run_stage

    for mod in RUN_STAGES[: [m.NAME for m in RUN_STAGES].index("baseline")]:
        run_stage(ctx, mod, None, quiet)
    if stops:
        with pytest.raises(SpendError, match="on course for .* Prompt caching may not be working"):
            baseline.run(ctx)
        partial = [e for e in ctx.state.data["spend"] if "partial, stopped" in e["detail"]]
        assert partial and ctx.state.spent() <= 0.017  # stopped before the cap, and what was billed is recorded
    else:
        baseline.run(ctx)
        assert ctx.state.spent() < 0.017


def test_hybrid_routing_simulation_picks_the_threshold_on_val(isolated_runs):
    from ft.hybrid import simulate

    ctx = _run(run_id="hy")
    out = simulate(ctx.run_dir)
    assert out["fallback"] == "gpt-6-luna" and out["primary_metric"] == "accuracy"  # the only baseline
    best = max(c["primary"] for c in out["val_curve"])
    assert out["val"]["hybrid_primary"] == best  # chosen on val
    assert out["val_curve"][0]["routed_share"] == 0.0  # threshold 0 routes nothing
    t = out["test"]
    if t["routed_share"] == 0:
        assert t["hybrid"]["accuracy"] == t["tuned"]["accuracy"]
    assert t["hybrid"]["cost_per_1k"] >= t["tuned"]["cost_per_1k"]


def test_cli_reports_a_failed_training_job_without_a_traceback(monkeypatch):
    """A stage failure with a known cause is one `error:` line plus how to resume, not an uncaught exception."""
    from typer.testing import CliRunner

    from ft.cli import app
    from ft.providers.base import JobStatus
    from ft.providers.fake import FakeTrainer

    failed = JobStatus(status="error", terminal=True, error="out of memory")
    monkeypatch.setattr(FakeTrainer, "poll", lambda self, job_id: failed)
    r = CliRunner().invoke(app, ["run", str(TINY), "--dry-run", "--yes", "--run-id", "train-fails"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)
    assert "out of memory" in r.output and "ft resume train-fails" in r.output
