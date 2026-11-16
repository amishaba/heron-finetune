"""M6: logs mode (spec 15). Offline: simulated logs from the tiny fixture, then a logs-mode dry run."""

import json
from pathlib import Path

import yaml
from rich.console import Console
from typer.testing import CliRunner

from ft.cli import app
from ft.connectors.logs import normalize, read_logs
from ft.context import REPO_ROOT, RunContext
from ft.pipeline import run_pipeline
from ft.providers.fake import FakeOptions
from ft.simulate import simulate_logs
from tests.conftest import TINY

quiet = Console(quiet=True)


def test_normalizes_openai_anthropic_and_chat_completion_shapes():
    native = {"id": "a", "request": {"model": "m", "messages": [{"role": "system", "content": "S"},
              {"role": "user", "content": "hi"}]}, "response": {"content": " refund ",
              "usage": {"input_tokens": 9, "output_tokens": 2}}, "latency_ms": 100}  # fmt: skip
    anthropic = {"request": {"model": "claude-x", "system": [{"type": "text", "text": "S"}], "messages": [
        {"role": "user", "content": [{"type": "text", "text": "first"}]}, {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]}]},
        "response": {"content": [{"type": "text", "text": "lost_card"}], "usage": {"input_tokens": 5,
                     "output_tokens": 1, "cache_read_input_tokens": 3}}}  # fmt: skip
    chat = {"id": "c", "request": {"model": "gpt", "messages": [{"role": "user", "content": "x"}]},
            "response": {"choices": [{"message": {"content": "refund"}}],
                         "usage": {"prompt_tokens": 7, "completion_tokens": 1}}}  # fmt: skip
    a, b, c = normalize(native), normalize(anthropic, 2), normalize(chat)
    assert (a["system_prompt"], a["last_user_message"], a["response_content"]) == ("S", "hi", "refund")
    assert a["usage"]["input_tokens"] == 9 and a["logged_model"] == "m"
    assert b["last_user_message"] == "second" and b["system_prompt"] == "S" and b["response_content"] == "lost_card"
    assert b["usage"]["cache_read_tokens"] == 3 and "assistant: ok" in b["conversation"] and b["id"]
    assert c["response_content"] == "refund" and c["usage"] == {"input_tokens": 7, "output_tokens": 1,
                                                                 "cache_read_tokens": 0}  # fmt: skip


def test_bad_lines_are_skipped_and_truth_is_joined(tmp_path):
    logs = tmp_path / "logs.jsonl"
    good = {"id": "1", "request": {"messages": [{"role": "user", "content": "q"}]}, "response": {"content": "y"}}
    logs.write_text(json.dumps(good) + "\nnot json\n" + json.dumps({"id": "2", "request": {"messages": []}}) + "\n")
    (tmp_path / "truth.jsonl").write_text(json.dumps({"id": "1", "label": "z"}) + "\n")
    rows, stats = read_logs(logs, tmp_path / "truth.jsonl")
    assert stats["rows"] == 1 and stats["skipped"] == 2 and rows[0]["truth"] == "z"


def _logs_task(tmp_path):
    out = tmp_path / "sim"
    res = simulate_logs(TINY, "gpt-6-luna", 400, out, dry_run=True, log=lambda m: None)
    cfg = yaml.safe_load((REPO_ROOT / "examples" / "banking77-logs" / "task.yaml").read_text())
    cfg["source"] = {"type": "logs", "path": str(out / "logs.jsonl"), "truth_sidecar": str(out / "truth.jsonl")}
    cfg["splits"]["test_max"] = 60
    path = tmp_path / "logs.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, res


def test_simulate_logs_writes_logs_truth_and_a_spend_record(tmp_path, isolated_runs):
    _path, res = _logs_task(tmp_path)
    assert res["written"] == 300 and res["failed"] == 0  # the fixture has 300 distinct messages
    log = json.loads((tmp_path / "sim" / "logs.jsonl").read_text().splitlines()[0])
    assert log["request"]["messages"][0]["role"] == "system" and "routing" in log["request"]["messages"][0]["content"]
    truth = (tmp_path / "sim" / "truth.jsonl").read_text()
    assert truth.count("\n") == 300 and 70 < res["logged_vs_truth"] < 100  # the fake frontier is ~90% right
    plan = json.loads((Path(res["record"]) / "plan.json").read_text())
    assert plan["kind"] == "simulate-logs" and plan["spend"][0]["simulated"]


def test_logs_run_reports_agreement_and_truth_accuracy(tmp_path, isolated_runs):
    path, _ = _logs_task(tmp_path)
    ctx = RunContext.create(path, dry_run=True, yes=True, run_id="logs", fake_opts=FakeOptions(tuned_q=0.99))
    run_pipeline(ctx, quiet)
    report = json.loads((ctx.run_dir / "report.json").read_text())
    kinds = {r["kind"]: r for r in report["comparison"]}
    assert set(kinds) == {"logged", "tuned"} and report["truth_mode"]
    assert kinds["logged"]["primary"] == 100.0 and kinds["logged"]["truth_accuracy"] is not None
    assert kinds["tuned"]["truth_accuracy"] is not None and report["verdict"]["passed_val"]
    assert "Against the truth labels" in report["verdict"]["sentence"]
    records = [json.loads(x) for x in (ctx.data_dir / "train.jsonl").read_text().splitlines()]
    assert all(r["target"] and "truth" in r["meta"] for r in records)  # truth rides along, never the target
    html = (ctx.run_dir / "report.html").read_text()
    assert "truth accuracy" in html and "pseudo-baseline" in html


def test_simulate_logs_cli_dry_run(tmp_path, isolated_runs):
    r = CliRunner().invoke(app, ["simulate-logs", str(TINY), "--out", str(tmp_path / "o"), "--n", "50", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "wrote 50 logs" in r.output


def test_self_agreement_bar_retests_the_logged_model(tmp_path, isolated_runs):
    path, _ = _logs_task(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["eval"]["pass_bar"] = {"compare_to": "logged_self_agreement", "max_gap_points": 2.0}
    path.write_text(yaml.safe_dump(cfg))
    ctx = RunContext.create(path, dry_run=True, yes=True, run_id="selfbar", fake_opts=FakeOptions(tuned_q=0.99))
    run_pipeline(ctx, quiet)
    retest = json.loads((ctx.baseline_dir("logged-retest") / "metrics.val.json").read_text())
    assert 70 < retest["primary"] < 100 and retest["n"] == len((ctx.data_dir / "val.jsonl").read_text().splitlines())
    p = json.loads((ctx.attempt_dir(1) / "metrics.val.json").read_text())["pass"]
    assert p["threshold"] == round(retest["primary"] - 2.0, 2) and "self-agreement" in p["rule"]
    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert "self-agreement" in report["verdict"]["headline"]


def test_self_agreement_bar_is_for_logs_only():
    import pytest

    from ft.config import parse_task

    cfg = yaml.safe_load(TINY.read_text())
    cfg["eval"]["pass_bar"] = {"compare_to": "logged_self_agreement", "max_gap_points": 2.0}
    with pytest.raises(Exception, match="only for targets.from: logs"):
        parse_task(cfg, where="t")
