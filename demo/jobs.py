"""Live demo runs: upload → the agent plans → the visitor approves → baselines, fine-tune, benchmark.

A thin layer over the real pipeline (nothing here is demo-only logic about models): `plan_folder` is the
M3 agent, `checkpoint_summary` is the human checkpoint, `run_pipeline` is the whole run with its spend
guard, retry loop and guaranteed endpoint teardown. The demo only adds upload checks, smaller val/test sets
so a run finishes in ~25 minutes, one paid run at a time, and a per-visitor daily limit.

Each run's demo state is `runs/<id>/demo.json`; its console output goes to `runs/<id>/demo.log`.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import re
import secrets
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from ft import pipeline
from ft.agent.session import checkpoint_summary, plan_folder, task_path_for
from ft.context import REPO_ROOT, RunContext, runs_dir
from ft.events import EventLog, now_iso
from ft.io import read_json, write_json
from ft.spend import budget_left_usd

DRY_RUN = os.environ.get("FT_DEMO_DRY_RUN") == "1"  # fakes, no spend: for local testing
DEMO_DIR = Path(os.environ.get("FT_DEMO_DIR", REPO_ROOT / "demo_data"))
SAMPLES = REPO_ROOT / "demo" / "samples"
MAX_BYTES = int(os.environ.get("FT_DEMO_MAX_BYTES", 5_000_000))
MAX_ROWS = int(os.environ.get("FT_DEMO_MAX_ROWS", 5000))
TEST_MAX = int(os.environ.get("FT_DEMO_TEST_MAX", 300))  # small test/val sets keep a run ~25 min, ~$5
VAL_MAX = int(os.environ.get("FT_DEMO_VAL_MAX", 200))
RUNS_PER_VISITOR_PER_DAY = int(os.environ.get("FT_DEMO_RUNS_PER_VISITOR_PER_DAY", 1))  # 0 = no limit
ID_RE = re.compile(r"^demo-[0-9a-f]{8}$")

STAGES = ["load", "split", "baseline", "prepare", "train", "predict", "evaluate", "report"]  # the page's timeline
ATTEMPT_STAGES = {mod.NAME for mod in pipeline.ATTEMPT_STAGES}


class DemoError(ValueError):
    """Shown to the visitor as is."""


# ---------------------------------------------------------------- state


def _dir(run_id: str) -> Path:
    if not ID_RE.match(run_id):
        raise DemoError("unknown run")
    return runs_dir() / run_id


def _read(run_id: str) -> dict[str, Any]:
    p = _dir(run_id) / "demo.json"
    if not p.exists():
        raise DemoError("unknown run")
    return read_json(p)


_lock = threading.RLock()  # re-entrant: approve() checks and updates a run's state under one hold


def _update(run_id: str, **fields: Any) -> dict[str, Any]:
    with _lock:
        p = _dir(run_id) / "demo.json"
        data = _read_json(p, {})
        data.update(fields, updated=now_iso())
        write_json(p, data)
        return data


def _console(run_id: str) -> Console:
    f = (_dir(run_id) / "demo.log").open("a", buffering=1)
    return Console(file=f, width=400, soft_wrap=True, color_system=None, highlight=False, force_terminal=False)


# ---------------------------------------------------------------- upload + plan


def _check_table(name: str, data: bytes) -> int:
    if len(data) > MAX_BYTES:
        raise DemoError(f"file is over {MAX_BYTES // 1_000_000} MB")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise DemoError("file must be UTF-8 text") from e
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if rows and len(rows[0]) < 2:
            raise DemoError("the CSV needs at least two columns: an input and a label")
    else:
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DemoError(f"line {i} isn't valid JSON") from e
                if not isinstance(obj, dict):
                    raise DemoError(f"line {i} isn't a JSON object")
                rows.append(obj)
    if not rows:
        raise DemoError("the file has no rows")
    if len(rows) > MAX_ROWS:
        raise DemoError(f"the demo takes up to {MAX_ROWS:,} rows; this file has {len(rows):,}")
    return len(rows)


def create(filename: str, data: bytes, visitor: str) -> str:
    """Save an upload and start the agent on it (in the background). Returns the run id."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename or "data.csv").name).lower()
    if not name.endswith((".csv", ".jsonl")):
        raise DemoError("upload a .csv or .jsonl file")
    rows = _check_table(name, data)
    run_id = f"demo-{secrets.token_hex(4)}"
    folder = DEMO_DIR / run_id
    folder.mkdir(parents=True)
    (folder / name).write_bytes(data)
    _update(run_id, id=run_id, status="planning", filename=name, rows=rows, visitor=visitor, created=now_iso(),
            dry_run=DRY_RUN)  # fmt: skip
    threading.Thread(target=_plan, args=(run_id, folder), daemon=True).start()
    return run_id


def create_from_sample(sample: str, visitor: str) -> str:
    path = SAMPLES / f"{sample}.csv"
    if not re.fullmatch(r"[a-z0-9-]+", sample) or not path.exists():
        raise DemoError("unknown sample")
    return create(path.name, path.read_bytes(), visitor)


def _plan(run_id: str, folder: Path) -> None:
    console = _console(run_id)
    try:
        console.print(f"agent is looking at {folder.name}/")
        result, _run_dir = plan_folder(folder, run_id, dry_run=DRY_RUN, log=lambda m: console.print(f"  {m}"))
        if result.outcome != "submitted":
            _update(run_id, status="declined", fit=result.fit, fit_label=result.fit_label, reason=result.reason,
                    suggestion=result.suggestion, agent_usd=result.usd, agent_turns=result.turns)  # fmt: skip
            console.print(f"agent declined: {result.reason}")
            return
        task_path = task_path_for(folder)
        cfg = yaml.safe_load(result.task_yaml)
        splits = cfg.setdefault("splits", {})
        splits["test_max"] = min(int(splits.get("test_max", TEST_MAX)), TEST_MAX)  # demo-sized sets
        splits["val_max"] = min(int(splits.get("val_max", VAL_MAX)), VAL_MAX)
        task_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        console.print(f"agent wrote task.yaml after {result.turns} turns")
        ctx = RunContext.create(task_path, dry_run=DRY_RUN, yes=True, run_id=run_id)
        pipeline.run_free_stages(ctx, console)
        summary = checkpoint_summary(ctx, folder)
        _update(run_id, status="planned", plan=summary, rationale=result.rationale, corrections=result.corrections,
                task_yaml=task_path.read_text(), agent_usd=result.usd, agent_turns=result.turns)  # fmt: skip
        console.print("checkpoint ready: waiting for approval")
    except Exception as e:  # shown to the visitor; the run folder keeps the details
        console.print(f"planning failed: {type(e).__name__}: {e}")
        _update(run_id, status="failed", error=_plain(e))


def _plain(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:400]


# ---------------------------------------------------------------- approve + run


_queue: queue.Queue[str] = queue.Queue()
_worker_started = False


def _approvals_path() -> Path:
    return runs_dir() / "demo_approvals.json"


def approve(run_id: str, visitor: str) -> dict[str, Any]:
    with _lock:  # check-and-queue is atomic, so a double click can't queue (and pay for) a run twice
        state = _read(run_id)
        if state["status"] != "planned":
            raise DemoError(f"this run is {state['status']}, not waiting for approval")
        est = (state.get("plan", {}).get("estimate") or {}).get("total_usd") or 0
        if not DRY_RUN and est > budget_left_usd(runs_dir()):
            raise DemoError("the demo's spending budget is used up for now; the featured runs show full results")
        p = _approvals_path()
        log = _read_json(p, {})
        day = datetime.now(UTC).date().isoformat()
        key = f"{day}:{visitor}"
        if RUNS_PER_VISITOR_PER_DAY and log.get(key, 0) >= RUNS_PER_VISITOR_PER_DAY and not DRY_RUN:
            raise DemoError(f"{RUNS_PER_VISITOR_PER_DAY} fine-tune per visitor per day; try again tomorrow")
        log[key] = log.get(key, 0) + 1
        write_json(p, log)
        state = _update(run_id, status="queued", approved=now_iso(), position=_queue.qsize() + 1)
    _queue.put(run_id)
    start_worker()
    return state


def start_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_work, daemon=True).start()


def _work() -> None:
    while True:
        run_id = _queue.get()
        console = _console(run_id)
        try:
            _update(run_id, status="running", started=now_iso(), position=0)
            ctx = RunContext.open(run_id, yes=True)
            ctx.events.emit("checkpoint", choice="approve", via="demo")
            pipeline.run_pipeline(ctx, console)
            _update(run_id, status="done", finished=now_iso())
        except Exception as e:  # the pipeline already tore down any endpoint; tell the visitor what happened
            console.print(f"run failed: {type(e).__name__}: {e}")
            _update(run_id, status="failed", error=_plain(e), finished=now_iso())


def resume_unfinished() -> None:
    """On start-up, put approved runs that didn't finish back in the queue (runs are resumable)."""
    root = runs_dir()
    if not root.exists():
        return
    for p in sorted(root.glob("demo-*/demo.json")):
        state = read_json(p)
        if state.get("status") in ("queued", "running"):
            _update(state["id"], status="queued", position=_queue.qsize() + 1)
            _queue.put(state["id"])
            start_worker()


# ---------------------------------------------------------------- views


def _stage_progress(stages: dict[str, Any]) -> list[dict[str, Any]]:
    """Stage statuses for the timeline. Attempt stages ("<n>/<stage>") show the latest attempt's."""
    if not stages:
        return []
    out = []
    for name in STAGES:
        if name in ATTEMPT_STAGES:
            keys = [k for k in stages if k.endswith(f"/{name}") and k.split("/", 1)[0].isdigit()]
            key = max(keys, key=lambda k: int(k.split("/", 1)[0])) if keys else None
            rec, attempt = (stages[key], int(key.split("/", 1)[0])) if key else ({}, None)
        else:
            rec, attempt = stages.get(name, {}), None
        out.append({"stage": name, "status": rec.get("status", "pending"), "attempt": attempt})
    return out


def _read_json(path: Path, default: Any) -> Any:
    return read_json(path) if path.exists() else default


def view(run_id: str) -> dict[str, Any]:
    """Everything the run page shows: demo state, stage progress, attempts and fixer decisions, results, log."""
    state = _read(run_id)
    run_dir = _dir(run_id)
    pipeline = _read_json(run_dir / "state.json", {})
    log_path = run_dir / "demo.log"
    out = {k: v for k, v in state.items() if k != "visitor"}
    out["log"] = [
        _scrub(_relative(line)) for line in (log_path.read_text().splitlines()[-60:] if log_path.exists() else [])
    ]
    out["progress"] = _stage_progress(pipeline.get("stages", {}))
    out["spent_usd"] = round(sum(e["usd"] for e in pipeline.get("spend", [])), 4)
    events = EventLog(run_dir / "events.jsonl").read()
    out["attempts"] = [
        {
            "attempt": e.get("attempt"),
            "val": e["data"]["val"],
            "passed": e["data"]["passed"],
            "threshold": e["data"]["threshold"],
        }
        for e in events
        if e["type"] == "attempt_result"
    ]
    out["fixes"] = [
        e["data"]["rationale"]
        for e in events
        if e["type"] == "decision" and str(e["data"].get("rationale", "")).startswith("fixer:")
    ]
    if state.get("status") == "done":
        report = _read_json(run_dir / "report.json", {})
        out["results"] = {"verdict": report.get("verdict"), "comparison": report.get("comparison")}
    out["queue_length"] = _queue.qsize()
    return out


def _relative(line: str) -> str:
    """Server file paths out of what visitors see (e.g. the report path the pipeline prints)."""
    for root, short in ((runs_dir(), "runs"), (DEMO_DIR, "uploads"), (REPO_ROOT, ".")):
        line = line.replace(str(root), short)
    return line


def report_data(run_id: str) -> dict[str, Any]:
    p = _dir(run_id) / "report.json"
    if not p.exists() or _read(run_id).get("status") != "done":
        raise DemoError("no report yet")
    return json.loads(_scrub(p.read_text()))


def report_html(run_id: str) -> str:
    p = _dir(run_id) / "report.html"
    if not p.exists():
        raise DemoError("no report yet")
    return _scrub(p.read_text())


ACCOUNT_RE = re.compile(r"[a-z0-9]+-\d{3,5}/(ft-)")  # Together account prefix on model ids


def _scrub(text: str) -> str:
    return ACCOUNT_RE.sub(r"<account>/\1", text)


def status() -> dict[str, Any]:
    return {"dry_run": DRY_RUN, "queue": _queue.qsize(), "budget_left_usd": round(budget_left_usd(runs_dir()), 2),
            "limits": {"rows": MAX_ROWS, "mb": MAX_BYTES / 1e6, "test": TEST_MAX, "val": VAL_MAX,
                       "runs_per_visitor_per_day": RUNS_PER_VISITOR_PER_DAY}}  # fmt: skip
