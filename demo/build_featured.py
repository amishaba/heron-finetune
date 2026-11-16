"""Snapshot finished real runs for the demo's "featured runs" (shown to every visitor, no spend).

Reads runs/<id>/report.json + report.html for the runs below and writes demo/featured/featured.json,
demo/featured/<slug>.report.json (the report page's data) and <slug>.html (the standalone report),
account names scrubbed. Numbers are copied from the reports, never typed.

The email run's report quotes real messages from the Enron corpus. The snapshot leaves them out, and the build
stops if an email address or a local path reaches demo/featured.

    .venv/bin/python demo/build_featured.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ft.stages.report import render_html

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = Path(__file__).parent / "featured"

FEATURED = [
    ("banking77", "proof-banking77", "Banking support intents · 77 classes",
     "Raw folder → the agent plans it → fine-tune → benchmark, with no human edits."),
    ("ledgar", "20261017-1702-ledgar", "Contract clause types · 100 classes",
     "A dataset the pipeline had never seen. The agent capped training data to fit the budget."),
    ("self-healing", "m4-banking77-weak-2", "Self-healing: banking intents from a broken config",
     "Attempt 1 scored 0.8. The fixer read the diagnosis, changed three settings, and attempt 2 passed."),
    ("email", "proof-enron", "Email folders of one Enron employee · split by time",
     "Retrained on the newest month before shipping: +5.9 macro-F1 on test. GPT few-shot still leads."
     " Quoted emails are left out of this public copy."),
]  # fmt: skip

QUOTES_REAL_EMAIL = {"email"}  # featured runs whose reports quote real people's messages
ACCOUNT_RE = re.compile(r"[a-z0-9]+-\d{3,5}/(ft-)")
MAILBOX_RE = re.compile(r"data/enron/maildir/[\w.-]+")
ADDRESS_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
FORBIDDEN_RE = re.compile(r"/Users/|/home/|maildir/")


def scrub(text: str) -> str:
    return ACCOUNT_RE.sub(r"<account>/\1", text)


def public_report(report: dict[str, Any]) -> dict[str, Any]:
    """A copy of a report without quoted messages or the mailbox's owner."""
    out = json.loads(json.dumps(report))
    for confusion in out["errors"].get("confusions", []):
        confusion["examples"] = []
    redaction = out["privacy"]["redaction"]
    redaction["stays_local"] = MAILBOX_RE.sub("one person's mailbox", redaction["stays_local"])
    return out


def check_public(out: Path, allowed: frozenset[str] = frozenset()) -> None:
    """Stop the build if anything private reached the output."""
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        text = path.read_text(errors="ignore")  # the icons are binary
        strays = sorted(set(ADDRESS_RE.findall(text)) - allowed)
        if strays:
            raise ValueError(f"{path}: email addresses in public output: {strays[:5]}")
        forbidden = FORBIDDEN_RE.search(text)
        if forbidden:
            raise ValueError(f"{path}: {forbidden.group()!r} in public output")


def summarize(slug: str, run_id: str, title: str, story: str) -> dict:
    r = json.loads((RUNS / run_id / "report.json").read_text())
    rows = [
        {k: row.get(k) for k in ("model", "kind", "primary", "ci95", "accuracy", "macro_f1", "cost_per_1k", "p50_ms")}
        for row in r["comparison"]
    ]
    v = r.get("verdict") or {}
    attempts = [
        {"n": a["n"], "val": a["val"], "passed": a["passed"], "test": a["test"], "changes": a.get("changes"),
         "by_fixer": a.get("by_fixer")}
        for a in r.get("attempts", [])
    ]  # fmt: skip
    return {
        "slug": slug,
        "title": title,
        "story": story,
        "dataset": r["task"]["name"],
        "primary_metric": r["primary_metric"],
        "n_test": r["setup"]["splits"]["test"],
        "n_train": r["setup"]["splits"]["train"],
        "comparison": rows,
        "sentence": scrub(v.get("sentence", "")),
        "word": v.get("word"),
        "diff": v.get("diff"),
        "diff_ci95": v.get("diff_ci95"),
        "attempts": attempts,
        "retrain": r.get("retrain"),
        "agent_rationale": r["setup"].get("agent_rationale", []),
        "interventions": r["interventions"]["count"],
        "spend_usd": r["spend"]["total_usd"],
        "report": f"/featured/{slug}",
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    out = []
    for slug, run_id, title, story in FEATURED:
        if not (RUNS / run_id / "report.json").exists():
            print(f"skip {run_id}: no report")
            continue
        out.append(summarize(slug, run_id, title, story))
        report_json = scrub((RUNS / run_id / "report.json").read_text())
        report_html = scrub((RUNS / run_id / "report.html").read_text())
        if slug in QUOTES_REAL_EMAIL:
            report = public_report(json.loads(report_json))
            report_json, report_html = json.dumps(report, indent=2), render_html(report)
        (OUT / f"{slug}.html").write_text(report_html)
        (OUT / f"{slug}.report.json").write_text(report_json)
        print(f"{slug}: {run_id}")
    (OUT / "featured.json").write_text(json.dumps(out, indent=2))
    check_public(OUT)


if __name__ == "__main__":
    main()
