"""Rank Enron mailboxes by hand-made folders x filed mail (both matter: many folders make the task
interesting, many emails per folder make it learnable), using the real loader and the exclusions and
min_per_folder from examples/enron/task.yaml (so counts match what a run would load).

    .venv/bin/python examples/enron/pick_employee.py [--top 10] [--write]

--write points task.yaml's source.path at the winner. Results are cached in data/enron/survey.json.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ft.config import load_task
from ft.connectors.mailfiles import read_messages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TASK = HERE / "task.yaml"
MAILDIR = ROOT / "data" / "enron" / "maildir"
CACHE = ROOT / "data" / "enron" / "survey.json"


def survey_one(user: str) -> dict:
    src = load_task(TASK).source
    rows, stats = read_messages(src, MAILDIR / user)
    # Under a time split the newest ~20% is test; folders that only start late can't be learned.
    cut = int(len(rows) * 0.7)
    early = {r["folder"] for r in rows[:cut]}
    test = rows[int(len(rows) * 0.8) :]
    seen = sum(r["folder"] in early for r in test) / len(test) if test else 0.0
    return {
        "user": user,
        "folders": len(stats["folders"]),
        "messages": stats["messages"],
        "test_in_known_folders": round(seen, 3),
        "biggest_share": round(max(stats["folders"].values()) / stats["messages"], 3) if rows else 0.0,
        "stats": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if CACHE.exists():
        results = json.loads(CACHE.read_text())
    else:
        users = sorted(p.name for p in MAILDIR.iterdir() if p.is_dir())
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(survey_one, users))
        CACHE.write_text(json.dumps(results, indent=1))
    results.sort(key=lambda r: (-r["folders"] * r["messages"], -r["folders"]))
    print(f"{'mailbox':<16}{'folders':>8}{'filed':>8}{'top share':>11}{'test in known folders':>23}")
    for r in results[: args.top]:
        print(
            f"{r['user']:<16}{r['folders']:>8}{r['messages']:>8}{r['biggest_share']:>11.0%}"
            f"{r['test_in_known_folders']:>23.0%}"
        )
    best = results[0]
    most_folders = max(results, key=lambda r: (r["folders"], r["messages"]))
    print(f"\npick: {best['user']} ({best['folders']} folders, {best['messages']} filed emails)")
    if most_folders["user"] != best["user"]:
        mf = most_folders
        print(f"most folders: {mf['user']} ({mf['folders']} folders, {mf['messages']} emails)")
    if args.write:
        text = TASK.read_text()
        TASK.write_text(re.sub(r"path: data/enron/maildir/\S+.*", f"path: data/enron/maildir/{best['user']}", text))
        print(f"task.yaml now points at data/enron/maildir/{best['user']}")


if __name__ == "__main__":
    main()
