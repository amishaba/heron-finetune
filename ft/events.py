"""Append-only events.jsonl: stage starts/ends, agent decisions, paid call summaries, interventions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: Path):
        self.path = path

    def emit(self, type: str, *, stage: str | None = None, attempt: int | None = None, **data: Any) -> None:
        row: dict[str, Any] = {"ts": now_iso(), "type": type}
        if stage is not None:
            row["stage"] = stage
        if attempt is not None:
            row["attempt"] = attempt
        if data:
            row["data"] = data
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [e for e in self.read() if e["type"] == type]
