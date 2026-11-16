"""state.json: stage status, attempts, external resource IDs, spend. Written atomically on every change.

External IDs (file, job, endpoint, deployment) are saved the moment they exist so that
`ft teardown` and `ft resume` work after a crash (spec 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ft.events import now_iso
from ft.io import read_json, write_json


class State:
    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    @classmethod
    def create(cls, path: Path, run_id: str, *, dry_run: bool, task_hash: str) -> State:
        data = {
            "run_id": run_id,
            "created_at": now_iso(),
            "dry_run": dry_run,
            "task_hash": task_hash,
            "stages": {},
            "attempts": {},
            "endpoints": {},
            "spend": [],
            "final_attempt": None,
        }
        st = cls(path, data)
        st.save()
        return st

    @classmethod
    def load(cls, path: Path) -> State:
        return cls(path, read_json(path))

    def save(self) -> None:
        write_json(self.path, self.data)

    # ---- stages (run-level stages keyed by name; attempt stages keyed "<n>/<name>")

    @staticmethod
    def key(stage: str, attempt: int | None) -> str:
        return stage if attempt is None else f"{attempt}/{stage}"

    def stage(self, stage: str, attempt: int | None = None) -> dict[str, Any]:
        return self.data["stages"].get(self.key(stage, attempt), {})

    def stage_done(self, stage: str, attempt: int | None, input_hash: str) -> bool:
        s = self.stage(stage, attempt)
        return s.get("status") == "done" and s.get("input_hash") == input_hash

    def mark_stage(self, stage: str, attempt: int | None, status: str, **extra: Any) -> None:
        k = self.key(stage, attempt)
        rec = self.data["stages"].setdefault(k, {})
        rec.update(status=status, updated_at=now_iso(), **extra)
        self.save()

    def invalidate_from(self, keys: list[str]) -> None:
        for k in keys:
            self.data["stages"].pop(k, None)
        self.save()

    # ---- attempts

    def attempt(self, n: int) -> dict[str, Any]:
        return self.data["attempts"].setdefault(str(n), {"n": n, "external": {}})

    def set_external(self, n: int, **ids: Any) -> None:
        self.attempt(n)["external"].update(ids)
        self.save()

    def external(self, n: int) -> dict[str, Any]:
        return self.attempt(n)["external"]

    # ---- endpoints

    def record_endpoint(self, name: str, **fields: Any) -> None:
        self.data["endpoints"].setdefault(name, {}).update(fields)
        self.save()

    def live_endpoints(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self.data["endpoints"].items() if not v.get("torn_down_at")}

    # ---- spend

    def add_spend(self, entry: dict[str, Any]) -> None:
        self.data["spend"].append({"ts": now_iso(), **entry})
        self.save()

    def spent(self, *, include_simulated: bool = True) -> float:
        return sum(e["usd"] for e in self.data["spend"] if include_simulated or not e.get("simulated"))
