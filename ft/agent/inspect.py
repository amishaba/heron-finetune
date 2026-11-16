"""Look at a raw data folder before any task.yaml exists: what the agent's tools read (spec 12).

Pure local code. Reads CSV / JSONL / JSON tables (capped rows), counts images, notices mail-like files,
and profiles every column with the same `field_profile` the pipeline's profile stage uses.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ft.stages.load import SPLIT_NAMES
from ft.stages.profile import field_profile

TABLE_SUFFIXES = {".csv", ".jsonl", ".json"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ROWS = 200_000  # per table, for profiling
MAX_FILES = 2_000_000
CELL_CHARS = 1_000  # sample values are truncated to this (spec 12)


class InspectError(ValueError):
    pass


def read_table(path: Path, limit: int = MAX_ROWS) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [row for _, row in zip(range(limit), csv.DictReader(f), strict=False)]
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if len(rows) >= limit:
                    break
                if line.strip():
                    obj = json.loads(line)
                    rows.append(obj if isinstance(obj, dict) else {"value": obj})
        return rows
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):  # {"data": [...]} or similar: take the first list of dicts
        obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
    return [r if isinstance(r, dict) else {"value": r} for r in obj[:limit]]


def _count_rows(path: Path) -> int:
    if path.suffix == ".json":
        return len(read_table(path, limit=10**9))
    with path.open("rb") as f:
        n = sum(1 for line in f if line.strip())
    return n - 1 if path.suffix == ".csv" else n


def _looks_like_mail(path: Path) -> bool:
    try:
        head = path.read_bytes()[:2000].decode("latin-1")
    except OSError:
        return False
    return head.startswith("From ") or ("Message-ID:" in head and "Subject:" in head)


def _json_column(values: list[Any]) -> bool:
    sample = [v for v in values[:50] if isinstance(v, str) and v.strip()]
    if not sample:
        return any(isinstance(v, dict | list) for v in values[:50])
    try:
        return all(isinstance(json.loads(v), dict | list) for v in sample)
    except (json.JSONDecodeError, TypeError):
        return False


class Folder:
    """A raw data folder, scanned once; tools read from this."""

    def __init__(self, root: Path, repo_root: Path):
        if not root.is_dir():
            raise InspectError(f"{root} is not a folder")
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.files: list[Path] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.root).parts):
                self.files.append(p)
                if len(self.files) >= MAX_FILES:
                    break
        self.tables = [p for p in self.files if p.suffix.lower() in TABLE_SUFFIXES]
        self.images = [p for p in self.files if p.suffix.lower() in IMAGE_SUFFIXES]
        self._cache: dict[Path, list[dict[str, Any]]] = {}

    # ---- paths: the agent names files relative to the folder; task.yaml paths are repo-relative

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def repo_path(self, rel: str = "") -> str:
        p = (self.root / rel).resolve()
        try:
            return str(p.relative_to(self.repo_root))
        except ValueError:
            return str(p)

    def file(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise InspectError(f"{rel} is outside the data folder")
        if not p.is_file():
            raise InspectError(f"no file {rel!r}; files: {[self.rel(t) for t in self.tables][:20]}")
        return p

    def rows(self, rel: str) -> list[dict[str, Any]]:
        p = self.file(rel)
        if p.suffix.lower() not in TABLE_SUFFIXES:
            raise InspectError(f"{rel} is not a CSV/JSONL/JSON table")
        if p not in self._cache:
            self._cache[p] = read_table(p)
        return self._cache[p]

    # ---- the profile (get_profile)

    def profile(self) -> dict[str, Any]:
        tables = {}
        for t in self.tables[:20]:
            rows = self.rows(self.rel(t))
            cols = list(rows[0]) if rows else []
            tables[self.rel(t)] = {
                "rows": _count_rows(t),
                "split_hint": SPLIT_NAMES.get(t.stem.lower()),
                "columns": {
                    c: {**field_profile([r.get(c) for r in rows]), "json": _json_column([r.get(c) for r in rows])}
                    for c in cols
                },
            }
            for col in tables[self.rel(t)]["columns"].values():  # keep the profile compact
                col["top_values"] = [[str(v)[:120], n] for v, n in col["top_values"][:15]]
        other = [p for p in self.files if p not in self.tables and p not in self.images]
        mail_like = sum(_looks_like_mail(p) for p in random.Random(0).sample(other, min(200, len(other))))
        return {
            "folder": self.repo_path(),
            "files": len(self.files),
            "by_extension": dict(Counter(p.suffix.lower() or "(none)" for p in self.files).most_common(12)),
            "tables": tables,
            "images": self._image_profile(),
            "mail_like_files_in_sample_of_200": mail_like,
        }

    def _image_profile(self) -> dict[str, Any] | None:
        if not self.images:
            return None
        from PIL import Image

        sizes = []
        for p in random.Random(0).sample(self.images, min(50, len(self.images))):
            try:
                with Image.open(p) as im:
                    sizes.append(im.size)
            except OSError:
                continue
        by_dir = Counter(str(p.parent.relative_to(self.root)) for p in self.images)
        return {
            "count": len(self.images),
            "folders": dict(by_dir.most_common(10)),
            "median_size": [int(statistics.median(s[0] for s in sizes)), int(statistics.median(s[1] for s in sizes))]
            if sizes
            else None,
            "sample_paths": [self.rel(p) for p in random.Random(1).sample(self.images, min(6, len(self.images)))],
        }

    # ---- sample_rows / value_counts

    def sample(self, rel: str, n: int, where: dict[str, Any] | None) -> list[dict[str, Any]]:
        rows = self.rows(rel)
        if where:
            col, want = where["column"], str(where["equals"])
            rows = [r for r in rows if str(r.get(col)) == want]
        picked = random.Random(len(rows)).sample(rows, min(max(1, n), 20, len(rows)))
        return [{k: _clip(v) for k, v in r.items()} for r in picked]

    def value_counts(self, rel: str, column: str, top: int) -> dict[str, Any]:
        rows = self.rows(rel)
        if rows and column not in rows[0]:
            raise InspectError(f"{rel} has no column {column!r}; columns: {list(rows[0])}")
        counts = Counter(_clip(r.get(column), 200) for r in rows)
        return {"rows": len(rows), "distinct": len(counts), "top": counts.most_common(min(top, 100))}


def _clip(v: Any, n: int = CELL_CHARS) -> Any:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else v
    return s[:n] + "…" if isinstance(s, str) and len(s) > n else s
