from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ft.config import parse_task

REPO = Path(__file__).resolve().parent.parent
TINY = REPO / "examples" / "fixtures" / "tiny_intents" / "task.yaml"


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    """Every test writes runs into its own temp folder, never ./runs."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("FT_RUNS_DIR", str(runs))
    return runs


@pytest.fixture
def tiny_raw() -> dict:
    return yaml.safe_load(TINY.read_text())


@pytest.fixture
def make_cfg(tiny_raw):
    def _make(**overrides):
        raw = copy.deepcopy(tiny_raw)
        for dotted, value in overrides.items():
            node = raw
            *parents, leaf = dotted.split("__")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return parse_task(raw)

    return _make


@pytest.fixture
def tiny_task_path(tmp_path, tiny_raw):
    """A writable copy of the tiny fixture's task.yaml (data paths stay repo-relative)."""

    def _write(**top_level):
        raw = copy.deepcopy(tiny_raw)
        raw.update(top_level)
        p = tmp_path / f"task-{len(list(tmp_path.glob('task-*.yaml')))}.yaml"
        p.write_text(yaml.safe_dump(raw))
        return p

    return _write
