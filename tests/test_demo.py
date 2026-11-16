"""The live demo (demo/), driven through its API in dry-run mode: fake models, no spend."""

import importlib
import time

import pytest
from fastapi.testclient import TestClient

from demo.build_featured import ACCOUNT_RE


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_DEMO_DRY_RUN", "1")
    monkeypatch.setenv("FT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("FT_DEMO_DIR", str(tmp_path / "uploads"))
    import demo.jobs
    import demo.server

    importlib.reload(demo.jobs)
    importlib.reload(demo.server)
    return TestClient(demo.server.app)


def _wait(c, rid, busy, seconds=120):
    for _ in range(seconds * 4):
        d = c.get(f"/api/runs/{rid}").json()
        if d["status"] not in busy:
            return d
        time.sleep(0.25)
    raise AssertionError(f"run stuck in {d['status']}")


def test_sample_runs_from_plan_to_report(client):
    rid = client.post("/api/runs", data={"sample": "banking77"}).json()["id"]
    d = _wait(client, rid, ("planning",))
    assert d["status"] == "planned" and d["plan"]["splits"]["test"] <= 300 and d["plan"]["estimate"]["total_usd"] > 0
    assert client.post(f"/api/runs/{rid}/approve").status_code == 200
    d = _wait(client, rid, ("queued", "running"), 300)
    assert d["status"] == "done", d.get("error")
    assert {p["status"] for p in d["progress"]} == {"done"} and d["results"]["verdict"]["sentence"]
    assert client.get(f"/api/runs/{rid}/report").status_code == 200
    assert "not waiting" in client.post(f"/api/runs/{rid}/approve").json()["error"]


def test_small_file_is_declined(client):
    rid = client.post("/api/runs", data={"sample": "too-small"}).json()["id"]
    d = _wait(client, rid, ("planning",))
    assert d["status"] == "declined" and "60 labeled rows" in d["reason"] and d["suggestion"]


def test_bad_uploads_are_refused(client):
    assert "csv or .jsonl" in client.post("/api/runs", files={"file": ("x.txt", b"a,b\n1,2\n")}).json()["error"]
    assert "two columns" in client.post("/api/runs", files={"file": ("x.csv", b"a\n1\n")}).json()["error"]
    assert "valid JSON" in client.post("/api/runs", files={"file": ("x.jsonl", b"{oops\n")}).json()["error"]
    assert client.get("/api/runs/demo-zzzz").json()["error"] == "unknown run"
    assert client.post("/api/runs", data={"sample": "../etc"}).json()["error"] == "unknown sample"


def test_featured_and_pages(client):
    assert client.get("/").status_code == 200 and client.get("/run/demo-0123abcd").status_code == 200
    featured = client.get("/api/featured").json()
    assert {f["slug"] for f in featured} >= {"banking77", "ledgar"}
    for f in featured:  # report pages are app pages; their data and the raw HTML are served separately, scrubbed
        assert client.get(f["report"]).status_code == 200
        data = client.get(f"/api/featured/{f['slug']}/report")
        raw = client.get(f"/featured/{f['slug']}/raw")
        assert data.json()["comparison"] and raw.status_code == 200
        assert not ACCOUNT_RE.search(data.text) and not ACCOUNT_RE.search(raw.text)
    assert client.get("/api/featured/../etc/report").status_code in (400, 404)


def test_double_approve_queues_once(client):
    rid = client.post("/api/runs", data={"sample": "banking77"}).json()["id"]
    _wait(client, rid, ("planning",))
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:
        codes = list(pool.map(lambda _: client.post(f"/api/runs/{rid}/approve").status_code, range(4)))
    assert codes.count(200) == 1 and codes.count(400) == 3
    _wait(client, rid, ("queued", "running"), 300)
