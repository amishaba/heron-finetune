"""The live demo's web server: the page, plus a small JSON API over demo/jobs.py.

Run locally:  .venv/bin/uvicorn demo.server:app --reload       (FT_DEMO_DRY_RUN=1 for fakes, no spend)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from demo import jobs  # noqa: E402  (after .env, so keys and budgets are set)

WEB = Path(__file__).parent / "web"
FEATURED = Path(__file__).parent / "featured"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    jobs.resume_unfinished()  # approved runs that didn't finish before a restart go back in the queue
    yield


app = FastAPI(title="Heron demo", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


def _visitor(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown"))[:64]


@app.exception_handler(jobs.DemoError)
def _demo_error(_request: Request, e: jobs.DemoError) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/run/{run_id}")
def run_page(run_id: str) -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status() -> dict:
    return jobs.status()


@app.post("/api/runs")
async def create_run(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    sample: Annotated[str | None, Form()] = None,
) -> dict:
    visitor = _visitor(request)
    if sample:
        return {"id": jobs.create_from_sample(sample, visitor)}
    if file is None:
        raise jobs.DemoError("upload a file or pick a sample")
    data = await file.read(jobs.MAX_BYTES + 1)
    return {"id": jobs.create(file.filename or "data.csv", data, visitor)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return jobs.view(run_id)


@app.post("/api/runs/{run_id}/approve")
def approve(run_id: str, request: Request) -> dict:
    jobs.approve(run_id, _visitor(request))
    return jobs.view(run_id)


@app.get("/api/runs/{run_id}/report", response_class=HTMLResponse)
def report(run_id: str) -> str:
    return jobs.report_html(run_id)


@app.get("/api/featured")
def featured() -> list:
    p = FEATURED / "featured.json"
    return json.loads(p.read_text()) if p.exists() else []


def _featured_file(slug: str, suffix: str) -> Path:
    if not slug.replace("-", "").isalnum():
        raise jobs.DemoError("unknown report")
    p = FEATURED / f"{slug}{suffix}"
    if not p.exists():
        raise jobs.DemoError("unknown report")
    return p


@app.get("/api/featured/{slug}/report")
def featured_report_data(slug: str) -> dict:
    return json.loads(_featured_file(slug, ".report.json").read_text())


@app.get("/api/runs/{run_id}/report.json")
def run_report_data(run_id: str) -> dict:
    return jobs.report_data(run_id)


@app.get("/featured/{slug}/raw", response_class=HTMLResponse)
def featured_report_raw(slug: str) -> str:
    """The pipeline's own standalone HTML report, as generated."""
    return _featured_file(slug, ".html").read_text()


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str) -> FileResponse:
    """Any other page path serves the app (its router sends unknown paths home); unknown API paths stay 404."""
    if path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404)
    return FileResponse(WEB / "index.html")
