"""Fireworks provider: LoRA training (Training API), on-demand deployments (REST), tuned inference (OpenAI SDK).

Facts checked on 2026-11-16 against docs.fireworks.ai and the source of fireworks-ai 1.2.33 and
fireworks-training-cookbook 2026.11.13.1 (the recipe's Config fields, its promotion step, the SDK's own
deployment calls). Not yet run against a live account, so models.yaml and prices.yaml keep Fireworks
unverified until a first run:
- managed fine-tuning jobs no longer accept new jobs. Training goes through the Training API, whose loop runs
  in the caller's Python process (`training.recipes.sft_loop` from fireworks-training-cookbook), so there is
  no server-side job to poll. `start` launches that loop as a detached worker (ft/providers/fireworks_train.py)
  and `poll` reads the status file the worker writes, which keeps the Trainer interface and `ft resume`;
- the recipe reads the same chat JSONL we already write, straight from disk: there is no upload step;
- the recipe picks the tokenizer and chat renderer from the base model, and promotes the finished LoRA to
  accounts/<account>/models/<output_model_id>;
- a fine-tune is served on an on-demand deployment (POST /v1/accounts/<account>/deployments with the
  fine-tuned model as `baseModel` and `deploymentShape: default`), billed per GPU-second;
- inference: OpenAI SDK at https://api.fireworks.ai/inference/v1 with
  model = "<model name>#accounts/<account>/deployments/<deployment id>".
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from ft.providers.base import Endpoint, JobStatus, ModelRef, ProviderError, TrainParams
from ft.providers.together_provider import TogetherTuned
from ft.registry import BaseModelInfo

API_BASE_URL = "https://api.fireworks.ai"
INFERENCE_BASE_URL = "https://api.fireworks.ai/inference/v1"
LORA_RANK = 16
MAX_SEQ_LEN = 32768  # serverless needs a concrete bound; longer rows are skipped (the cookbook example's value)
MIN_ROWS, MAX_ROWS = 3, 3_000_000  # dataset size the docs accept


class FireworksError(ProviderError):
    pass


def _api_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise FireworksError("FIREWORKS_API_KEY is not set (add to .env)")
    return key


def _request(method: str, path: str, *, client: httpx.Client | None = None, **kwargs: Any) -> dict[str, Any]:
    """One REST call; returns the parsed JSON body. Errors carry the status code so callers can match on it."""
    headers = {"Authorization": f"Bearer {_api_key()}"}
    if client is None:
        with httpx.Client(base_url=API_BASE_URL, timeout=60) as own:
            resp = own.request(method, path, headers=headers, **kwargs)
    else:
        resp = client.request(method, path, headers=headers, **kwargs)
    if resp.status_code >= 400:
        raise FireworksError(f"{method} {path} failed ({resp.status_code}): {resp.text}"[:1500])
    return resp.json() if resp.content else {}


def account_id(client: httpx.Client | None = None) -> str:
    """FIREWORKS_ACCOUNT_ID, or the one account the API key reaches."""
    env = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if env:
        return env
    accounts = _request("GET", "/v1/accounts", client=client).get("accounts") or []
    if len(accounts) != 1:
        raise FireworksError(f"the API key reaches {len(accounts)} accounts; set FIREWORKS_ACCOUNT_ID")
    return accounts[0]["name"].split("/")[-1]


def model_id(suffix: str) -> str:
    """An output model id Fireworks accepts: 1-63 chars of a-z, 0-9 and hyphens."""
    return re.sub(r"[^a-z0-9-]+", "-", f"ft-{suffix}".lower()).strip("-")[-63:].lstrip("-")


# ---------------------------------------------------------------- training


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FireworksTrainer:
    poll_seconds = 30.0

    def __init__(self, models: dict[str, BaseModelInfo] | None = None) -> None:
        self.models = models or {}

    def preflight(self, base_model: str) -> None:
        _api_key()
        if importlib.util.find_spec("training") is None or importlib.util.find_spec("fireworks") is None:
            raise FireworksError("the Fireworks training SDK isn't installed: uv pip install -e '.[fireworks]'")

    def limits(self, base_model: str) -> dict[str, float]:
        """Fireworks publishes no per-model limits, so these are wide; the fixer's own range still applies."""
        return {"min_learning_rate": 1e-8, "max_learning_rate": 1.0, "max_epochs": 20.0}

    def upload(self, path: Path) -> str:
        """Nothing to upload: the training loop reads the JSONL from disk. Returns the file's path."""
        rows = sum(1 for line in path.open() if line.strip())
        if not MIN_ROWS <= rows <= MAX_ROWS:
            raise FireworksError(f"{path.name} has {rows} rows; Fireworks takes {MIN_ROWS} to {MAX_ROWS:,}")
        return str(path.resolve())

    def estimate(self, file_id: str, params: TrainParams) -> float | None:
        return None  # no estimate endpoint; the train stage uses the local estimate

    def start(self, file_id: str, params: TrainParams) -> str:
        if not params.lora:
            raise FireworksError("serverless training on Fireworks is LoRA only (set train.lora: true)")
        info = self.models.get(params.base_model)
        job_dir = Path(file_id).parent / "fireworks_job"
        job_dir.mkdir(exist_ok=True)
        output_id = model_id(params.suffix)
        job = {
            "base_model": params.base_model,
            "dataset": file_id,
            "epochs": params.epochs,
            "learning_rate": params.learning_rate,
            "lora_rank": LORA_RANK,
            "max_seq_len": MAX_SEQ_LEN,
            "output_model_id": output_id,
            "model": f"accounts/{account_id()}/models/{output_id}",
            "tokenizer_model": info.tokenizer_model if info else None,
            "renderer": info.renderer if info else None,
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2))
        with (job_dir / "worker.log").open("ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "ft.providers.fireworks_train", str(job_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # training outlives this process; `ft resume` polls it
                env=os.environ.copy(),
            )
        (job_dir / "status.json").write_text(json.dumps({"status": "running", "pid": proc.pid}))
        return str(job_dir)

    def poll(self, job_id: str) -> JobStatus:
        job_dir = Path(job_id)
        try:
            st = json.loads((job_dir / "status.json").read_text())
        except (OSError, json.JSONDecodeError):
            return JobStatus(status="running", terminal=False)  # the worker is mid-write
        status = st.get("status", "running")
        if status == "running" and not _alive(st.get("pid")):
            log = job_dir / "worker.log"
            tail = log.read_text(errors="ignore")[-600:] if log.exists() else ""
            return JobStatus(status="failed", terminal=True, error=f"the training process exited: {tail}".strip())
        return JobStatus(status=status, terminal=status in ("completed", "failed"), error=st.get("error"))

    def result(self, job_id: str) -> ModelRef:
        job = json.loads((Path(job_id) / "job.json").read_text())
        return ModelRef(model_object_id=job["model"], base_model=job["base_model"])


# ---------------------------------------------------------------- serving


class FireworksServer:
    poll_seconds = 15.0

    def __init__(self, client: httpx.Client | None = None) -> None:
        self.client = client or httpx.Client(base_url=API_BASE_URL, timeout=60)
        self._account: str | None = None

    @property
    def account(self) -> str:
        if not self._account:
            self._account = account_id(self.client)
        return self._account

    def _deployments(self, suffix: str = "") -> str:
        return f"/v1/accounts/{self.account}/deployments{suffix}"

    def create(self, model: ModelRef, name: str) -> Endpoint:
        """One replica, live-merged. Scaling to zero is off so readiness means a replica is up; the serve
        stage's own teardown is what stops billing."""
        body = {
            "baseModel": model.model_object_id,
            "displayName": name,
            "deploymentShape": "default",
            "minReplicaCount": 1,
            "maxReplicaCount": 1,
        }
        try:
            dep = _request("POST", self._deployments(), client=self.client, params={"deploymentId": name}, json=body)
        except FireworksError as e:
            if "(409)" not in str(e):
                raise
            dep = _request("GET", self._deployments(f"/{name}"), client=self.client)  # already created: reuse it
        dep_name = dep.get("name") or f"accounts/{self.account}/deployments/{name}"
        return Endpoint(
            id=dep_name.split("/")[-1],
            name=name,
            model=f"{model.model_object_id}#{dep_name}",
            deployment_id=dep_name.split("/")[-1],
            hardware_config=dep.get("acceleratorType"),
        )

    def is_ready(self, endpoint: Endpoint) -> bool:
        dep = _request("GET", self._deployments(f"/{endpoint.id}"), client=self.client)
        state = dep.get("state")
        if state in ("FAILED", "DELETED", "DELETING"):
            raise FireworksError(f"deployment {endpoint.id} is {state}: {dep.get('status')}")
        ready = (dep.get("replicaStats") or {}).get("readyReplicaCount", 1)
        return state == "READY" and (ready or 0) >= 1

    def teardown(self, endpoint_id: str) -> None:
        """Delete the deployment. Already gone counts as done."""
        try:
            _request(
                "DELETE", self._deployments(f"/{endpoint_id}"), client=self.client, params={"ignoreChecks": "true"}
            )
        except FireworksError as e:
            if re.search(r"\(404\)|not.?found", str(e), re.I):
                return
            raise

    def list_ft_endpoints(self) -> list[Endpoint]:
        out: list[Endpoint] = []
        token: str | None = None
        while True:
            params = {"pageSize": 200, **({"pageToken": token} if token else {})}
            page = _request("GET", self._deployments(), client=self.client, params=params)
            for dep in page.get("deployments") or []:
                short = dep["name"].split("/")[-1]
                if short.startswith("ft-") and dep.get("state") not in ("DELETED", "DELETING"):
                    out.append(Endpoint(id=short, name=short, model=f"{dep.get('baseModel', '')}#{dep['name']}"))
            token = page.get("nextPageToken")
            if not token:
                return out


# ---------------------------------------------------------------- tuned inference


class FireworksTuned(TogetherTuned):
    """The same OpenAI-SDK calls as Together, at Fireworks' base URL. Thinking is switched off with
    `reasoning_effort: none` (their documented switch for Qwen models)."""

    base_url = INFERENCE_BASE_URL
    key_env = "FIREWORKS_API_KEY"
    thinking_off = {"reasoning_effort": "none"}
