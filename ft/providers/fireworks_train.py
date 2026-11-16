"""The Fireworks training worker: one supervised LoRA fine-tune, run as its own process.

Fireworks' Training API runs the training loop in the caller's process, so FireworksTrainer.start launches
this module detached and polls the status file it writes:

    python -m ft.providers.fireworks_train <job_dir>

<job_dir>/job.json holds the settings; <job_dir>/status.json ends as "completed" or "failed" (with the error).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def write_status(job_dir: Path, **status: Any) -> None:
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps({**status, "pid": os.getpid()}))
    tmp.replace(job_dir / "status.json")  # atomic: the poller never reads half a file


def recipe_config(job: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    """Arguments for the cookbook's `sft_loop.Config`, from job.json."""
    cfg = {
        "serverless": True,
        "base_model": job["base_model"],
        "dataset": job["dataset"],
        "lora_rank": job["lora_rank"],
        "max_seq_len": job["max_seq_len"],
        "epochs": job["epochs"],
        "learning_rate": job["learning_rate"],
        "log_path": str(job_dir / "log"),
        "output_model_id": job["output_model_id"],
    }
    if job.get("tokenizer_model"):
        cfg["tokenizer_model"] = job["tokenizer_model"]
    if job.get("renderer"):
        cfg["renderer_name"] = job["renderer"]
    return cfg


def run(job_dir: Path) -> None:
    job = json.loads((job_dir / "job.json").read_text())
    write_status(job_dir, status="running")
    try:
        from training.recipes import sft_loop  # fireworks-training-cookbook

        sft_loop.main(sft_loop.Config(**recipe_config(job, job_dir)))
    except BaseException as e:  # noqa: BLE001 (a killed or crashed loop must still leave a status)
        write_status(job_dir, status="failed", error=f"{type(e).__name__}: {e}"[:1500])
        raise
    write_status(job_dir, status="completed")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
