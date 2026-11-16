"""Glue between the CLI and the agent: plan a folder into a run folder, and the human checkpoint (spec 12)."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ft.agent.agent import PlanResult, plan
from ft.agent.estimate import EstimateError, estimate_run
from ft.agent.inspect import Folder
from ft.context import REPO_ROOT, make_run_id, repo_models, repo_prices, runs_dir
from ft.io import read_json, read_jsonl
from ft.providers.base import AgentLLM


def agent_llm(dry_run: bool) -> AgentLLM:
    if dry_run:
        from ft.agent.fake import FakeAgent

        return FakeAgent()
    from ft.providers.openai_provider import OpenAIAgent

    return OpenAIAgent()


def default_run_id(data_dir: Path) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in data_dir.resolve().name.lower()).strip("-")[:30]
    return make_run_id(slug or "data")


def plan_folder(
    data_dir: Path, run_id: str, *, dry_run: bool, log: Callable[[str], None] = print
) -> tuple[PlanResult, Path]:
    run_dir = runs_dir() / run_id
    if (run_dir / "state.json").exists() or (run_dir / "plan.json").exists():
        raise FileExistsError(f"run {run_id} already exists")
    run_dir.mkdir(parents=True, exist_ok=True)
    folder = Folder(data_dir, REPO_ROOT)
    result = plan(
        folder,
        agent_llm(dry_run),
        repo_prices(dry_run=dry_run),
        repo_models(),
        run_dir,
        runs_dir(),
        log=log,
    )
    return result, run_dir


def checkpoint_summary(ctx, data_dir: Path, n_examples: int = 10) -> dict[str, Any]:
    """What the human approves: input -> output, label source, split sizes, base model, cost, and 10
    random test examples with their expected answers (spec 12). Computed after the free local stages, so
    split sizes are the real ones."""
    cfg = ctx.cfg
    splits = read_json(ctx.data_dir / "splits.json")
    test = read_jsonl(ctx.data_dir / "test.jsonl")
    picks = random.Random(cfg.splits.seed).sample(test, min(n_examples, len(test)))
    try:
        est = estimate_run(cfg, Folder(data_dir, REPO_ROOT), ctx.prices, ctx.models)
    except EstimateError as e:
        est = {"total_usd": None, "lines": [], "note": str(e)}
    out = cfg.task.output
    target = f"{out.field} ({'labels: ' + str(out.labels) if cfg.task.type == 'classify' else 'JSON fields'})"
    return {
        "task": f"{cfg.task.type}: {cfg.task.input.kind} '{cfg.task.input.field}' -> {target}",
        "instruction": cfg.task.instruction,
        "label_source": cfg.targets.from_,
        "splits": splits["sizes"],
        "split_method": splits["method"],
        "labels": len(ctx.labels()) if cfg.task.type == "classify" else None,
        "base_model": cfg.train.base_model,
        "metric": cfg.eval.primary_metric,
        "baselines": [b.name for b in cfg.baselines],
        "estimate": est,
        "budget": cfg.budget.max_spend_usd,
        "examples": [{"input": _show(ctx, r), "expected": r["target"]} for r in picks],
        "warnings": splits.get("warnings", []),
    }


def _show(ctx, record: dict[str, Any]) -> str:
    if ctx.cfg.task.input.kind == "image":
        return f"[image {record['input']}]"
    text = " ".join(str(record["input"]).split())
    return text if len(text) <= 160 else text[:159] + "…"


def task_path_for(data_dir: Path) -> Path:
    return Path(data_dir) / "task.yaml"
