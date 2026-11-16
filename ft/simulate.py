"""`ft simulate-logs` (spec 15.2): run a labeled dataset's inputs through a frontier model with an
"app-style" system prompt (a plausible product prompt, unlike the baseline prompt) and save the calls as
logs in the native format. The dataset's own labels go to a separate truth sidecar, used only to evaluate.

Spend: estimated first and checked against the all-runs budget; the cost is recorded in
runs/<sim id>/plan.json, which the all-runs total counts.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ft.agent.estimate import split_rows
from ft.agent.inspect import Folder
from ft.config import ConfigError, load_task
from ft.context import REPO_ROOT, make_run_id, repo_prices, runs_dir
from ft.events import now_iso
from ft.io import write_json
from ft.providers.base import BaselineSettings, Item, PromptSpec
from ft.spend import (
    REASONING_TOKENS_ESTIMATE,
    Prices,
    SpendError,
    spend_entry,
    total_budget_usd,
    total_real_spend,
)

CHUNK = 400
MAX_OUTPUT_TOKENS = 32

APP_PROMPT = """\
You power the automatic routing in a customer-facing app. Each message below was typed by a real user.
{instruction}
Reply with exactly one of these codes and nothing else, so the app can route it:
{labels}"""


def app_prompt(instruction: str, labels: list[str]) -> str:
    return APP_PROMPT.format(instruction=instruction.strip(), labels="\n".join(labels))


def _estimate(prices: Prices, model: str, effort: str, system: str, rows: list[dict], field: str) -> float:
    price = prices.token_price(model)
    sys_tokens = len(system) / 2  # label lists tokenize densely; err high
    inp = sum(len(str(r.get(field, ""))) / 4 + 8 for r in rows)
    out = len(rows) * (MAX_OUTPUT_TOKENS + REASONING_TOKENS_ESTIMATE.get(effort, 500))
    return price.cost(sys_tokens * len(rows) + inp, sys_tokens * (len(rows) - 1) if sys_tokens >= 1024 else 0, out)


def simulate_logs(
    task_path: Path,
    model: str,
    n: int,
    out_dir: Path,
    *,
    effort: str = "low",
    seed: int = 42,
    dry_run: bool = False,
    confirm: Callable[[str], bool] = lambda _m: True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cfg = load_task(task_path)
    if cfg.task.type != "classify" or cfg.source.type not in ("csv", "jsonl"):
        raise ConfigError("simulate-logs needs a classify task with a csv or jsonl source")
    _by_hint, rows = split_rows(cfg, Folder(REPO_ROOT / cfg.source.path, REPO_ROOT))
    field, label_field = cfg.task.input.field, cfg.task.output.field
    seen, pool = set(), []
    for r in rows:  # one log per distinct input, so the run's split has nothing to dedupe across splits
        text = str(r.get(field) or "").strip()
        if text and str(r.get(label_field) or "").strip() and text not in seen:
            seen.add(text)
            pool.append(r)
    rng = random.Random(seed)
    rng.shuffle(pool)
    picked = pool[:n]
    labels = sorted({str(r[label_field]) for r in pool})
    system = app_prompt(cfg.task.instruction, labels)
    prices = repo_prices(dry_run=dry_run)
    est = _estimate(prices, model, effort, system, picked, field)
    if not dry_run:
        spent, limit = total_real_spend(runs_dir()), total_budget_usd()
        if spent + est > limit:
            raise SpendError(f"simulate-logs: ~${est:.2f} would take all runs to ${spent + est:.2f}, over ${limit:.2f}")
        if not confirm(
            f"simulate-logs: {len(picked):,} calls to {model} ({effort}), estimated ${est:.2f} "
            f"(all runs ${spent:.2f} of ${limit:.2f}). Continue?"
        ):
            raise SpendError("simulate-logs: cancelled by user")

    if dry_run:
        from ft.providers.fake import FakeFrontier, FakeOptions

        frontier = FakeFrontier(FakeOptions(seed=seed, labels=labels))
    else:
        from ft.providers.openai_provider import OpenAIFrontier

        frontier = OpenAIFrontier()
    prompt = PromptSpec(system=system, max_output_tokens=MAX_OUTPUT_TOKENS)
    settings = BaselineSettings(
        model=model, reasoning_effort=effort, concurrency=16, cache_key=f"ft-sim-{cfg.name}"[:64]
    )
    items = [Item(id=f"log-{i:05d}", input=str(r[field]), target=str(r[label_field])) for i, r in enumerate(picked)]
    preds = []
    for start in range(0, len(items), CHUNK):
        preds += asyncio.run(frontier.predict(items[start : start + CHUNK], prompt, settings))
        log(f"simulate-logs: {min(start + CHUNK, len(items))}/{len(items)}")

    price = prices.token_price(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    usd, written, failed = 0.0, 0, 0
    truth = {it.id: it.target for it in items}
    by_id = {it.id: it for it in items}
    with (out_dir / "logs.jsonl").open("w") as f_logs, (out_dir / "truth.jsonl").open("w") as f_truth:
        for p in preds:
            u = p.usage
            usd += price.cost_of(u)
            if p.error or p.prediction_raw is None:
                failed += 1
                continue
            log_row = {
                "id": p.id,
                "timestamp": now_iso(),
                "request": {"model": model, "messages": [{"role": "system", "content": system},
                                                         {"role": "user", "content": by_id[p.id].input}]},
                "response": {"content": p.prediction_raw.strip(),
                             "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                                       "cache_read_tokens": u.cache_read_tokens}},
                "latency_ms": round(p.latency_ms, 1),
                "metadata": {"simulated": True, "source_task": cfg.name, "reasoning_effort": effort},
            }  # fmt: skip
            f_logs.write(json.dumps(log_row, ensure_ascii=False) + "\n")
            f_truth.write(json.dumps({"id": p.id, "truth": truth[p.id]}, ensure_ascii=False) + "\n")
            written += 1

    sim_id = make_run_id(f"sim-{cfg.name}")
    rec_dir = runs_dir() / sim_id
    entry = spend_entry(
        "simulate-logs", "openai", usd, f"{len(preds)} calls to {model} ({effort})", simulated=dry_run, model=model
    )
    # plan.json: where the all-runs total looks for spend that has no run of its own
    plan = {"spend": [{"ts": now_iso(), **entry}], "kind": "simulate-logs", "out": str(out_dir)}
    write_json(rec_dir / "plan.json", plan)
    agree = sum(1 for p in preds if p.prediction_raw and p.prediction_raw.strip() == truth[p.id]) / max(written, 1)
    return {"written": written, "failed": failed, "usd": round(usd, 4), "estimate_usd": round(est, 4),
            "logged_vs_truth": round(agree * 100, 2), "out": str(out_dir), "record": str(rec_dir)}  # fmt: skip
