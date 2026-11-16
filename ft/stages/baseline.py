"""baseline: run every frontier model in `baselines` on val and test (spec 9.6).

Each baseline is cached on its own (`baselines/<name>/done.json`, keyed by its settings, prompt and the
val/test data), so adding a baseline to a finished run only pays for the new one.

`few_shot: N` puts N training examples per label in the system prompt. They are drawn once from
train (never val/test), saved to `few_shot.json`, and sit in the stable prefix so OpenAI's automatic
prompt caching covers them (cached input is ~10x cheaper).

Spend guard for cached prompts: the no-cache worst case can be ~10x the expected cost, so the stage is
checked against the cached estimate plus the worst case for the first chunk of calls, then every chunk
projects the stage total from what was actually billed and stops before the run cap would be passed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from ft.config import LOGGED_SELF_AGREEMENT, ConfigError, stable_hash
from ft.events import now_iso
from ft.io import read_json, read_jsonl, write_json, write_jsonl
from ft.providers.base import BaselineSettings, Item, Prediction, PromptSpec
from ft.spend import OPENAI_TOKENS_PER_PATCH, REASONING_TOKENS_ESTIMATE, SpendError, SpendStop, frontier_cost_per_1k
from ft.stages import (
    CHUNK,
    StageResult,
    TooManyFailures,
    predict_split,
    prediction_row,
    run_predictions,
    upstream,
)
from ft.stages.evaluate import RETEST_DIR, compute_metrics

NAME = "baseline"
PROVIDER = "openai"


def input_hash(ctx) -> str:
    c = ctx.cfg
    contexts = [ctx.resolve(b.context_file).read_text() if b.context_file else None for b in c.baselines]
    return stable_hash(
        [upstream(ctx, "label", "split"), [b.model_dump() for b in c.baselines], contexts, c.task.instruction]
    )


def select_examples(ctx, n_per_label: int) -> list[dict[str, Any]]:
    """N train examples per label, seeded; order: label order, then sampled order."""
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in read_jsonl(ctx.data_dir / "train.jsonl"):
        by_label[str(r["target"])].append(r)
    rng = random.Random(f"{ctx.cfg.splits.seed}-few-shot")
    out = []
    for label in ctx.labels():
        rows = by_label.get(label, [])
        for r in rng.sample(rows, min(n_per_label, len(rows))):
            out.append({"id": r["id"], "input": r["input"], "target": r["target"]})
    return out


def examples_for(ctx, b, d) -> list[dict[str, Any]] | None:
    if not b.few_shot:
        return None
    path = d / "few_shot.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("n_per_label") == b.few_shot and cached.get("data") == upstream(ctx, "split"):
            return cached["examples"]
    examples = select_examples(ctx, b.few_shot)
    write_json(path, {"n_per_label": b.few_shot, "data": upstream(ctx, "split"), "examples": examples})
    return examples


def baseline_key(ctx, b, prompt) -> str:
    """Identity of one baseline's results: its settings, exact prompt, and the val/test data."""
    return stable_hash(
        [
            upstream(ctx, "label", "split"),
            {"model": b.model, "effort": b.reasoning_effort, "few_shot": b.few_shot},
            prompt.system,
            prompt.json_schema,
            prompt.max_output_tokens,
        ]
    )


def estimate_usd(ctx, b, system: str, items: list[Item], max_out: int) -> tuple[float, float]:
    """(low, high): high assumes no prompt caching; low assumes the system prompt is cached after the
    first call (only possible when it's >= 1,024 tokens)."""
    from ft.images import patches

    price = ctx.prices.token_price(b.model)
    task = ctx.task_type()
    sys_tokens = len(system) // 2  # label lists tokenize densely (~2 chars/token measured); err high
    user_tokens = sum(
        len(it.input) // 4 + 8 + (round(patches(it.image) * OPENAI_TOKENS_PER_PATCH) if it.image else 0) for it in items
    )
    # The answer: a label, or a JSON object about as long as the targets (~3 chars/token for JSON).
    answer = max(min(max_out, 32), round(sum(len(task.train_target(it.target)) for it in items) / len(items) / 3) + 8)
    out_tokens = len(items) * (answer + REASONING_TOKENS_ESTIMATE.get(b.reasoning_effort, 100))
    high = price.cost(sys_tokens * len(items) + user_tokens, 0, out_tokens)
    if sys_tokens < 1024:
        return high, high
    cached = sys_tokens * (len(items) - 1)
    low = price.cost(sys_tokens * len(items) + user_tokens, cached, out_tokens, cache_write_tokens=sys_tokens)
    return low, high


RETEST_MAX_TOKENS = 32


def retest_logged(ctx) -> tuple[str, list]:
    """Logs mode, self-agreement bar: re-ask the logged model the val inputs with its own logged prompt and
    settings, and score how often it gives the same answer again. Cached by data and settings."""
    info = read_json(ctx.data_dir / "logged_prompt.json")
    model, system = info.get("model"), info.get("system_prompt")
    if not system or not model or len(info.get("models", {})) != 1:
        raise ConfigError(
            "the self-agreement bar needs one logged model and one system prompt across the logs; "
            "use pass_bar.min_value instead"
        )
    try:
        price = ctx.prices.token_price(model)
    except SpendError as e:
        raise ConfigError(f"can't re-ask the logged model {model} ({e}); use pass_bar.min_value instead") from e
    d = ctx.baseline_dir(RETEST_DIR)
    key = stable_hash([upstream(ctx, "label", "split"), info])
    done = d / "done.json"
    if done.exists() and read_json(done).get("key") == key:
        return f"logged model self-agreement {_primary(d, 'val')} (cached)", [d / "metrics.val.json"]
    records = read_jsonl(ctx.data_dir / "val.jsonl")
    items = [Item(id=r["id"], input=r["input"], target=r["target"]) for r in records]
    est = price.cost(len(items) * (len(system) / 2 + 60), 0, len(items) * (RETEST_MAX_TOKENS + 500))
    ctx.spend.check(NAME, PROVIDER, est, f"re-ask the logged model {model} on {len(items)} val inputs (self-agreement)")
    settings = BaselineSettings(
        model=model,
        reasoning_effort=info.get("reasoning_effort"),
        concurrency=ctx.cfg.eval.concurrency,
        cache_key=f"ft-{ctx.run_id}-retest"[:64],
    )
    prompt = PromptSpec(system=system, max_output_tokens=RETEST_MAX_TOKENS)
    preds = run_predictions(ctx, lambda part: ctx.providers.frontier.predict(part, prompt, settings), items, "retest")
    by = {r["id"]: r["target"] for r in records}
    usd = sum(price.cost_of(p.usage) for p in preds)
    same = sum(1 for p in preds if (p.prediction_raw or "").strip() == by[p.id])
    value = round(same / max(len(preds), 1) * 100, 2)
    write_jsonl(d / "val.predictions.jsonl", [{"id": p.id, "logged": by[p.id], "retest": p.prediction_raw,
                                               "error": p.error} for p in preds])  # fmt: skip
    write_json(d / "metrics.val.json", {"primary": value, "n": len(preds), "changed": len(preds) - same,
                                        "model": model, "usd": round(usd, 6)})  # fmt: skip
    ctx.spend.record(NAME, PROVIDER, usd, f"logged model retest on {len(preds)} val inputs", model=model)
    write_json(done, {"key": key, "finished_at": now_iso()})
    return f"logged model self-agreement {value} on {len(preds)} val inputs", [d / "metrics.val.json"]


def run(ctx) -> StageResult:
    cfg = ctx.cfg
    pre: list[str] = []
    pre_out: list = []
    if cfg.eval.pass_bar.compare_to == LOGGED_SELF_AGREEMENT:
        line, out = retest_logged(ctx)
        pre.append(line)
        pre_out += out
    if not cfg.baselines:
        return StageResult("; ".join(pre) or "no baselines configured", pre_out)
    task = ctx.task_type()
    splits = {s: ctx.items(read_jsonl(ctx.data_dir / f"{s}.jsonl")) for s in ("val", "test")}
    outputs = list(pre_out)
    summary = list(pre)
    for b in cfg.baselines:
        d = ctx.baseline_dir(b.name)
        context = ctx.resolve(b.context_file).read_text() if b.context_file else None
        examples = examples_for(ctx, b, d)
        prompt = task.baseline_prompt(context, examples)
        key = baseline_key(ctx, b, prompt)
        done = d / "done.json"
        if done.exists() and read_json(done).get("key") == key:
            ctx.log(f"{b.name}: cached, not re-run")
            outputs += [d / f"metrics.{s}.json" for s in splits]
            summary.append(f"{b.name}: val {_primary(d, 'val')}, test {_primary(d, 'test')} (cached)")
            continue

        settings = BaselineSettings(
            model=b.model,
            reasoning_effort=b.reasoning_effort,
            concurrency=cfg.eval.concurrency,
            cache_key=f"ft-{ctx.run_id}-{b.name}"[:64],
        )
        all_items = splits["val"] + splits["test"]
        low, high = estimate_usd(ctx, b, prompt.system, all_items, prompt.max_output_tokens)
        price = ctx.prices.token_price(b.model)

        def paid(preds: list[Prediction], price=price) -> float:
            return sum(price.cost_of(p.usage) for p in preds)

        if high - low > 0.005:  # caching expected: guard on that, plus the worst case for the first chunk
            first = min(len(all_items), CHUNK) / len(all_items)
            guard = low + (high - low) * first
            detail = (
                f"{b.name} on val + test: {len(all_items)} calls (${low:.2f} cached, ${high:.2f} if caching fails; "
                f"checked against the billed cost every {CHUNK} calls)"
            )
        else:
            guard, detail = high, f"{b.name} on val + test: {len(all_items)} calls (${high:.2f})"
        ctx.spend.check(NAME, PROVIDER, guard, detail)
        # spend before this baseline, and calls/cost of its finished splits
        tally = {"before": ctx.state.spent(), "total": len(all_items), "calls": 0, "usd": 0.0}

        def project(preds: list[Prediction], b=b, tally=tally) -> None:
            calls, so_far = tally["calls"] + len(preds), tally["usd"] + paid(preds)
            projected = so_far / max(calls, 1) * tally["total"]
            if tally["before"] + projected > ctx.spend.cap_usd:
                raise SpendStop(
                    f"{NAME}: {b.name} is on course for ${projected:.2f} (${so_far:.2f} for the first {calls} "
                    f"calls), which would pass this run's cap of ${ctx.spend.cap_usd:.2f}; stopped. Prompt caching "
                    "may not be working.",
                    preds,
                )

        usage: dict[str, Any] = {}
        for split, items in splits.items():
            try:
                preds, run_meta = predict_split(
                    ctx,
                    lambda part, prompt=prompt, settings=settings: ctx.providers.frontier.predict(
                        part, prompt, settings
                    ),
                    items,
                    f"{b.name} {split}",
                    cfg.eval.concurrency,
                    after_chunk=project,
                )
            except (TooManyFailures, SpendStop) as e:  # record what the calls that did go through cost, then stop
                ctx.spend.record(
                    NAME,
                    PROVIDER,
                    paid(e.predictions),
                    f"{b.name} {split}: partial, stopped ({e})"[:300],
                    model=b.model,
                )
                raise
            usd = paid(preds)
            pred_path = d / f"{split}.predictions.jsonl"
            write_jsonl(
                pred_path,
                [prediction_row(task, p, structured=True, usd=round(price.cost_of(p.usage), 8)) for p in preds],
            )
            write_json(d / f"{split}.run.json", run_meta)
            ctx.spend.record(NAME, PROVIDER, usd, f"{b.name} {split}: {len(preds)} calls", model=b.model)
            tally["calls"] += len(preds)
            tally["usd"] += usd
            usage[split] = {
                "requests": len(preds),
                "input_tokens": sum(p.usage.input_tokens for p in preds),
                "cache_read_tokens": sum(p.usage.cache_read_tokens for p in preds),
                "cache_write_tokens": sum(p.usage.cache_write_tokens for p in preds),
                "output_tokens": sum(p.usage.output_tokens for p in preds),
                "usd": round(usd, 6),
            }
            m = compute_metrics(ctx, task, split, pred_path, run_meta)
            per_1k = frontier_cost_per_1k(usd, len(preds))
            m["cost"] = {
                "usd": round(usd, 6),
                "cost_per_1k": None if per_1k is None else round(per_1k, 4),
                "cost_per_request": round(usd / len(preds), 8) if preds else None,
            }
            write_json(d / f"metrics.{split}.json", m)
            outputs += [pred_path, d / f"metrics.{split}.json"]
        write_json(d / "usage.json", usage)
        write_json(done, {"key": key, "finished_at": now_iso(), "settings": b.model_dump(exclude_none=True)})
        summary.append(f"{b.name}: val {_primary(d, 'val')}, test {_primary(d, 'test')}")
    return StageResult("; ".join(summary), outputs)


def _primary(d, split: str) -> float:
    return read_json(d / f"metrics.{split}.json")["primary"]
