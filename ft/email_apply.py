"""`ft email apply --run <run_id>`: label new mail with a trained email model, on demand (spec 14.6).

1. Fetch mail received since the last checkpoint (~/.config/ft/email_state.json; first time: last 24 h).
2. Render and redact it exactly like the training data.
3. Deploy the run's best model, classify, tear down (same guarantees as the pipeline).
4. ADD the label "ft/Probably skip" to predicted skip and "ft/Important" to predicted important.
   Never archive, delete or mark anything read.
5. Save the checkpoint and a log of what was labeled (message ids and predictions only, no content).

No scheduling: this runs only when you call it.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from ft.connectors import gmail
from ft.context import RunContext
from ft.events import now_iso
from ft.providers.base import Item
from ft.redact import Redactor, config_dir
from ft.stages import run_predictions
from ft.stages.serve import deployed

FIRST_RUN_LOOKBACK_MS = 24 * 3600 * 1000


class ApplyError(RuntimeError):
    pass


def _state_path():
    return config_dir() / "email_state.json"


def load_checkpoint(run_id: str) -> int | None:
    p = _state_path()
    return json.loads(p.read_text()).get(run_id, {}).get("last_ms") if p.exists() else None


def save_checkpoint(run_id: str, last_ms: int) -> None:
    p = _state_path()
    data = json.loads(p.read_text()) if p.exists() else {}
    data[run_id] = {"last_ms": last_ms, "updated_at": now_iso()}
    p.write_text(json.dumps(data, indent=2))


def apply(
    ctx: RunContext,
    *,
    service=None,
    since_ms: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cfg = ctx.cfg
    if cfg.source.type != "gmail":
        raise ApplyError(f"run {ctx.run_id} is not an email run (source.type is {cfg.source.type})")
    final = ctx.state.data.get("final_attempt")
    if not final:
        raise ApplyError(f"run {ctx.run_id} has no evaluated model yet")
    if "model_object_id" not in ctx.state.external(final):
        raise ApplyError(f"attempt {final} of {ctx.run_id} has no trained model")

    service = service or gmail.get_service(gmail.SCOPE_MODIFY)
    after_ms = since_ms or load_checkpoint(ctx.run_id) or int(time.time() * 1000) - FIRST_RUN_LOOKBACK_MS
    rows = gmail.fetch_since(service, after_ms, log)
    if not rows:
        log("no new mail since the last checkpoint")
        return {"messages": 0}

    redactor = Redactor()
    items = [Item(id=r["id"], input=gmail.render(r, redactor)) for r in rows]
    task = ctx.task_type(final)
    prompt = task.tuned_prompt()
    conc = cfg.eval.concurrency
    with deployed(ctx, ctx.tuned_model(final), final, task.eval_minutes(len(items), conc)) as ep:
        preds = run_predictions(ctx, lambda part: ctx.providers.tuned.predict(ep, part, prompt, conc), items, "apply")

    predicted = {p.id: task.parse(p.prediction_raw, structured=False) for p in preds}
    by_label: dict[str, list[str]] = {lbl: [] for lbl in gmail.APPLY_LABELS}
    for mid, parsed in predicted.items():
        if parsed.valid and parsed.value in by_label:
            by_label[parsed.value].append(mid)
    label_ids = gmail.ensure_labels(service, gmail.APPLY_LABELS.values())
    for lbl, ids in by_label.items():
        if ids:
            gmail.add_labels(service, label_ids[gmail.APPLY_LABELS[lbl]], ids)

    newest = max(r["internal_ms"] for r in rows)
    save_checkpoint(ctx.run_id, newest)
    counts = Counter(p.value if p.valid else "(invalid)" for p in predicted.values())
    entry = {
        "ts": now_iso(),
        "run_id": ctx.run_id,
        "attempt": final,
        "messages": len(rows),
        "predictions": dict(counts),
        "labeled": {gmail.APPLY_LABELS[k]: v for k, v in by_label.items()},
        "checkpoint_ms": newest,
    }
    with (config_dir() / "email_apply_log.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    ctx.events.emit("email_apply", messages=len(rows), predictions=dict(counts))
    return entry
