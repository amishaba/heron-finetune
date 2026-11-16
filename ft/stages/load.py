"""load: read the source into normalized records {id, input, target, meta} (spec 9.1)."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ft.config import (
    ConfigError,
    CsvSource,
    GmailSource,
    HfDatasetSource,
    ImageFolderSource,
    JsonlSource,
    ListOf,
    LogsSource,
    MaildirSource,
    MboxSource,
    TargetMap,
    TargetMapField,
    stable_hash,
)
from ft.io import file_hash, read_jsonl, write_json, write_jsonl
from ft.stages import StageResult
from ft.tasks import get_task_type

NAME = "load"
# File names (and `split` column values) that mark a predefined split. The agent's folder profile, its cost
# estimate and this stage all read the map from here, so they agree on which rows are held out.
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "dev": "val", "test": "test", "testing": "test"}  # fmt: skip


def source_files(src: CsvSource | JsonlSource, root: Path) -> list[tuple[Path, str | None]]:
    """(file, split hint) pairs for a csv/jsonl source whose path resolves to `root`: the files named in
    `source.files`, the one file, or every file of that type in the folder (hinted by name)."""
    ext = ".csv" if src.type == "csv" else ".jsonl"
    if src.files:
        pairs = [(root / name, split) for split, name in src.files.model_dump().items() if name]
    elif root.is_file():
        pairs = [(root, None)]
    else:
        pairs = [(p, SPLIT_NAMES.get(p.stem.lower())) for p in sorted(root.glob(f"*{ext}"))]
    if not pairs:
        raise ConfigError(f"source.path {root} has no {ext} files")
    for p, _ in pairs:
        if not p.exists():
            raise ConfigError(f"source file not found: {p}")
    order = {"train": 0, "val": 1, "test": 2, None: 3}
    return sorted(pairs, key=lambda fp: order[fp[1]])


def input_hash(ctx) -> str:
    src = ctx.cfg.source
    raw = None
    if isinstance(src, CsvSource | JsonlSource):
        raw = file_hash(*[p for p, _ in source_files(src, ctx.resolve(src.path))])
    elif isinstance(src, GmailSource) and ctx.resolve(src.export).exists():
        raw = [file_hash(ctx.resolve(src.export)), ctx.cfg.targets.rules]
    elif isinstance(src, MaildirSource | MboxSource):
        raw = tree_fingerprint(ctx.resolve(src.path))
    elif isinstance(src, ImageFolderSource):
        raw = [file_hash(ctx.resolve(src.labels_file)), tree_fingerprint(ctx.resolve(src.path))]
    elif isinstance(src, LogsSource):
        raw = file_hash(*[ctx.resolve(x) for x in (src.path, src.truth_sidecar) if x])
    return stable_hash(
        [
            src.model_dump(mode="json"),
            ctx.cfg.task.input.model_dump(),
            _output_shape(ctx.cfg.task.output),
            raw,
        ]
    )


def _output_shape(output) -> dict[str, Any]:
    """The parts of task.output that change what load writes. An extract schema's field *types* only
    change scoring, so they're left out (switching a field from text to number mustn't re-download data).
    Classify outputs hash exactly as before."""
    dumped = output.model_dump(by_alias=True)
    if output.schema_:
        dumped["schema"] = {
            k: (sorted(v.list_of) if isinstance(v, ListOf) else "scalar") for k, v in output.schema_.items()
        }
    return dumped


def tree_fingerprint(root: Path) -> list[Any]:
    """Cheap change detector for big mail trees (Enron is ~500k files): count, total size, newest mtime."""
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    stats = [p.stat() for p in files]
    return [len(stats), sum(s.st_size for s in stats), max((s.st_mtime_ns for s in stats), default=0)]


def _unique_id(seen: dict[str, int], base: str) -> str:
    """`base`, or `base-2`, `base-3`, ... when earlier records already took it."""
    n = seen[base] = seen.get(base, 0) + 1
    return base if n == 1 else f"{base}-{n}"


def _dropped_note(dropped: dict[str, int]) -> str:
    note = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in dropped.items() if v)
    return f" (dropped {note})" if note else ""


def _read_rows(path: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "csv":
        with path.open(newline="", encoding="utf-8-sig") as f:  # tolerate a BOM (Excel exports)
            return list(csv.DictReader(f))
    return read_jsonl(path)


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return ""
    return "" if obj is None else obj


def apply_target_map(row: dict[str, Any], tm: TargetMap) -> dict[str, Any]:
    """Generic JSON-column → target builder. Dataset-specific mappings live in examples/*/task.yaml."""
    raw = row.get(tm.column)
    doc = json.loads(raw) if isinstance(raw, str) else (raw or {})
    if tm.root:
        doc = _dig(doc, tm.root) or {}
    out: dict[str, Any] = {}
    for name, spec in tm.fields.items():
        if isinstance(spec, str):
            out[name] = _dig(doc, spec)
            continue
        assert isinstance(spec, TargetMapField)
        val = _dig(doc, spec.path)
        if spec.as_list:
            val = [] if val == "" else (val if isinstance(val, list) else [val])
            if spec.item:
                val = [{k: _dig(v, src) if isinstance(v, dict) else "" for k, src in spec.item.items()} for v in val]
        out[name] = val
    return out


def _run_gmail(ctx) -> StageResult:
    from ft.connectors.gmail import FEATURES, load_export
    from ft.connectors.rules import Rules
    from ft.redact import ENTITIES, Redactor

    src = ctx.cfg.source
    redactor = Redactor()
    rules = Rules(ctx.cfg.targets.rules or [], FEATURES)
    # The export already covers the pull window (months, min_age_days); re-filtering against "now" here
    # would make reruns of the same run see different data.
    ctx.log("rendering and redacting messages (Presidio)")
    records, dropped = load_export(ctx.resolve(src.export), rules, redactor)
    out = ctx.data_dir / "records.jsonl"
    write_jsonl(out, records)
    redaction = {
        "messages": len(records),
        "entities_replaced": dict(redactor.counts.most_common()),
        "entity_types_checked": ENTITIES,
        "sender": "salted hash token + domain (salt in ~/.config/ft/salt, never uploaded)",
        "fields_rendered": [
            "sender token + domain",
            "recipient count",
            "Gmail category",
            "unsubscribe header",
            "subject (redacted)",
            "first 300 words of body (redacted)",
        ],
        "stays_local": "the raw export (data/email/), sender names and addresses, the salt, this report",
    }
    write_json(ctx.data_dir / "redaction.json", redaction)
    write_json(ctx.data_dir / "load_stats.json", {"rows": len(records), "dropped": dropped})
    ctx.events.emit(
        "load_stats", stage=NAME, rows=len(records), dropped=dropped, redacted=dict(redactor.counts.most_common())
    )
    return StageResult(
        f"{len(records)} emails rendered and redacted ({sum(redactor.counts.values())} entities replaced)",
        [out, ctx.data_dir / "redaction.json"],
        {"rows": len(records)},
    )


def _run_mailfiles(ctx) -> StageResult:
    from ft.connectors import mailfiles
    from ft.redact import ENTITIES, Redactor

    src = ctx.cfg.source
    root = ctx.resolve(src.path)
    ctx.log(f"reading {src.type} at {root}")
    try:
        rows, stats = mailfiles.read_messages(src, root)
    except mailfiles.MailFilesError as e:
        raise ConfigError(str(e)) from e
    if len(stats["folders"]) < 2:
        raise ConfigError(
            f"{src.path}: {len(stats['folders'])} folder(s) left after exclusions and min_per_folder="
            f"{src.min_per_folder}; folders seen: {stats['excluded_folders'] or 'none'}"
        )
    redactor = Redactor() if src.redact else mailfiles.NoRedaction()
    ctx.log(f"{mailfiles.summarize(stats)}; rendering" + (" and redacting (Presidio)" if src.redact else ""))
    records = [
        {
            "id": r["id"],
            "input": mailfiles.render(r, redactor, redact_sender=src.redact),
            "target": r["folder"],
            "meta": {"date": r["date"]},
        }
        for r in rows
    ]
    out = ctx.data_dir / "records.jsonl"
    write_jsonl(out, records)
    write_json(
        ctx.data_dir / "redaction.json",
        {
            "messages": len(records),
            "redacted": src.redact,
            "entities_replaced": dict(redactor.counts.most_common()),
            "entity_types_checked": ENTITIES if src.redact else [],
            "sender": "salted hash token + domain (salt in ~/.config/ft/salt, never uploaded)"
            if src.redact
            else "address as-is (redact: false)",
            "fields_rendered": mailfiles.fields_rendered(src.redact),
            "stays_local": f"the mail files ({src.path}), message dates, this report"
            + (", sender addresses, the salt" if src.redact else ""),
        },
    )
    write_json(ctx.data_dir / "load_stats.json", {"rows": len(records), **stats})
    ctx.events.emit("load_stats", stage=NAME, rows=len(records), dropped=stats["dropped"], folders=stats["folders"])
    return StageResult(mailfiles.summarize(stats), [out, ctx.data_dir / "redaction.json"], {"rows": len(records)})


@contextmanager
def _hf_cache(streaming: bool):
    """While streaming, point Hugging Face's caches (metadata, xet chunks) at a temp folder that is deleted
    afterwards, so nothing lands in ~/.cache/huggingface. Must wrap the first `datasets` import: the HF
    libraries read these variables at import time."""
    if not streaming:
        yield
        return
    keys = ("HF_HOME", "HF_XET_CHUNK_CACHE_SIZE_BYTES")
    saved = {k: os.environ.get(k) for k in keys}
    with tempfile.TemporaryDirectory(prefix="ft-hf-") as tmp:
        os.environ.update({"HF_HOME": tmp, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"})
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _shrink_row(row: dict[str, Any], field: str, max_side: int) -> dict[str, Any]:
    """Downscale a streamed image right away, so ~1,000 full-size photos never sit in memory at once."""
    from PIL import Image

    img = row.get(field)
    if isinstance(img, Image.Image) and max(img.size) > max_side:
        scale = max_side / max(img.size)
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        row = {**row, field: small, "_orig_size": img.size}
    return row


def _image_rows(ctx) -> list[tuple[dict[str, Any], str | None]]:
    """(row, split hint) pairs from an image source. Rows hold a PIL image or a file path in the input field."""
    src = ctx.cfg.source
    in_field = ctx.cfg.task.input.field
    if isinstance(src, HfDatasetSource):
        mapping = {k: v for k, v in (src.splits.model_dump() if src.splits else {"train": "train"}).items() if v}
        out = []
        with _hf_cache(src.streaming):
            from datasets import load_dataset

            for ours, theirs in mapping.items():
                how = "streaming" if src.streaming else "downloading"
                ctx.log(f"{how} {src.dataset} ({theirs} split) from Hugging Face")
                kwargs = {"streaming": True} if src.streaming else {}
                for i, row in enumerate(load_dataset(src.dataset, src.name, split=theirs, **kwargs)):
                    out.append((_shrink_row(row, in_field, ctx.cfg.task.input.max_side), ours))
                    if (i + 1) % 100 == 0:
                        ctx.log(f"{theirs}: {i + 1} rows")
        return out
    root = ctx.resolve(src.path)
    labels = ctx.resolve(src.labels_file)
    if not labels.exists():
        raise ConfigError(f"labels_file not found: {labels}")
    rows = _read_rows(labels, "csv" if labels.suffix == ".csv" else "jsonl")
    if rows and in_field not in rows[0]:
        raise ConfigError(f"{labels.name}: no column '{in_field}' with image file names; columns are {sorted(rows[0])}")
    return [({**r, in_field: root / str(r[in_field])}, SPLIT_NAMES.get(str(r.get("split", "")).lower())) for r in rows]


def _run_images(ctx) -> StageResult:
    """Image sources: each image is downscaled once and saved under data/images/<content hash>.jpg."""
    from PIL import Image, UnidentifiedImageError

    from ft.images import save_downscaled

    cfg = ctx.cfg
    src = cfg.source
    in_field, out_field = cfg.task.input.field, cfg.task.output.field
    extract = cfg.task.type == "extract"
    conform = get_task_type(cfg, None).conform if extract else None
    image_dir = ctx.data_dir / "images"
    records: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    dropped = {"unreadable_image": 0, "missing_target": 0}
    per_split: dict[str, int] = {}
    for row, hint in _image_rows(ctx):
        value = row.get(in_field)
        try:
            img = value if isinstance(value, Image.Image) else Image.open(value)
            name, size = save_downscaled(img, image_dir, cfg.task.input.max_side)
            if "_orig_size" in row:  # shrunk while streaming; keep the true original size
                size["orig_width"], size["orig_height"] = row["_orig_size"]
        except (OSError, UnidentifiedImageError, AttributeError, TypeError):
            dropped["unreadable_image"] += 1
            continue
        target = apply_target_map(row, src.target_map) if src.target_map else row.get(out_field)
        if extract:
            target = conform(target)
            if not any(target.values()):
                dropped["missing_target"] += 1
                continue
        elif target is None or not str(target).strip():
            dropped["missing_target"] += 1
            continue
        else:
            target = str(target)
        meta = {"split_hint": hint, **size} if hint else dict(size)
        rid = _unique_id(ids, name.removesuffix(".jpg"))
        records.append({"id": rid, "input": name, "target": target, "meta": meta})
        per_split[hint or "unsplit"] = per_split.get(hint or "unsplit", 0) + 1
    if not records:
        raise ConfigError(f"no usable images loaded (dropped: {dropped})")
    out = ctx.data_dir / "records.jsonl"
    write_jsonl(out, records)
    write_json(ctx.data_dir / "load_stats.json", {"rows": len(records), "splits": per_split, "dropped": dropped})
    ctx.events.emit("load_stats", stage=NAME, rows=len(records), splits=per_split, dropped=dropped)
    sizes = ", ".join(f"{k} {v}" for k, v in per_split.items())
    return StageResult(
        f"{len(records)} images ({sizes}), longest side <= {cfg.task.input.max_side}px{_dropped_note(dropped)}",
        [out],
        {"rows": len(records)},
    )


def logged_prompt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What it takes to re-ask the logged model (for the self-agreement bar): its system prompt and settings,
    when every log shares them; None where they differ."""

    def one(values):
        distinct = set(values)
        return distinct.pop() if len(distinct) == 1 else None

    models = Counter(str(r.get("logged_model")) for r in rows)
    return {
        "system_prompt": one(r.get("system_prompt") or "" for r in rows),
        "model": models.most_common(1)[0][0] if models else None,
        "models": dict(models),
        "reasoning_effort": one((r.get("metadata") or {}).get("reasoning_effort") for r in rows),
    }


def _run_logs(ctx) -> StageResult:
    """Logs mode (spec 15): the logged response is the target; the truth sidecar, when given, rides along
    in meta for evaluation only."""
    from ft.connectors.logs import FIELDS, LogsError, read_logs

    src = ctx.cfg.source
    in_field, out_field = ctx.cfg.task.input.field, ctx.cfg.task.output.field
    if in_field not in FIELDS or out_field not in FIELDS:
        raise ConfigError(f"logs tasks read fields {FIELDS}; got input '{in_field}', output '{out_field}'")
    try:
        rows, stats = read_logs(ctx.resolve(src.path), ctx.resolve(src.truth_sidecar) if src.truth_sidecar else None)
    except LogsError as e:
        raise ConfigError(str(e)) from e
    records, dropped, ids = [], {"blank_input": 0, "blank_response": 0}, {}
    for r in rows:
        text, target = r[in_field], r[out_field]
        if not text.strip():
            dropped["blank_input"] += 1
            continue
        if not str(target).strip():
            dropped["blank_response"] += 1
            continue
        meta = {
            k: r[k] for k in ("logged_model", "usage", "latency_ms", "timestamp", "metadata") if r.get(k) is not None
        }
        if "truth" in r:
            meta["truth"] = r["truth"]
        meta["source_file"] = Path(src.path).name
        records.append({"id": _unique_id(ids, r["id"]), "input": text, "target": str(target), "meta": meta})
    out = ctx.data_dir / "records.jsonl"
    write_jsonl(out, records)
    write_json(ctx.data_dir / "logged_prompt.json", logged_prompt(rows))
    info = {**stats, "rows": len(records), "dropped": dropped}
    ctx.events.emit("load_stats", stage=NAME, **info)
    write_json(ctx.data_dir / "load_stats.json", info)
    truth = f", {stats['with_truth']:,} with truth labels" if stats["with_truth"] else ""
    return StageResult(f"{len(records):,} logged calls from {', '.join(stats['models'])}{truth}", [out],
                       {"rows": len(records)})  # fmt: skip


def _run_table(ctx) -> StageResult:
    """CSV / JSONL files: one record per row with a non-blank input (and, for classify, a label)."""
    src = ctx.cfg.source
    in_field = ctx.cfg.task.input.field
    out_field = ctx.cfg.task.output.field
    records: list[dict[str, Any]] = []
    ids: dict[str, int] = {}
    per_file: dict[str, int] = {}
    dropped = {"blank_input": 0, "missing_target": 0}
    classify = ctx.cfg.task.type == "classify"
    for path, hint in source_files(src, ctx.resolve(src.path)):
        rows = _read_rows(path, src.type)
        if rows and in_field not in rows[0]:
            raise ConfigError(f"{path.name}: no input column '{in_field}'; columns are {sorted(rows[0])}")
        if rows and not src.target_map and out_field not in rows[0]:
            raise ConfigError(f"{path.name}: no output column '{out_field}'; columns are {sorted(rows[0])}")
        per_file[path.name] = len(rows)
        for row in rows:
            raw_input = row.get(in_field)
            text = "" if raw_input is None else str(raw_input)
            if not text.strip():
                dropped["blank_input"] += 1
                continue
            target = apply_target_map(row, src.target_map) if src.target_map else row.get(out_field)
            if classify:
                if isinstance(target, list | dict):
                    raise ConfigError(
                        f"{path.name}: '{out_field}' holds a {type(target).__name__} ({str(target)[:60]}); "
                        "classify expects exactly one label per row (multi-label isn't supported)"
                    )
                if target is None or not str(target).strip():
                    dropped["missing_target"] += 1
                    continue
                target = str(target)  # JSONL labels may be ints/bools; labels are always strings
            meta = {k: v for k, v in row.items() if k not in (in_field, out_field)}
            meta["source_file"] = path.name
            if hint:
                meta["split_hint"] = hint
            rid = _unique_id(ids, hashlib.sha1(text.encode()).hexdigest()[:12])
            records.append({"id": rid, "input": text, "target": target, "meta": meta})

    # Duplicates (exact and near) are resolved in the split stage, which knows which split wins.
    out = ctx.data_dir / "records.jsonl"
    write_jsonl(out, records)
    ctx.events.emit("load_stats", stage=NAME, rows=len(records), files=per_file, dropped=dropped)
    write_json(ctx.data_dir / "load_stats.json", {"rows": len(records), "files": per_file, "dropped": dropped})
    return StageResult(
        f"{len(records)} records from {len(per_file)} file(s){_dropped_note(dropped)}", [out], {"rows": len(records)}
    )


def run(ctx) -> StageResult:
    """One loader per source type; each writes data/records.jsonl and data/load_stats.json."""
    src = ctx.cfg.source
    if isinstance(src, CsvSource | JsonlSource):
        return _run_table(ctx)
    if isinstance(src, GmailSource):
        return _run_gmail(ctx)
    if isinstance(src, MaildirSource | MboxSource):
        return _run_mailfiles(ctx)
    if isinstance(src, LogsSource):
        return _run_logs(ctx)
    if isinstance(src, HfDatasetSource | ImageFolderSource):
        if ctx.cfg.task.input.kind != "image":
            raise NotImplementedError("text datasets from Hugging Face aren't supported yet; export to CSV/JSONL")
        return _run_images(ctx)
    raise NotImplementedError(f"source type '{src.type}' isn't supported")
