"""split: train / val / test, stratified or by time, with a leakage check (spec 9.3)."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from ft.config import ConfigError, stable_hash
from ft.io import read_jsonl, write_json, write_jsonl
from ft.stages import StageResult, upstream

NAME = "split"

Row = dict[str, Any]


class LeakageError(Exception):
    pass


def input_hash(ctx) -> str:
    c = ctx.cfg
    return stable_hash([upstream(ctx, "load"), c.splits.model_dump(), c.task.type, c.task.output.labels, c.source.type])


def stratified_take(
    rows: list[Row], n: int, key: Callable[[Row], Any] | None, rng: random.Random, keep_one: bool = False
):
    """Pick n rows, proportionally per class (largest remainder). Returns (taken, rest).
    keep_one: never take a class's last row (used when carving val/test out of train, so rare classes
    still have at least one training example)."""
    if n >= len(rows):
        return list(rows), []
    if n <= 0:
        return [], list(rows)
    if key is None:
        idx = set(rng.sample(range(len(rows)), n))
    else:
        groups: dict[Any, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            groups[key(r)].append(i)
        quotas = {g: n * len(ix) / len(rows) for g, ix in groups.items()}
        alloc = {g: int(q) for g, q in quotas.items()}
        leftover = n - sum(alloc.values())
        by_remainder = sorted(groups, key=lambda g: (-(quotas[g] - alloc[g]), str(g)))
        for g in by_remainder[:leftover]:
            alloc[g] += 1
        if keep_one:
            alloc = {g: min(a, len(groups[g]) - 1) for g, a in alloc.items()}
        idx = set()
        for g, ix in groups.items():
            idx.update(rng.sample(ix, min(alloc[g], len(ix))))
    taken = [r for i, r in enumerate(rows) if i in idx]
    rest = [r for i, r in enumerate(rows) if i not in idx]
    return taken, rest


def _norm(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def check_leakage(splits: dict[str, list[Row]]) -> None:
    owner: dict[str, str] = {}
    clashes: list[tuple[str, str, str]] = []
    for name, rows in splits.items():
        for r in rows:
            k = _norm(r["input"])
            if k in owner and owner[k] != name:
                clashes.append((owner[k], name, r["input"][:80]))
            owner.setdefault(k, name)
    if clashes:
        sample = "; ".join(f"{a}/{b}: {t!r}" for a, b, t in clashes[:3])
        raise LeakageError(f"{len(clashes)} inputs appear in more than one split (after normalization): {sample}")


# When copies of one input land in different splits, the most valuable split keeps its copy:
# test > val > train. Copies inside one predefined val/test split are all kept (no leakage, and the
# test set keeps its published size). Copies in train or unsplit data keep only the first, because
# val/test may later be carved from those rows.
SPLIT_PRIORITY = {"test": 3, "val": 2, "train": 1, None: 0}


def resolve_duplicates(rows: list[Row], use_hints: bool) -> tuple[list[Row], list[dict[str, Any]]]:
    """Drop duplicate inputs (after case/whitespace normalization). Returns (kept, dropped log)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[_norm(r["input"])].append(i)

    def hint(i: int) -> str | None:
        return rows[i]["meta"].get("split_hint") if use_hints else None

    keep: set[int] = set()
    dropped: list[dict[str, Any]] = []
    for idxs in groups.values():
        top = max(SPLIT_PRIORITY.get(hint(i), 0) for i in idxs)
        winners = [i for i in idxs if SPLIT_PRIORITY.get(hint(i), 0) == top]
        if top < SPLIT_PRIORITY["val"]:
            winners = winners[:1]
        keep.update(winners)
        for i in idxs:
            if i in winners:
                continue
            first = rows[winners[0]]
            dropped.append(
                {
                    "id": rows[i]["id"],
                    "input": rows[i]["input"],
                    "target": rows[i]["target"],
                    "split": hint(i) or "unsplit",
                    "kept_id": first["id"],
                    "kept_split": hint(winners[0]) or "unsplit",
                    "exact": rows[i]["input"] == first["input"],
                    "label_conflict": rows[i]["target"] != first["target"],
                }
            )
    return [r for i, r in enumerate(rows) if i in keep], dropped


def make_splits(records: list[Row], cfg) -> dict[str, list[Row]]:
    s = cfg.splits
    rng = random.Random(s.seed)
    key = (lambda r: r["target"]) if cfg.task.type == "classify" else None

    if s.by_time:
        missing = [r["id"] for r in records if not r["meta"].get(s.by_time)]
        if missing:
            raise ConfigError(f"splits.by_time='{s.by_time}' but {len(missing)} records have no such field")
        ordered = sorted(records, key=lambda r: str(r["meta"][s.by_time]))
        n_test = min(s.test_max, max(1, round(len(ordered) * s.test_fraction)))
        rest, test = ordered[: len(ordered) - n_test], ordered[len(ordered) - n_test :]
        n_val = min(s.val_max, max(1, round(len(rest) * s.val_fraction)))
        train, val = rest[: len(rest) - n_val], rest[len(rest) - n_val :]
        return {"train": train, "val": val, "test": test}

    hinted: dict[str, list[Row]] = defaultdict(list)
    for r in records:
        hinted[r["meta"].get("split_hint") if s.use_predefined else None].append(r)
    train = hinted.pop("train", []) + hinted.pop(None, [])
    val = hinted.pop("val", [])
    test = hinted.pop("test", [])

    if not test:
        n_test = min(s.test_max, max(1, round(len(train) * s.test_fraction)))
        test, train = stratified_take(train, n_test, key, rng, keep_one=True)
    elif len(test) > s.test_max:
        test, _unused = stratified_take(test, s.test_max, key, rng)
    if not val:
        n_val = min(s.val_max, max(1, round(len(train) * s.val_fraction)))
        val, train = stratified_take(train, n_val, key, rng, keep_one=True)
    elif len(val) > s.val_max:
        val, _unused = stratified_take(val, s.val_max, key, rng)
    return {"train": train, "val": val, "test": test}


def merge_label_variants(records: list[Row]) -> dict[str, list[str]]:
    """Labels that differ only in case/whitespace ("Billing", " billing ") are one class. Rewrite every
    variant to the most common spelling (ties: alphabetical). Returns {canonical: [variants merged]}."""
    from ft.tasks.classify import normalize_label

    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    for r in records:
        spellings[normalize_label(r["target"])][r["target"]] += 1
    canonical, merged = {}, {}
    for counts in spellings.values():
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for spelling in counts:
            canonical[spelling] = best
        if len(counts) > 1:
            merged[best] = sorted(s for s in counts if s != best)
    for r in records:
        r["target"] = canonical[r["target"]]
    return merged


def resolve_labels(cfg, splits: dict[str, list[Row]]) -> list[str]:
    declared = cfg.task.output.labels
    if declared == "auto":
        return sorted({str(r["target"]) for r in splits["train"]})
    unknown = {str(r["target"]) for rows in splits.values() for r in rows} - set(declared)
    if unknown:
        raise ConfigError(f"targets not in task.output.labels: {sorted(unknown)[:10]}")
    return list(declared)


def run(ctx) -> StageResult:
    cfg = ctx.cfg
    records = read_jsonl(ctx.data_dir / "records.jsonl")
    variants = merge_label_variants(records) if cfg.task.type == "classify" else {}
    if variants:
        ctx.events.emit("label_variants_merged", stage=NAME, merged=variants)
    use_hints = cfg.splits.use_predefined and not cfg.splits.by_time
    records, dropped = resolve_duplicates(records, use_hints)
    write_jsonl(ctx.data_dir / "duplicates_dropped.jsonl", dropped)
    dup_summary = {
        "total": len(dropped),
        "by_split": dict(Counter(f"{d['split']} (kept in {d['kept_split']})" for d in dropped)),
        "exact": sum(d["exact"] for d in dropped),
        "label_conflicts": sum(d["label_conflict"] for d in dropped),
    }
    if dropped:
        ctx.events.emit("duplicates_dropped", stage=NAME, **dup_summary)
    splits = make_splits(records, cfg)
    check_leakage(splits)

    outputs = []
    for name, rows in splits.items():
        if not rows:
            raise ConfigError(f"split '{name}' is empty")
        p = ctx.data_dir / f"{name}.jsonl"
        write_jsonl(p, rows)
        outputs.append(p)

    sizes = {k: len(v) for k, v in splits.items()}
    dist = {}
    warnings: list[str] = []
    if cfg.task.type == "classify":
        labels = resolve_labels(cfg, splits)
        write_json(ctx.data_dir / "labels.json", labels)
        outputs.append(ctx.data_dir / "labels.json")
        ctx._labels = labels
        ctx.fake_opts.labels = labels
        dist = {k: dict(Counter(str(r["target"]) for r in v).most_common()) for k, v in splits.items()}
        unseen = sorted({str(r["target"]) for k in ("val", "test") for r in splits[k]} - set(labels))
        if unseen:
            warnings.append(f"labels in val/test but not in train (counted as wrong): {unseen}")
        counts = Counter(str(r["target"]) for r in splits["train"])
        top_label, top_n = counts.most_common(1)[0]
        top_share = top_n / len(splits["train"])
        if top_share > 0.40 and cfg.eval.primary_metric == "accuracy":
            warnings.append(
                f"'{top_label}' is {top_share:.0%} of train; accuracy rewards guessing it. Consider macro_f1."
            )
        later = Counter(str(r["target"]) for k in ("val", "test") for r in splits[k])
        warnings += category_warnings(counts, later, by_time=bool(cfg.splits.by_time))
    for w in warnings:
        ctx.events.emit("warning", stage=NAME, message=w)
    hints = {r["meta"].get("split_hint") for r in records} if use_hints else set()
    if cfg.splits.by_time:
        method = "by_time"
    elif {"val", "test"} <= hints:
        method = "predefined"
    else:
        method = "predefined+carved" if hints - {None} else "carved"
    meta = {
        "sizes": sizes,
        "method": method,
        "duplicates_dropped": dup_summary,
        "label_variants_merged": variants,
        "warnings": warnings,
        "label_distribution": dist,
    }
    write_json(ctx.data_dir / "splits.json", meta)
    outputs.append(ctx.data_dir / "splits.json")
    outputs.append(ctx.data_dir / "duplicates_dropped.jsonl")
    ctx.events.emit("split_stats", stage=NAME, sizes=sizes, method=method)
    dup_note = f"; {len(dropped)} duplicates dropped" if dropped else ""
    return StageResult(
        f"train {sizes['train']}, val {sizes['val']}, test {sizes['test']} ({method}{dup_note})", outputs
    )


THIN_ABSOLUTE = 5  # fewer training examples than this: a small model can't learn the class
THIN_RELATIVE = 0.2  # ...or fewer than this share of the median class's count
LATE_SHARE = 0.5  # under a time split: at least this share of a class's examples come after the train cutoff
LATE_MIN_TOTAL = 10


def category_warnings(train: Counter, later: Counter, *, by_time: bool) -> list[str]:
    """Classes the tuned model will struggle with, said before any results (NOTES.md, M4 design note 2):
    very few training examples (absolutely, or relative to the median class), and under a time split,
    classes whose examples mostly arrive after the train cutoff."""
    out = []
    median = float(np.median(list(train.values()))) if train else 0.0
    rare = sorted(lbl for lbl, c in train.items() if c < THIN_ABSOLUTE)
    if rare:
        out.append(f"classes with fewer than {THIN_ABSOLUTE} training examples: {rare}")
    thin = sorted(
        (lbl for lbl, c in train.items() if THIN_ABSOLUTE <= c < THIN_RELATIVE * median), key=lambda x: train[x]
    )
    if thin and median >= 2 * THIN_ABSOLUTE:
        listed = ", ".join(f"{lbl} ({train[lbl]})" for lbl in thin[:15])
        out.append(
            f"classes with under {THIN_RELATIVE:.0%} of the median class's {median:g} training examples: {listed}"
        )
    if by_time:
        late = []
        for lbl in set(train) | set(later):
            total = train[lbl] + later[lbl]
            if total >= LATE_MIN_TOTAL and later[lbl] / total >= LATE_SHARE:
                late.append((lbl, train[lbl], later[lbl]))
        if late:
            late.sort(key=lambda x: x[2] / (x[1] + x[2]), reverse=True)
            listed = ", ".join(f"{lbl} ({tr} train / {lt} later)" for lbl, tr, lt in late[:15])
            out.append(
                f"classes that mostly arrive after the train cutoff (at least {LATE_SHARE:.0%} of their examples "
                f"are in val + test), so the tuned model sees few of them: {listed}"
            )
    return out
