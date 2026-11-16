import random
from collections import Counter

import pytest

from ft.config import ConfigError
from ft.stages.split import (
    LeakageError,
    check_leakage,
    make_splits,
    resolve_duplicates,
    resolve_labels,
    stratified_take,
)


def _rows(n_per_class: dict[str, int], hint: str | None = None, start: int = 0):
    rows, i = [], start
    for lbl, n in n_per_class.items():
        for _ in range(n):
            meta = {"split_hint": hint} if hint else {}
            rows.append({"id": f"r{i}", "input": f"message number {i}", "target": lbl, "meta": meta})
            i += 1
    return rows


def test_stratified_take_keeps_proportions():
    rows = _rows({"a": 600, "b": 300, "c": 100})
    taken, rest = stratified_take(rows, 100, lambda r: r["target"], random.Random(0))
    assert Counter(r["target"] for r in taken) == {"a": 60, "b": 30, "c": 10}
    assert len(rest) == 900
    assert not {r["id"] for r in taken} & {r["id"] for r in rest}


def test_stratified_take_exact_count_with_remainders():
    rows = _rows({"a": 7, "b": 5, "c": 3})
    taken, _ = stratified_take(rows, 4, lambda r: r["target"], random.Random(1))
    assert len(taken) == 4


def test_predefined_test_and_carved_val(make_cfg):
    cfg = make_cfg(splits__val_fraction=0.1, splits__val_max=1000, splits__test_max=1000)
    recs = _rows({"a": 500, "b": 500}, hint="train") + _rows({"a": 50, "b": 50}, hint="test", start=5000)
    s = make_splits(recs, cfg)
    assert len(s["test"]) == 100 and all(r["meta"]["split_hint"] == "test" for r in s["test"])
    assert len(s["val"]) == 100 and Counter(r["target"] for r in s["val"]) == {"a": 50, "b": 50}
    assert len(s["train"]) == 900


def test_test_cap_is_stratified(make_cfg):
    cfg = make_cfg(splits__test_max=40)
    recs = _rows({"a": 100}, hint="train") + _rows({"a": 60, "b": 20}, hint="test", start=1000)
    s = make_splits(recs, cfg)
    assert Counter(r["target"] for r in s["test"]) == {"a": 30, "b": 10}


def test_carves_test_when_none_predefined(make_cfg):
    cfg = make_cfg(splits__use_predefined=False, splits__test_max=50, splits__val_max=50)
    s = make_splits(_rows({"a": 300, "b": 300}), cfg)
    assert len(s["test"]) == 50 and len(s["val"]) == 50 and len(s["train"]) == 500


def test_by_time_puts_newest_in_test(make_cfg):
    cfg = make_cfg(splits__by_time="date", splits__test_max=20, splits__val_max=20, splits__val_fraction=0.1)
    recs = _rows({"a": 100, "b": 100})
    random.Random(0).shuffle(recs)
    for i, r in enumerate(sorted(recs, key=lambda r: r["id"])):
        r["meta"]["date"] = f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}"
    s = make_splits(recs, cfg)
    newest_train = max(r["meta"]["date"] for r in s["train"])
    assert newest_train < min(r["meta"]["date"] for r in s["val"])
    assert max(r["meta"]["date"] for r in s["val"]) < min(r["meta"]["date"] for r in s["test"])


def test_by_time_requires_the_field(make_cfg):
    cfg = make_cfg(splits__by_time="date")
    with pytest.raises(ConfigError, match="by_time"):
        make_splits(_rows({"a": 10}), cfg)


def test_leakage_check_normalizes_case_and_space():
    a = {"id": "1", "input": "Where is  my CARD", "target": "x", "meta": {}}
    b = {"id": "2", "input": "where is my card", "target": "x", "meta": {}}
    with pytest.raises(LeakageError):
        check_leakage({"train": [a], "test": [b]})
    check_leakage({"train": [a], "test": [{**b, "input": "where is my refund"}]})


def _r(i, text, hint=None, target="a"):
    return {"id": str(i), "input": text, "target": target, "meta": {"split_hint": hint} if hint else {}}


def test_duplicate_across_train_and_test_keeps_the_test_copy():
    rows = [_r(1, "How do I unblock my PIN?", "train"), _r(2, "how do i  unblock my pin?", "test")]
    kept, dropped = resolve_duplicates(rows, use_hints=True)
    assert [r["id"] for r in kept] == ["2"]
    assert dropped[0]["id"] == "1" and dropped[0]["split"] == "train" and dropped[0]["kept_split"] == "test"


def test_duplicates_inside_test_are_all_kept():
    rows = [_r(1, "Which ATMs accept this card?", "test"), _r(2, "\nWhich ATMs accept this card?", "test")]
    kept, dropped = resolve_duplicates(rows, use_hints=True)
    assert len(kept) == 2 and dropped == []


def test_duplicates_inside_train_keep_first():
    rows = [_r(1, "Hi There", "train"), _r(2, "hi  there", "train", target="b")]
    kept, dropped = resolve_duplicates(rows, use_hints=True)
    assert [r["id"] for r in kept] == ["1"]
    assert dropped[0]["label_conflict"] is True and dropped[0]["exact"] is False


def test_val_beats_train_and_test_beats_val():
    rows = [_r(1, "same", "train"), _r(2, "SAME", "val"), _r(3, "same ", "test")]
    kept, dropped = resolve_duplicates(rows, use_hints=True)
    assert [r["id"] for r in kept] == ["3"] and len(dropped) == 2


def test_without_hints_one_copy_survives():
    rows = [_r(1, "same", "test"), _r(2, "same", "train")]
    kept, _ = resolve_duplicates(rows, use_hints=False)
    assert [r["id"] for r in kept] == ["1"]


def test_auto_labels_come_from_train(make_cfg):
    cfg = make_cfg()
    splits = {"train": _rows({"b": 2, "a": 2}), "val": _rows({"a": 1}), "test": _rows({"c": 1})}
    assert resolve_labels(cfg, splits) == ["a", "b"]


def test_explicit_labels_reject_unknown_targets(make_cfg):
    cfg = make_cfg(task__output__labels=["a", "b"])
    with pytest.raises(ConfigError, match="not in task.output.labels"):
        resolve_labels(cfg, {"train": _rows({"a": 1, "z": 1}), "val": [], "test": []})


def test_load_and_the_agent_agree_on_which_file_names_mark_a_split():
    """`testing.csv` must be held out by the load stage too, not only hidden from the agent's sampling."""
    from pathlib import Path

    from ft.agent import inspect, tools
    from ft.stages import load

    assert load.SPLIT_NAMES["testing"] == "test" and load.SPLIT_NAMES["valid"] == "val"
    assert tools.SPLIT_NAMES is load.SPLIT_NAMES and inspect.SPLIT_NAMES is load.SPLIT_NAMES
    assert tools.AgentTools._test_file(None, str(Path("data") / "Testing.csv"))
