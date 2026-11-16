import copy

import pytest

from ft.config import ConfigError, apply_patch, load_task, parse_task
from tests.conftest import REPO


def test_examples_validate():
    cfg = load_task(REPO / "examples" / "banking77" / "task.yaml")
    assert cfg.task.type == "classify"
    assert cfg.targets.from_ == "dataset"
    assert cfg.eval.pass_bar.is_relative


def test_unknown_key_is_an_error(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["train"]["epochz"] = 3
    with pytest.raises(ConfigError, match="train.epochz"):
        parse_task(raw)


def test_source_is_discriminated_on_type(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["source"] = {"type": "hf_dataset", "path": "x"}  # hf_dataset has no `path`
    with pytest.raises(ConfigError):
        parse_task(raw)


def test_pass_bar_needs_exactly_one_shape(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["eval"]["pass_bar"] = {"compare_to": "best_baseline", "max_gap_points": 2, "min_value": 90}
    with pytest.raises(ConfigError, match="either"):
        parse_task(raw)
    raw["eval"]["pass_bar"] = {}
    with pytest.raises(ConfigError):
        parse_task(raw)


def test_relative_bar_needs_a_baseline(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["baselines"] = []
    with pytest.raises(ConfigError, match="min_value"):
        parse_task(raw)
    raw["eval"]["pass_bar"] = {"min_value": 95}
    assert not parse_task(raw).eval.pass_bar.is_relative


def test_metric_must_fit_task(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["eval"]["primary_metric"] = "field_accuracy"
    with pytest.raises(ConfigError, match="does not fit"):
        parse_task(raw)


def test_frontier_targets_need_a_model(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["targets"] = {"from": "frontier"}
    with pytest.raises(ConfigError, match="frontier_model"):
        parse_task(raw)


def test_classify_cannot_have_schema(tiny_raw):
    raw = copy.deepcopy(tiny_raw)
    raw["task"]["output"]["schema"] = {"total": "money"}
    with pytest.raises(ConfigError, match="only for extract"):
        parse_task(raw)


def test_patch_is_revalidated(make_cfg):
    cfg = make_cfg()
    patched = apply_patch(cfg, {"train": {"epochs": 5}})
    assert patched.train.epochs == 5 and patched.train.base_model == cfg.train.base_model
    with pytest.raises(ConfigError):
        apply_patch(cfg, {"train": {"epochs": 9}})  # > 6 total (spec 13)
