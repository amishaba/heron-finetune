"""Extraction and images (M2): normalization, parsing, scoring, provider payloads, image sources and a
full dry run of the synthetic receipts fixture (same schema and target_map as CORD)."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from rich.console import Console

from ft.config import ConfigError, parse_task
from ft.context import REPO_ROOT, RunContext
from ft.pipeline import run_pipeline
from ft.providers.base import Item
from ft.providers.openai_provider import user_input
from ft.providers.together_provider import user_content
from ft.tasks import get_task_type
from ft.tasks.base import ScoreRow
from ft.tasks.extract import normalize

quiet = Console(quiet=True)
RECEIPTS = REPO_ROOT / "examples" / "fixtures" / "tiny_receipts" / "task.yaml"
CORD = REPO_ROOT / "examples" / "cord" / "task.yaml"


@pytest.fixture
def task():
    return get_task_type(parse_task(yaml.safe_load(CORD.read_text())), None)


def test_money_number_text_and_date_normalization():
    for a, b in [("60.000", "60,000"), ("Rp 60.000", "60000"), ("40,000.", "40.000"), ("Rp 0.000", "0")]:
        assert normalize(a, "money") == normalize(b, "money")
    assert normalize("", "money") == "" != normalize("0", "money")
    assert normalize("  Nasi   GORENG ", "text") == "nasi goreng"
    assert normalize("1 x", "text") != normalize("1", "text")  # counts are text: formats must match
    assert normalize("2024-03-05", "date") == normalize("05/03/2024", "date") == "2024-03-05"
    assert normalize("5 Mar 2024", "date") == "2024-03-05"


def test_conform_fills_every_field_and_wraps_single_items(task):
    t = task.conform({"total_price": 75000, "menu": {"name": "KOPI", "price": "15.000"}})
    assert t == {
        "total_price": "75000",
        "subtotal_price": "",
        "tax_price": "",
        "menu": [{"name": "KOPI", "count": "", "price": "15.000"}],
    }
    assert list(json.loads(task.train_target(t))) == ["total_price", "subtotal_price", "tax_price", "menu"]


def test_parse_accepts_fences_and_surrounding_text_but_needs_every_key(task):
    good = '{"total_price":"1","subtotal_price":"","tax_price":"","menu":[]}'
    assert task.parse(good, structured=True).valid
    assert task.parse(f"```json\n{good}\n```", structured=False).valid
    assert task.parse(f"Here you go: {good} done", structured=False).valid
    assert not task.parse('{"total_price":"1"}', structured=False).valid  # missing keys
    assert not task.parse("not json", structured=False).valid and not task.parse(None, structured=False).valid


def test_json_schema_is_strict_and_complete(task):
    s = task.json_schema()
    assert s["additionalProperties"] is False and s["required"] == list(task.schema)
    item = s["properties"]["menu"]["items"]
    assert item["additionalProperties"] is False and item["required"] == ["name", "count", "price"]


def test_field_accuracy_by_hand(task):
    true = {
        "total_price": "60.000",
        "subtotal_price": "",
        "tax_price": "",
        "menu": [{"name": "A", "count": "1", "price": "10.000"}, {"name": "B", "count": "2", "price": "50.000"}],
    }
    pred = {
        "total_price": "60,000",
        "subtotal_price": "",
        "tax_price": "6.000",
        "menu": [{"name": "a", "count": "1", "price": "10000"}, {"name": "C", "count": "2", "price": "50.000"}],
    }
    rows = [
        ScoreRow("1", "x.jpg", true, task.conform(pred), True),  # total ok, sub ok, tax wrong, menu F1 0.5
        ScoreRow("2", "y.jpg", true, None, False),  # invalid: 0 everywhere
    ]
    sc = task.scorer(rows)
    assert sc.metric_fn("field_accuracy")(np.array([0])) == pytest.approx((1 + 1 + 0 + 0.5) / 4 * 100)
    assert sc.metric_fn("field_accuracy")(np.arange(2)) == pytest.approx((1 + 1 + 0 + 0.5) / 8 * 100)
    d = sc.details()
    assert d["json_valid_rate"] == 50.0
    assert {f["field"]: f["score"] for f in d["per_field"]}["menu"] == 25.0
    tax = next(e for e in d["field_errors"] if e["field"] == "tax_price")
    assert tax["examples"][0]["predicted"] == "6.000"


def test_provider_payloads_carry_the_image(tmp_path):
    img = tmp_path / "r.jpg"
    Image.new("RGB", (40, 60), "white").save(img)
    it = Item(id="1", input="", image=str(img))
    oa = user_input(it)
    assert oa[0]["content"][0]["type"] == "input_image"
    assert oa[0]["content"][0]["image_url"].startswith("data:image/jpeg;base64,")
    tg = user_content(it)
    assert tg[0]["type"] == "image_url" and tg[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user_input(Item(id="2", input="hello")) == "hello" == user_content(Item(id="2", input="hello"))


def test_image_inputs_need_an_image_source():
    raw = yaml.safe_load(CORD.read_text())
    with pytest.raises(ConfigError, match="image inputs"):
        parse_task({**raw, "source": {"type": "csv", "path": "x.csv"}})


def test_receipts_dry_run_end_to_end():
    ctx = RunContext.create(RECEIPTS, dry_run=True, yes=True, run_id="rcpt")
    run_pipeline(ctx, quiet)
    meta = json.loads((ctx.data_dir / "splits.json").read_text())
    assert meta["sizes"] == {"train": 20, "val": 5, "test": 5} and meta["method"] == "predefined"
    images = list((ctx.data_dir / "images").glob("*.jpg"))
    assert len(images) == 30 and all(max(Image.open(p).size) <= 1280 for p in images)

    row = json.loads((ctx.attempt_dir(1) / "train.together.jsonl").read_text().splitlines()[0])
    system, user, assistant = row["messages"]
    # Together rejects files mixing string and multimodal content: every message is a list of parts.
    assert all(isinstance(m["content"], list) for m in row["messages"])
    assert system["content"] == [{"type": "text", "text": "Extract the receipt's totals and line items as JSON."}]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    answer = json.loads(assistant["content"][0]["text"])
    assert set(answer) == {"total_price", "subtotal_price", "tax_price", "menu"}

    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert report["primary_metric"] == "field_accuracy" and report["task"]["input_kind"] == "image"
    assert {f["field"] for f in report["errors"]["per_field"]} == {"total_price", "subtotal_price", "tax_price", "menu"}
    assert "fields" in (ctx.run_dir / "report.html").read_text()


def test_hf_dataset_source_saves_images_and_maps_targets(tmp_path, monkeypatch):
    rows = {
        "train": [
            {
                "image": Image.new("RGB", (3000, 1500), "white"),
                "ground_truth": json.dumps(
                    {
                        "gt_parse": {
                            "total": {"total_price": "10.000"},
                            "menu": {"nm": "KOPI", "cnt": "1", "price": "10.000"},
                        }
                    }
                ),
            }
            for _ in range(1)
        ]
        + [
            {
                "image": Image.new("RGB", (100, 100), (i, i, i)),
                "ground_truth": json.dumps({"gt_parse": {"total": {"total_price": str(i)}}}),
            }
            for i in range(1, 6)
        ],
        "validation": [
            {
                "image": Image.new("RGB", (100, 100), (200, 0, i)),
                "ground_truth": json.dumps({"gt_parse": {"total": {"total_price": str(i)}}}),
            }
            for i in range(3)
        ],
        "test": [
            {
                "image": Image.new("RGB", (100, 100), (0, 200, i)),
                "ground_truth": json.dumps({"gt_parse": {"total": {"total_price": str(i)}}}),
            }
            for i in range(3)
        ]
        + [{"image": Image.new("RGB", (50, 50), "white"), "ground_truth": json.dumps({"gt_parse": {}})}],
    }
    import datasets

    calls = []

    def fake_load(name, config, split, **kw):
        calls.append((split, kw, os.environ.get("HF_HOME")))
        return rows[split]

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.delenv("HF_HOME", raising=False)
    ctx = RunContext.create(CORD, dry_run=True, yes=True, run_id="hf")
    from ft.stages import load

    res = load.run(ctx)
    recs = [json.loads(line) for line in (ctx.data_dir / "records.jsonl").read_text().splitlines()]
    assert len(recs) == 12 and "dropped 1 missing target" in res.summary
    big = next(r for r in recs if r["meta"]["orig_width"] == 3000)
    assert (big["meta"]["width"], big["meta"]["height"]) == (1280, 640)
    assert big["target"]["menu"] == [{"name": "KOPI", "count": "1", "price": "10.000"}]
    assert {r["meta"]["split_hint"] for r in recs} == {"train", "val", "test"}
    assert all(Path(ctx.data_dir / "images" / r["input"]).exists() for r in recs)
    # streamed through a temp HF cache that is gone afterwards; the environment is restored
    assert all(kw == {"streaming": True} and home and "ft-hf-" in home for _, kw, home in calls)
    assert not Path(calls[0][2]).exists() and "HF_HOME" not in os.environ


def test_changing_a_field_type_does_not_reload_data():
    from ft.stages.load import _output_shape

    raw = yaml.safe_load(CORD.read_text())
    a = parse_task(raw).task.output
    raw["task"]["output"]["schema"]["menu"]["list_of"]["count"] = "text"
    b = parse_task(raw).task.output
    assert _output_shape(a) == _output_shape(b)  # scoring-only change: same load hash
    raw["task"]["output"]["schema"]["menu"]["list_of"]["unit"] = "money"
    assert _output_shape(parse_task(raw).task.output) != _output_shape(a)  # new field: reload


def test_count_is_scored_as_a_number(task):
    true = {
        "total_price": "",
        "subtotal_price": "",
        "tax_price": "",
        "menu": [{"name": "A", "count": "1 x", "price": "5"}],
    }
    pred = {
        "total_price": "",
        "subtotal_price": "",
        "tax_price": "",
        "menu": [{"name": "a", "count": "1", "price": "5"}],
    }
    sc = task.scorer([ScoreRow("1", "x.jpg", true, task.conform(pred), True)])
    assert sc.metric_fn("field_accuracy")(np.array([0])) == 100.0
