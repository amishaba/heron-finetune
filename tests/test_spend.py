import json

import pytest

from ft.events import EventLog
from ft.spend import (
    Prices,
    SpendError,
    SpendGuard,
    TokenPrice,
    break_even_requests_per_day,
    estimate_endpoint_usd,
    frontier_cost_per_1k,
    total_real_spend,
    tuned_cost_per_1k,
)
from ft.state import State

PRICES = {
    "openai": {"verified": True, "m": {"input": 0.20, "cached_input": 0.02, "output": 1.20}, "batch_discount": 0.5},
    "together": {
        "verified": False,
        "lora_training_per_mtok": {"Qwen/Qwen3.5-4B": 0.34},
        "training_min_charge": 4.0,
        "endpoint_per_hour": {},
    },
}


def test_token_cost_with_cache():
    p = TokenPrice(input=0.20, output=1.20, cached_input=0.02)
    # 1M input of which 400k cached, 100k output
    assert p.cost(1_000_000, 400_000, 100_000) == pytest.approx(0.6 * 0.20 + 0.4 * 0.02 + 0.1 * 1.20)
    assert p.cost(1_000_000, 0, 0, batch_discount=0.5) == pytest.approx(0.10)
    w = TokenPrice(input=0.20, output=1.20, cached_input=0.02, cache_write=0.25)
    # 1M input: 600k cache writes at 1.25x, 400k plain
    assert w.cost(1_000_000, 0, 0, cache_write_tokens=600_000) == pytest.approx(0.4 * 0.20 + 0.6 * 0.25)


def test_cost_per_1k_and_break_even():
    assert frontier_cost_per_1k(0.30, 2000) == pytest.approx(0.15)
    assert tuned_cost_per_1k(per_hour=3.0, throughput_rph=60_000) == pytest.approx(0.05)
    # 24/7 endpoint at $3/h = $72/day; frontier $0.00015/request → 480k requests/day
    assert break_even_requests_per_day(3.0, 0.00015) == pytest.approx(480_000)
    assert frontier_cost_per_1k(1.0, 0) is None and tuned_cost_per_1k(3.0, 0) is None


def test_training_estimate_has_a_minimum():
    from ft.spend import estimate_training_usd

    assert estimate_training_usd(1_000_000, 3, 0.34, min_charge=4.0) == pytest.approx(4.0)
    assert estimate_training_usd(10_000_000, 3, 0.34, min_charge=4.0) == pytest.approx(10.2)


def test_endpoint_estimate_includes_provisioning():
    assert estimate_endpoint_usd(eval_minutes=5, per_hour=6.0) == pytest.approx((10 + 5) * 6.0 / 60)


def test_unverified_or_missing_prices_block_real_runs():
    real = Prices(PRICES, dry_run=False)
    real.require("openai")
    with pytest.raises(SpendError, match="not verified"):
        real.require("together")
    with pytest.raises(SpendError, match="endpoint hourly price"):
        real.endpoint_per_hour("cr_unknown")
    with pytest.raises(SpendError, match="no price for model"):
        real.token_price("unknown-model")
    Prices(PRICES, dry_run=True).require("together")  # dry runs are allowed


def _guard(tmp_path, *, cap=1.0, dry_run=False, yes=True, confirm=lambda m: True):
    runs = tmp_path / "runs"
    run = runs / "r1"
    run.mkdir(parents=True, exist_ok=True)
    st = State.create(run / "state.json", "r1", dry_run=dry_run, task_hash="x")
    g = SpendGuard(
        state=st,
        events=EventLog(run / "events.jsonl"),
        prices=Prices(PRICES, dry_run=dry_run),
        cap_usd=cap,
        runs_dir=runs,
        dry_run=dry_run,
        yes=yes,
        confirm=confirm,
    )
    return g, st, runs


def test_guard_stops_over_run_cap(tmp_path):
    g, st, _ = _guard(tmp_path, cap=1.0)
    g.check("baseline", "openai", 0.6, "x")
    g.record("baseline", "openai", 0.6, "x")
    with pytest.raises(SpendError, match="cap"):
        g.check("baseline", "openai", 0.5, "y")
    assert st.spent() == pytest.approx(0.6)


def test_guard_respects_total_budget_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_TOTAL_BUDGET_USD", "10")
    g, _, runs = _guard(tmp_path, cap=50)
    other = runs / "old"
    other.mkdir()
    (other / "state.json").write_text(json.dumps({"spend": [{"usd": 9.5}, {"usd": 99, "simulated": True}]}))
    assert total_real_spend(runs) == pytest.approx(9.5)  # simulated spend never counts
    with pytest.raises(SpendError, match="FT_TOTAL_BUDGET_USD"):
        g.check("train", "openai", 1.0, "x")


def test_guard_asks_unless_yes(tmp_path):
    asked = []
    g, _, _ = _guard(tmp_path, yes=False, confirm=lambda m: asked.append(m) or False)
    with pytest.raises(SpendError, match="cancelled"):
        g.check("baseline", "openai", 0.1, "x")
    assert asked


def test_dry_run_spend_is_marked_simulated(tmp_path):
    g, st, runs = _guard(tmp_path, dry_run=True)
    g.check("serve", "together", 0.5, "x")  # unverified prices fine in dry run
    g.record("serve", "together", 0.5, "x")
    assert st.data["spend"][0]["simulated"] is True
    assert total_real_spend(runs) == 0
