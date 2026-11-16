"""Price table, cost estimates, the spend guard, and cost-per-1k / break-even math (spec 11).

Dry runs price everything too (marked `simulated`), so the report's cost sections are exercised,
but simulated spend never counts toward real budgets.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ft.events import EventLog
from ft.providers.base import Usage
from ft.state import State

# Used only in --dry-run, where prices.yaml has gaps we haven't verified yet.
DRY_RUN_ENDPOINT_PER_HOUR = 2.50
ENDPOINT_PROVISION_MINUTES = 10.0
# Rough output tokens a frontier call spends on reasoning, per effort setting. For estimates only (actuals
# come from usage).
REASONING_TOKENS_ESTIMATE = {None: 500, "none": 0, "low": 150, "medium": 500, "high": 1500, "xhigh": 3000, "max": 6000}
# OpenAI bills images per 32x32 patch times a model multiplier (1.2 for the GPT-5.6 family; assumed for
# gpt-6-luna until measured). Estimate only; actual usage is recorded per call.
OPENAI_TOKENS_PER_PATCH = 1.2


class SpendError(Exception):
    """A paid stage can't run: over budget, missing or unverified price, or the user said no."""


class SpendStop(SpendError):
    """Stopped partway through a paid stage; carries the calls already made so they can be recorded."""

    def __init__(self, message: str, predictions: list):
        super().__init__(message)
        self.predictions = predictions


@dataclass
class TokenPrice:
    input: float
    output: float
    cached_input: float
    cache_write: float | None = None  # OpenAI GPT-5.6+: 1.25x input for tokens written to the cache

    def cost(
        self,
        input_tokens: int,
        cached_tokens: int,
        output_tokens: int,
        cache_write_tokens: int = 0,
        batch_discount: float = 1.0,
    ) -> float:
        """input_tokens includes cached and cache-write tokens; reasoning tokens are billed as output."""
        uncached = max(input_tokens - cached_tokens - cache_write_tokens, 0)
        write_rate = self.cache_write if self.cache_write is not None else self.input
        usd = (
            uncached * self.input
            + cached_tokens * self.cached_input
            + cache_write_tokens * write_rate
            + output_tokens * self.output
        ) / 1e6
        return usd * batch_discount

    def cost_of(self, usage: Usage) -> float:
        """What one call's reported usage cost."""
        return self.cost(usage.input_tokens, usage.cache_read_tokens, usage.output_tokens, usage.cache_write_tokens)


TRAIN_PROVIDERS = ("together", "fireworks")


class Prices:
    def __init__(self, data: dict[str, Any], *, dry_run: bool):
        self.data = data
        self.dry_run = dry_run

    @classmethod
    def load(cls, path: Path, *, dry_run: bool) -> Prices:
        return cls(yaml.safe_load(path.read_text()), dry_run=dry_run)

    def verified(self, provider: str) -> bool:
        return bool(self.data.get(provider, {}).get("verified"))

    def require(self, provider: str) -> None:
        if self.dry_run:
            return
        if provider not in self.data:
            raise SpendError(f"prices.yaml has no '{provider}' section; refusing to run a paid stage")
        if not self.verified(provider):
            raise SpendError(
                f"prices.yaml: '{provider}' prices are not verified. Check the provider's pricing page, "
                f"update the numbers and set `verified: true` before a real run."
            )

    def token_price(self, model: str) -> TokenPrice:
        entry = self.data.get("openai", {}).get(model)
        if not entry:
            raise SpendError(f"prices.yaml has no price for model '{model}'")
        return TokenPrice(
            input=entry["input"],
            output=entry["output"],
            cached_input=entry.get("cached_input", entry["input"]),
            cache_write=entry.get("cache_write"),
        )

    def batch_discount(self) -> float:
        return float(self.data.get("openai", {}).get("batch_discount", 1.0))

    def _trainer_section(self, table: str, key: str | None) -> dict[str, Any]:
        """The training provider's section whose `table` lists `key` (model ids and hardware names don't
        repeat across providers); Together's when none does."""
        for provider in TRAIN_PROVIDERS:
            section = self.data.get(provider) or {}
            if key is not None and key in (section.get(table) or {}):
                return section
        return self.data.get("together") or {}

    def training_per_mtok(self, model: str) -> float:
        table = self._trainer_section("lora_training_per_mtok", model).get("lora_training_per_mtok", {})
        if model in table:
            return float(table[model])
        raise SpendError(f"prices.yaml has no LoRA training price for '{model}'")

    def training_min_charge(self, model: str | None = None) -> float:
        return float(self._trainer_section("lora_training_per_mtok", model).get("training_min_charge", 0.0))

    def endpoint_per_hour(self, config: str | None) -> float:
        table = self._trainer_section("endpoint_per_hour", config).get("endpoint_per_hour") or {}
        if config and config in table:
            return float(table[config])
        if self.dry_run:
            return DRY_RUN_ENDPOINT_PER_HOUR
        raise SpendError(
            f"prices.yaml has no endpoint hourly price for config '{config}'. "
            "Look up the provider's rate for that hardware and add it after confirming."
        )

    def image_tokens(self) -> int:
        return int(self.data.get("together", {}).get("vision_tokens_per_image_estimate", 1000))


def total_real_spend(runs_dir: Path) -> float:
    """Real (non-simulated) spend across every run folder. A folder holding only an agent plan (plan.json,
    no state.json yet) counts its agent spend; once the run starts, that spend lives in state.json."""
    total = 0.0
    for d in runs_dir.iterdir() if runs_dir.exists() else []:
        p = d / "state.json" if (d / "state.json").exists() else d / "plan.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        total += sum(e["usd"] for e in data.get("spend", []) if not e.get("simulated"))
    return total


def total_budget_usd() -> float:
    return float(os.environ.get("FT_TOTAL_BUDGET_USD", "100"))


def budget_left_usd(runs_dir: Path) -> float:
    """What's left of the all-runs budget."""
    return total_budget_usd() - total_real_spend(runs_dir)


def spend_entry(stage: str, provider: str, usd: float, detail: str, *, simulated: bool, **extra: Any) -> dict[str, Any]:
    """One row of a spend ledger (state.json's `spend`, or plan.json's before a run exists). Everything that
    records spend builds its row here, so `total_real_spend` and the report read one shape."""
    return {"stage": stage, "provider": provider, "usd": round(usd, 6), "detail": detail, "simulated": simulated,
            **extra}  # fmt: skip


class SpendGuard:
    def __init__(
        self,
        *,
        state: State,
        events: EventLog,
        prices: Prices,
        cap_usd: float,
        runs_dir: Path,
        dry_run: bool,
        yes: bool,
        confirm: Callable[[str], bool],
    ):
        self.state = state
        self.events = events
        self.prices = prices
        self.cap_usd = cap_usd
        self.runs_dir = runs_dir
        self.dry_run = dry_run
        self.yes = yes
        self.confirm = confirm

    def remaining(self) -> float:
        """What's left under this run's cap."""
        return self.cap_usd - self.state.spent()

    def check(self, stage: str, provider: str, estimate_usd: float, detail: str, attempt: int | None = None) -> None:
        """Call before every paid stage. Raises SpendError to stop the run."""
        self.prices.require(provider)
        spent = self.state.spent()
        self.events.emit(
            "spend_estimate",
            stage=stage,
            attempt=attempt,
            provider=provider,
            estimate_usd=round(estimate_usd, 4),
            spent_usd=round(spent, 4),
            cap_usd=self.cap_usd,
            simulated=self.dry_run,
            detail=detail,
        )
        if spent + estimate_usd > self.cap_usd:
            raise SpendError(
                f"{stage}: estimated ${estimate_usd:.2f} + ${spent:.2f} already spent exceeds this run's "
                f"cap of ${self.cap_usd:.2f}. Raise budget.max_spend_usd or --max-spend to continue."
            )
        if self.dry_run:
            return
        overall = total_real_spend(self.runs_dir)
        limit = total_budget_usd()
        if overall + estimate_usd > limit:
            raise SpendError(
                f"{stage}: estimated ${estimate_usd:.2f} would take total spend across all runs to "
                f"${overall + estimate_usd:.2f}, over FT_TOTAL_BUDGET_USD=${limit:.2f}."
            )
        if not self.yes and not self.confirm(
            f"{stage}: {detail}\n  estimated ${estimate_usd:.2f} (this run so far ${spent:.2f} of "
            f"${self.cap_usd:.2f}; all runs ${overall:.2f} of ${limit:.2f}). Continue?"
        ):
            raise SpendError(f"{stage}: cancelled by user")

    def record(
        self, stage: str, provider: str, usd: float, detail: str, attempt: int | None = None, **extra: Any
    ) -> None:
        entry = spend_entry(stage, provider, usd, detail, simulated=self.dry_run, **extra)
        if attempt is not None:
            entry["attempt"] = attempt
        self.state.add_spend(entry)
        self.events.emit(
            "paid_call",
            stage=stage,
            attempt=attempt,
            provider=provider,
            usd=round(usd, 6),
            simulated=self.dry_run,
            detail=detail,
        )


# ---------------------------------------------------------------- estimates


def estimate_training_usd(training_tokens: int, epochs: int, per_mtok: float, min_charge: float = 0.0) -> float:
    """Billed tokens = dataset tokens x epochs (plus eval tokens, which we don't use); jobs have a minimum."""
    return max(training_tokens * epochs * per_mtok / 1e6, min_charge)


def estimate_endpoint_usd(eval_minutes: float, per_hour: float) -> float:
    return (ENDPOINT_PROVISION_MINUTES + eval_minutes) * per_hour / 60


# ---------------------------------------------------------------- cost per 1k, break-even (spec 11.3)


def frontier_cost_per_1k(total_usd: float, n_requests: int) -> float | None:
    return None if n_requests == 0 else total_usd / n_requests * 1000


def tuned_cost_per_1k(per_hour: float, throughput_rph: float) -> float | None:
    return None if throughput_rph <= 0 else per_hour / throughput_rph * 1000


def break_even_requests_per_day(per_hour: float, frontier_cost_per_request: float) -> float | None:
    """Requests/day at which a 24/7 endpoint costs the same as calling the frontier model."""
    return None if frontier_cost_per_request <= 0 else per_hour * 24 / frontier_cost_per_request
