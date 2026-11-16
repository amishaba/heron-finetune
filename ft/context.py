"""RunContext: paths, config, state, event log, providers and spend guard for one run."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ft.config import ConfigError, TaskConfig, apply_patch, dump_task, load_task, stable_hash
from ft.events import EventLog, now_iso
from ft.io import read_json, write_json
from ft.providers.base import Item, ModelRef, Providers
from ft.providers.fake import FakeFrontier, FakeOptions, FakeServer, FakeTrainer, FakeTuned
from ft.registry import BaseModelInfo, load_models
from ft.spend import Prices, SpendGuard
from ft.state import State
from ft.tasks import get_task_type
from ft.tasks.base import TaskType

REPO_ROOT = Path(__file__).resolve().parent.parent


def runs_dir() -> Path:
    return Path(os.environ.get("FT_RUNS_DIR", REPO_ROOT / "runs"))


def make_run_id(name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M')}-{name}"


def repo_models() -> dict[str, BaseModelInfo]:
    return load_models(REPO_ROOT / "models.yaml")


def repo_prices(*, dry_run: bool) -> Prices:
    return Prices.load(REPO_ROOT / "prices.yaml", dry_run=dry_run)


def build_providers(
    dry_run: bool, models: dict[str, BaseModelInfo], fake_opts: FakeOptions, provider: str = "together"
) -> Providers:
    if dry_run:
        return Providers(
            frontier=FakeFrontier(fake_opts),
            trainer=FakeTrainer(fake_opts, set(models)),
            server=FakeServer(fake_opts),
            tuned=FakeTuned(fake_opts),
        )
    from ft.providers.openai_provider import OpenAIFrontier

    missing = [k for k in ("OPENAI_API_KEY", f"{provider.upper()}_API_KEY") if not os.environ.get(k)]
    if missing:
        raise ConfigError(f"missing {', '.join(missing)} (add to .env)")
    if provider == "fireworks":
        from ft.providers.fireworks_provider import FireworksServer, FireworksTrainer, FireworksTuned

        return Providers(
            frontier=OpenAIFrontier(),
            trainer=FireworksTrainer(models),
            server=FireworksServer(),
            tuned=FireworksTuned(),
        )
    from ft.providers.together_provider import TogetherServer, TogetherTrainer, TogetherTuned

    return Providers(
        frontier=OpenAIFrontier(),
        trainer=TogetherTrainer(),
        server=TogetherServer({m.id: m.deploy_config for m in models.values()}),
        tuned=TogetherTuned(),
    )


@dataclass
class RunContext:
    run_dir: Path
    cfg: TaskConfig
    state: State
    events: EventLog
    providers: Providers
    spend: SpendGuard
    prices: Prices
    models: dict[str, BaseModelInfo]
    dry_run: bool
    yes: bool
    fake_opts: FakeOptions
    attempt: int | None = None
    log: Callable[[str], None] = field(default=lambda _msg: None, repr=False)  # progress lines (set by pipeline)
    _labels: list[str] | None = field(default=None, repr=False)

    # ---------------------------------------------------------------- construction

    @classmethod
    def create(
        cls,
        task_path: Path,
        *,
        dry_run: bool,
        yes: bool,
        run_id: str | None = None,
        max_spend: float | None = None,
        confirm: Callable[[str], bool] = lambda _msg: False,
        fake_opts: FakeOptions | None = None,
    ) -> RunContext:
        cfg = load_task(task_path)
        if not dry_run:  # fail before creating a run folder (e.g. missing keys)
            build_providers(dry_run, repo_models(), FakeOptions(), cfg.train.provider)
        run_id = run_id or make_run_id(cfg.name)
        run_dir = runs_dir() / run_id
        if (run_dir / "state.json").exists():
            raise FileExistsError(f"run {run_id} already exists; use `ft resume {run_id}`")
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task_path, run_dir / "task.yaml")
        state = State.create(run_dir / "state.json", run_id, dry_run=dry_run, task_hash=stable_hash(dump_task(cfg)))
        state.data["max_spend_usd"] = max_spend
        state.data["source_task_path"] = str(task_path)
        state.data["fake_opts"] = fake_opts.to_state() if fake_opts else None
        plan = run_dir / "plan.json"
        if plan.exists():  # the agent planned this run: its spend moves into the run's ledger
            state.data["spend"] += read_json(plan).get("spend", [])
        state.save()
        return cls._assemble(run_dir, cfg, state, dry_run=dry_run, yes=yes, confirm=confirm, fake_opts=fake_opts)

    @classmethod
    def open(
        cls,
        run_id: str,
        *,
        yes: bool,
        confirm: Callable[[str], bool] = lambda _msg: False,
        fake_opts: FakeOptions | None = None,
    ) -> RunContext:
        run_dir = runs_dir() / run_id
        if not (run_dir / "state.json").exists():
            raise FileNotFoundError(f"no run named {run_id} under {runs_dir()}")
        state = State.load(run_dir / "state.json")
        cfg = load_task(run_dir / "task.yaml")
        if fake_opts is None and state.data.get("fake_opts"):
            fake_opts = FakeOptions.from_state(state.data["fake_opts"])
        return cls._assemble(
            run_dir, cfg, state, dry_run=state.data["dry_run"], yes=yes, confirm=confirm, fake_opts=fake_opts
        )

    @staticmethod
    def replace_task(run_id: str, new_task: Path) -> list[str]:
        """Swap in a new task.yaml for an existing run (validated first). Returns the changed top-level
        sections and logs a `task_updated` event; the report lists it."""
        run_dir = runs_dir() / run_id
        old = dump_task(load_task(run_dir / "task.yaml"))
        new_cfg = load_task(new_task)
        new = dump_task(new_cfg)
        if new_cfg.name != old["name"]:
            raise ConfigError(f"task name changed ({old['name']} → {new_cfg.name}); start a new run instead")
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        if changed:
            shutil.copyfile(run_dir / "task.yaml", run_dir / f"task.before-{datetime.now():%Y%m%d-%H%M%S}.yaml")
            shutil.copyfile(new_task, run_dir / "task.yaml")
            EventLog(run_dir / "events.jsonl").emit("task_updated", source=str(new_task), changed=changed)
        return changed

    @classmethod
    def _assemble(cls, run_dir, cfg, state, *, dry_run, yes, confirm, fake_opts) -> RunContext:
        fake_opts = fake_opts or FakeOptions(seed=cfg.splits.seed)
        events = EventLog(run_dir / "events.jsonl")
        prices = repo_prices(dry_run=dry_run)
        models = repo_models()
        cap = state.data.get("max_spend_usd") or cfg.budget.max_spend_usd
        guard = SpendGuard(
            state=state,
            events=events,
            prices=prices,
            cap_usd=cap,
            runs_dir=runs_dir(),
            dry_run=dry_run,
            yes=yes,
            confirm=confirm,
        )
        providers = build_providers(dry_run, models, fake_opts, cfg.train.provider)
        ctx = cls(
            run_dir=run_dir,
            cfg=cfg,
            state=state,
            events=events,
            providers=providers,
            spend=guard,
            prices=prices,
            models=models,
            dry_run=dry_run,
            yes=yes,
            fake_opts=fake_opts,
        )
        if (run_dir / "data" / "labels.json").exists():
            fake_opts.labels = ctx.labels()
        return ctx

    # ---------------------------------------------------------------- paths

    @property
    def run_id(self) -> str:
        return self.state.data["run_id"]

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    def attempt_dir(self, n: int | None = None) -> Path:
        return self.run_dir / "attempts" / str(n if n is not None else self.attempt)

    # ---------------------------------------------------------------- attempts
    # An attempt folder holds task.patch.yaml (its config changes; absent for attempt 1 as written) and
    # origin.json (who designed it: the fixer, a human, or a retrain; absent for attempt 1).

    def attempt_patch(self, n: int) -> dict[str, Any]:
        path = self.attempt_dir(n) / "task.patch.yaml"
        return (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}

    def attempt_origin(self, n: int) -> dict[str, Any]:
        path = self.attempt_dir(n) / "origin.json"
        return read_json(path) if path.exists() else {}

    def write_attempt(self, n: int, patch: dict[str, Any], *, by: str, **origin: Any) -> None:
        """Plan attempt n: who designed it and its config patch. The patch is written last; its presence
        marks the attempt as ready to run (the pipeline resumes from it)."""
        adir = self.attempt_dir(n)
        adir.mkdir(parents=True, exist_ok=True)
        write_json(adir / "origin.json", {"by": by, **origin, "ts": now_iso()})
        (adir / "task.patch.yaml").write_text(yaml.safe_dump(patch, sort_keys=False))

    def next_attempt(self) -> int:
        return max((int(k) for k in self.state.data["attempts"]), default=0) + 1

    def is_retrain(self, n: int | None = None) -> bool:
        """An attempt that retrains a val-chosen config on train + val (no val score of its own)."""
        return self.attempt_origin(n if n is not None else self.attempt).get("by") == "retrain"

    def eval_splits(self, n: int | None = None) -> tuple[str, ...]:
        """The splits an attempt is scored on. A retrain has seen val, so it only gets test."""
        return ("test",) if self.is_retrain(n) else ("val", "test")

    def val_metrics(self, n: int) -> dict[str, Any]:
        return read_json(self.attempt_dir(n) / "metrics.val.json")

    def evaluated_attempts(self) -> list[int]:
        """Attempts with a val score (so not retrains), in order."""
        attempts = (int(k) for k in self.state.data["attempts"])
        return sorted(n for n in attempts if (self.attempt_dir(n) / "metrics.val.json").exists())

    def best_attempt(self) -> int | None:
        """The final model's attempt: best val score, earliest on ties (spec 13)."""
        scored = [(self.val_metrics(n)["primary"], -n, n) for n in self.evaluated_attempts()]
        return max(scored)[2] if scored else None

    def tuned_model(self, n: int) -> ModelRef:
        """The model attempt n trained (its train stage must have finished)."""
        return ModelRef(
            model_object_id=self.state.external(n)["model_object_id"], base_model=self.cfg_for(n).train.base_model
        )

    def baseline_dir(self, name: str) -> Path:
        return self.run_dir / "baselines" / name.replace("/", "__")

    @staticmethod
    def resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    # ---------------------------------------------------------------- config

    def cfg_for(self, n: int | None = None) -> TaskConfig:
        """Effective config for an attempt: task.yaml + that attempt's patch (retry loop, M4)."""
        n = n if n is not None else self.attempt
        if n is None:
            return self.cfg
        patch = self.attempt_patch(n)
        return apply_patch(self.cfg, patch) if patch else self.cfg

    def items(self, rows: list[dict[str, Any]]) -> list[Item]:
        """Records -> provider Items. Image records store a file name under data/images/."""
        if self.cfg.task.input.kind == "image":
            return [Item(id=r["id"], input="", target=r["target"], image=str(self.image_path(r))) for r in rows]
        return [Item(id=r["id"], input=r["input"], target=r["target"]) for r in rows]

    def image_path(self, record: dict[str, Any]) -> Path:
        return self.data_dir / "images" / record["input"]

    def labels(self) -> list[str]:
        if self._labels is None:
            self._labels = read_json(self.data_dir / "labels.json")
        return self._labels

    def task_type(self, n: int | None = None) -> TaskType:
        labels = self.labels() if self.cfg.task.type == "classify" else None
        return get_task_type(self.cfg_for(n), labels)
