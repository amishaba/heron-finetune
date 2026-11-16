"""The agent's tools (spec 12). Each takes JSON arguments and returns text (JSON) or images.

The agent may read anything in the folder except test files' rows: those it sees only through the
profile's column summaries, so the test set stays held out (the split itself happens after approval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ft.agent.estimate import EstimateError
from ft.agent.inspect import Folder, InspectError, read_table
from ft.agent.rules import RuleResult, apply_rules
from ft.config import ConfigError, CsvSource, ImageFolderSource, JsonlSource, TaskConfig, parse_task
from ft.registry import BaseModelInfo
from ft.spend import Prices, SpendError
from ft.stages.load import SPLIT_NAMES, source_files

RATIONALE_KEYS = ("task", "label_source", "base_model", "metric", "split", "baselines")
# Why a plan was declined: the data doesn't suit fine-tuning, or it does but the provider has no model for it.
DECLINE_FITS = {
    "not_a_fit": "not a good fit for fine-tuning",
    "no_available_model": "good fit, but no available model on {provider} can serve it",
}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


TOOLS: list[dict[str, Any]] = [
    {"name": "get_profile", "description": "Profile of the data folder: files, tables with per-column stats "
     "(dtype, nulls, lengths, cardinality, top values, whether it holds JSON), images, mail-like files.",
     "parameters": _obj({})},
    {"name": "sample_rows", "description": "Random rows from a table (values truncated to 1,000 chars). "
     "Optional filter: rows where `column` equals `equals`. Test files can't be sampled.",
     "parameters": _obj({"file": {"type": "string"}, "n": {"type": "integer"},
                         "where": {"anyOf": [{"type": "null"}, _obj({"column": {"type": "string"},
                                                                       "equals": {"type": "string"}})]}})},
    {"name": "value_counts", "description": "Most common values of one column in a table (not test files).",
     "parameters": _obj({"file": {"type": "string"}, "column": {"type": "string"}, "top": {"type": "integer"}})},
    {"name": "view_images", "description": "Look at up to 6 images from the folder (paths relative to it).",
     "parameters": _obj({"paths": {"type": "array", "items": {"type": "string"}}})},
    {"name": "list_base_models", "description": "Deployable base models that can be fine-tuned for an input "
     "kind ('text' or 'image'), smallest first.",
     "parameters": _obj({"kind": {"type": "string", "enum": ["text", "image"]}})},
    {"name": "estimate_cost", "description": "Estimated cost and time of a full run (baselines, training, "
     "endpoint) for a draft task.yaml, after the fixed rules are applied, plus the split sizes it would produce.",
     "parameters": _obj({"task_yaml": {"type": "string"}})},
    {"name": "validate_task", "description": "Apply the fixed rules (base model, baselines, test size, metric, "
     "pass bar, concurrency, budget) and check the draft against the schema and the data (files, columns, "
     "labels, cost limits). Returns errors, warnings, the corrections made, and the corrected task_yaml.",
     "parameters": _obj({"task_yaml": {"type": "string"}})},
    {"name": "submit_task", "description": "Submit the final task.yaml (must validate; the fixed rules are "
     "applied again). Give a one-line rationale for each decision. Ends the session.",
     "parameters": _obj({"task_yaml": {"type": "string"},
                         "rationale": _obj({k: {"type": "string"} for k in RATIONALE_KEYS})})},
    {"name": "decline", "description": "Don't fine-tune this data here. fit='not_a_fit': fine-tuning isn't "
     "a good fit for the data. fit='no_available_model': it would be a good fit, but list_base_models found no "
     "model on this provider that serves the input kind (suggest trying another provider). Explain why and "
     "suggest what would work instead. Ends the session.",
     "parameters": _obj({"fit": {"type": "string", "enum": list(DECLINE_FITS)}, "reason": {"type": "string"},
                         "suggestion": {"type": "string"}})},
]  # fmt: skip


@dataclass
class Outcome:
    kind: str  # "submitted" | "declined"
    task_yaml: str | None = None
    rationale: dict[str, str] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)  # rules applied to the submitted draft
    reason: str | None = None
    suggestion: str | None = None
    fit: str | None = None  # a DECLINE_FITS key, for declines


class AgentTools:
    def __init__(
        self,
        folder: Folder,
        prices: Prices,
        models: dict[str, BaseModelInfo],
        provider: str = "together",
        runs_root: Path | None = None,  # for the all-runs budget check; None skips it
    ):
        self.folder = folder
        self.prices = prices
        self.models = models
        self.provider = provider
        self.runs_root = runs_root
        self.unserved: set[str] = set()  # input kinds list_base_models found no model for
        self.outcome: Outcome | None = None

    def call(self, name: str, args: dict[str, Any]) -> str | list[str]:
        """Runs one tool. Returns text, or a list of image paths for view_images. Errors come back as text
        so the agent can correct itself."""
        try:
            return getattr(self, f"_{name}")(**args)
        except (InspectError, EstimateError, ConfigError, SpendError, OSError, ValueError, KeyError) as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
        except (AttributeError, TypeError) as e:
            return json.dumps({"error": f"bad call to {name}: {e}"})

    def _test_file(self, rel: str) -> bool:
        return SPLIT_NAMES.get(Path(rel).stem.lower()) == "test"

    # ---- tools

    def _get_profile(self) -> str:
        return json.dumps(self.folder.profile(), default=str)

    def _sample_rows(self, file: str, n: int, where: dict[str, str] | None = None) -> str:
        if self._test_file(file):
            return json.dumps({"error": "test files are held out; sample the train file instead"})
        return json.dumps(self.folder.sample(file, n, where), default=str, ensure_ascii=False)

    def _value_counts(self, file: str, column: str, top: int) -> str:
        if self._test_file(file):
            return json.dumps({"error": "test files are held out; use the train file instead"})
        return json.dumps(self.folder.value_counts(file, column, top), default=str, ensure_ascii=False)

    def _view_images(self, paths: list[str]) -> list[str] | str:
        if not paths:
            return json.dumps({"error": "give 1-6 image paths"})
        return [str(self.folder.file(p)) for p in paths[:6]]

    def _list_base_models(self, kind: str) -> str:
        ok = sorted(
            (m for m in self.models.values() if m.provider == self.provider and m.deployable and m.accepts(kind)),
            key=lambda m: m.size_b,
        )
        out = [{"id": m.id, "size_b": m.size_b, "notes": m.notes} for m in ok]
        if not out:
            self.unserved.add(kind)
            return json.dumps(
                {
                    "models": [],
                    "note": f"no deployable model on {self.provider} is verified to serve "
                    f"{kind} inputs; if the data is otherwise a good fit, decline with "
                    "fit='no_available_model'",
                }
            )
        return json.dumps({"models": out})

    def _estimate_cost(self, task_yaml: str) -> str:
        rules = self.rules(task_yaml)
        if rules.estimate is None:
            return json.dumps({"error": "; ".join(rules.errors)})
        return json.dumps({**rules.estimate, "corrections": rules.corrections, "errors": rules.errors})

    def _validate_task(self, task_yaml: str) -> str:
        return json.dumps(self.validate(task_yaml))

    def _submit_task(self, task_yaml: str, rationale: dict[str, str]) -> str:
        result = self.validate(task_yaml)
        if not result["valid"]:
            return json.dumps({"submitted": False, **result})
        self.outcome = Outcome(
            "submitted", task_yaml=result["task_yaml"], rationale=rationale, corrections=result["corrections"]
        )
        return json.dumps({"submitted": True, "warnings": result["warnings"], "corrections": result["corrections"]})

    def _decline(self, reason: str, suggestion: str, fit: str = "not_a_fit") -> str:
        if fit not in DECLINE_FITS:
            return json.dumps({"error": f"fit must be one of {list(DECLINE_FITS)}"})
        if fit == "no_available_model":
            if not self.unserved:
                return json.dumps(
                    {
                        "error": "no_available_model needs list_base_models to have returned no model "
                        "for the input kind; call it first, or decline with fit='not_a_fit'"
                    }
                )
            if "provider" not in suggestion.lower():  # the one suggestion this kind of decline always needs
                kinds = " and ".join(sorted(self.unserved))
                suggestion = (
                    suggestion.rstrip(". ")
                    + (". " if suggestion.strip() else "")
                    + (f"Try another fine-tuning provider whose deployed models serve {kinds} inputs.")
                )
        self.outcome = Outcome("declined", reason=reason, suggestion=suggestion, fit=fit)
        return json.dumps({"declined": True})

    def fit_label(self, fit: str | None) -> str:
        return DECLINE_FITS.get(fit or "", "").replace("{provider}", self.provider.capitalize())

    # ---- validation against the data

    def _load(self, task_yaml: str) -> dict[str, Any]:
        try:
            raw = yaml.safe_load(task_yaml)
        except yaml.YAMLError as e:
            raise ConfigError(f"not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError("task.yaml must be a mapping")
        return raw

    def _parse(self, task_yaml: str) -> TaskConfig:
        return parse_task(self._load(task_yaml), where="draft task.yaml")

    def rules(self, task_yaml: str) -> RuleResult:
        try:
            raw = self._load(task_yaml)
        except ConfigError as e:
            return RuleResult(raw={}, errors=str(e).splitlines())
        return apply_rules(raw, self.folder, self.prices, self.models, self.runs_root)

    def validate(self, task_yaml: str) -> dict[str, Any]:
        """Rules first (corrections, cost limits), then the corrected draft against the data. The
        corrected text is returned as task_yaml (the agent's own text when nothing changed)."""
        rules = self.rules(task_yaml)
        if rules.cfg is None:
            return {"valid": False, "errors": rules.errors, "warnings": [], "corrections": rules.corrections,
                    "task_yaml": task_yaml}  # fmt: skip
        cfg = rules.cfg
        fixed = yaml.safe_dump(rules.raw, sort_keys=False, allow_unicode=True) if rules.corrections else task_yaml
        errors: list[str] = list(rules.errors)
        warnings: list[str] = []
        repo = self.folder.repo_root
        src = cfg.source
        base = self.models.get(cfg.train.base_model)
        if base is None or not base.deployable:
            errors.append(f"train.base_model {cfg.train.base_model} is not a deployable model in models.yaml")
        elif not base.accepts(cfg.task.input.kind):
            errors.append(f"{base.id} is not verified to serve {cfg.task.input.kind} inputs (models.yaml)")
        for b in cfg.baselines:
            try:
                self.prices.token_price(b.model)
            except SpendError as e:
                errors.append(str(e))
        rows: list[dict[str, Any]] = []
        if isinstance(src, CsvSource | JsonlSource):
            try:
                files = [f for f, _hint in source_files(src, repo / src.path)]
            except ConfigError as e:
                files = []
                errors.append(str(e))
            need = [cfg.task.input.field, src.target_map.column if src.target_map else cfg.task.output.field]
            if cfg.splits.by_time:
                need.append(cfg.splits.by_time)
            for f in files:
                part = read_table(f, limit=5000)
                cols = set(part[0]) if part else set()
                missing = [c for c in need if c not in cols]
                if missing:
                    errors.append(f"{f.name}: missing columns {missing}; columns are {sorted(cols)}")
                rows += part
        elif isinstance(src, ImageFolderSource):
            labels = repo / src.labels_file
            if not labels.exists():
                errors.append(f"labels_file {src.labels_file} not found")
            else:
                rows = read_table(labels, limit=5000)
                cols = set(rows[0]) if rows else set()
                if cfg.task.input.field not in cols:
                    errors.append(f"{labels.name}: no column {cfg.task.input.field!r} with image file names")
                else:
                    gone = [
                        r[cfg.task.input.field]
                        for r in rows[:20]
                        if not (repo / src.path / str(r[cfg.task.input.field])).exists()
                    ]
                    if gone:
                        errors.append(f"images not found under {src.path}: {gone[:3]}")
        elif src.type in ("maildir", "mbox"):
            if not (repo / src.path).exists():
                errors.append(f"source.path {src.path} not found")
        else:
            warnings.append(f"source type {src.type} can't be checked against the data before loading")
        out = cfg.task.output
        if cfg.task.type == "classify" and rows and not src.target_map and out.field in rows[0]:
            values = {str(r.get(out.field)) for r in rows if str(r.get(out.field, "")).strip()}
            if isinstance(out.labels, list) and values - set(out.labels):
                errors.append(
                    f"labels in the data but not in task.output.labels: {sorted(values - set(out.labels))[:10]}"
                )
            if len(values) < 2:
                errors.append(f"{out.field} has fewer than 2 distinct values")
        if rules.estimate and not rules.estimate["fits_budget"]:
            errors.append(
                f"estimated cost ${rules.estimate['total_usd']} exceeds budget.max_spend_usd {cfg.budget.max_spend_usd}"
            )
        return {"valid": not errors, "errors": errors, "warnings": warnings, "corrections": rules.corrections,
                "task_yaml": fixed}  # fmt: skip
