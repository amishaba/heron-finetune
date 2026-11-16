"""task.yaml schema (spec section 7). Unknown keys are errors everywhere."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- source


class TargetMapField(Strict):
    path: str
    as_list: bool = False
    item: dict[str, str] | None = None


class TargetMap(Strict):
    """Builds `target` from a JSON column (spec 9.1)."""

    column: str
    root: str | None = None
    fields: dict[str, str | TargetMapField]


class _SourceBase(Strict):
    target_map: TargetMap | None = None


class FilesSplit(Strict):
    train: str | None = None
    val: str | None = None
    test: str | None = None


class CsvSource(_SourceBase):
    type: Literal["csv"]
    path: str
    files: FilesSplit | None = None


class JsonlSource(_SourceBase):
    type: Literal["jsonl"]
    path: str
    files: FilesSplit | None = None


class ImageFolderSource(_SourceBase):
    """Images in `path` plus a CSV/JSONL labels file with one row per image: task.input.field holds the
    file name (relative to `path`), the target comes from task.output.field or target_map, and an optional
    `split` column (train/val/test) marks predefined splits."""

    type: Literal["image_folder"]
    path: str
    labels_file: str


class HfDatasetSource(_SourceBase):
    type: Literal["hf_dataset"]
    dataset: str
    name: str | None = None  # dataset config name, when it has several
    splits: FilesSplit | None = None  # our split -> the dataset's split name; default: all of "train"
    # Read the dataset as a stream (no copy in ~/.cache/huggingface): images are downscaled in memory and
    # only the small JPEGs are kept in the run folder. false = normal download into the HF cache.
    streaming: bool = True


class GmailSource(_SourceBase):
    type: Literal["gmail"]
    months: int = Field(12, ge=1, le=60)
    min_age_days: int = Field(7, ge=0)
    export: str = "data/email/messages.jsonl"  # written by `ft email pull`; runs read only this file


class _MailFilesSource(_SourceBase):
    """Local mail files; the folder a message was filed in is its label (ft/connectors/mailfiles.py)."""

    path: str
    exclude_folders: list[str] = []  # glob patterns on the folder path, case-insensitive ("inbox", "sent*")
    min_per_folder: int = Field(20, ge=1)  # smaller folders are dropped (and listed in load_stats.json)
    redact: bool = True  # false only for public corpora; personal mail should always be redacted


class MaildirSource(_MailFilesSource):
    type: Literal["maildir"]


class MboxSource(_MailFilesSource):
    type: Literal["mbox"]
    folder_from: Literal["file", "header"] = "file"
    folder_header: str = "X-Gmail-Labels"  # Google Takeout


MAIL_SOURCES = {"gmail", "maildir", "mbox"}


class LogsSource(_SourceBase):
    type: Literal["logs"]
    path: str
    truth_sidecar: str | None = None


Source = Annotated[
    CsvSource
    | JsonlSource
    | ImageFolderSource
    | HfDatasetSource
    | GmailSource
    | MaildirSource
    | MboxSource
    | LogsSource,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------- task

FieldType = Literal["text", "money", "number", "date"]


class ListOf(Strict):
    list_of: dict[str, FieldType]


class TaskInput(Strict):
    kind: Literal["text", "image"] = "text"
    field: str
    max_side: int = Field(1280, ge=64, le=4096)  # images: longest side after downscaling at load (spec 9.5)


class TaskOutput(Strict):
    field: str
    labels: Literal["auto"] | list[str] | None = None
    schema_: dict[str, FieldType | ListOf] | None = Field(default=None, alias="schema")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Task(Strict):
    type: Literal["classify", "extract"]
    input: TaskInput
    output: TaskOutput
    instruction: str

    @model_validator(mode="after")
    def _shape(self) -> Task:
        if self.type == "classify":
            if self.output.labels is None:
                raise ValueError("classify tasks need task.output.labels ('auto' or a list)")
            if self.output.schema_ is not None:
                raise ValueError("task.output.schema is only for extract tasks")
            if isinstance(self.output.labels, list) and len(set(self.output.labels)) < 2:
                raise ValueError("classify needs at least 2 distinct labels")
        else:
            if self.output.schema_ is None:
                raise ValueError("extract tasks need task.output.schema")
            if self.output.labels is not None:
                raise ValueError("task.output.labels is only for classify tasks")
            if _leaf_count(self.output.schema_) > 15:
                raise ValueError("extract schemas are limited to 15 leaf fields (spec 7.2)")
        return self


def _leaf_count(schema: dict[str, Any]) -> int:
    return sum(len(v.list_of) if isinstance(v, ListOf) else 1 for v in schema.values())


# ---------------------------------------------------------------- targets, splits


class Targets(Strict):
    from_: Literal["dataset", "behavior", "frontier", "logs"] = Field(alias="from")
    frontier_model: str | None = None
    rules: list[dict[str, str]] | None = None  # behavior rules (M5)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _shape(self) -> Targets:
        if self.from_ == "frontier" and not self.frontier_model:
            raise ValueError("targets.frontier_model is required when targets.from is 'frontier'")
        if self.from_ == "behavior" and not self.rules:
            raise ValueError("targets.rules is required when targets.from is 'behavior'")
        return self


class Splits(Strict):
    use_predefined: bool = True
    val_fraction: float = Field(0.1, gt=0, lt=1)
    val_max: int = Field(1000, gt=0)
    test_max: int = Field(2000, gt=0)
    test_fraction: float = Field(0.2, gt=0, lt=1)  # used only when there is no predefined test split
    by_time: str | None = None
    seed: int = 42


# ---------------------------------------------------------------- baselines, train, eval, budget

# Luna accepts these (developers.openai.com model page, checked 2026-10-16); there is no "minimal".
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


class Baseline(Strict):
    """One frontier model to compare against. OpenAI-shaped."""

    model: str
    reasoning_effort: ReasoningEffort | None = None
    context_file: str | None = None
    # N training examples per label shown in the (cached) system prompt; classify only.
    few_shot: int | None = Field(None, ge=1, le=20)
    # Unique id for folders, the report and pass_bar.compare_to. Defaults to the model name when only
    # one baseline uses that model, else "<model>-<effort>[-<N>shot]".
    name: str | None = Field(None, pattern=r"^[A-Za-z0-9._-]{1,80}$")


class Train(Strict):
    provider: Literal["together", "fireworks"] = "together"
    base_model: str
    fallback_models: list[str] = []
    lora: bool = True
    epochs: int = Field(3, ge=1, le=6)
    learning_rate: float = Field(1.0e-4, gt=0)
    max_examples: int | None = Field(None, gt=0)
    # Classify only. Show the label list in the tuned model's system prompt, in training and at inference,
    # so a label's name can help when it has few examples.
    labels_in_prompt: bool = False
    # Classify only. Give rare classes more weight by repeating their training rows: weight =
    # (median class size / class size) ** p, p = 0.5 for sqrt or 1 for balanced, clipped to [1, cap].
    class_weighting: Literal["none", "sqrt", "balanced"] = "none"
    class_weight_cap: float = Field(5.0, ge=1, le=20)
    # After the loop picks a config by val, retrain it on train + val for the shipped model and its test
    # numbers (NOTES.md, M4 design note 1). null = on for time splits (val is the period just before test).
    retrain_with_val: bool | None = None

    def hash_view(self) -> dict[str, Any]:
        """model_dump for stage hashes. Options added later count only when set, so runs made before
        they existed keep their hashes (and aren't re-run or re-polled)."""
        later = {"labels_in_prompt": False, "class_weighting": "none", "class_weight_cap": 5.0,
                 "retrain_with_val": None}  # fmt: skip
        return {k: v for k, v in self.model_dump().items() if k not in later or v != later[k]}


# Logs mode: the bar is relative to how often the logged model agrees with itself when re-asked the val
# inputs (NOTES.md, M6), since imitation can't be held to a higher standard than the model being imitated.
LOGGED_SELF_AGREEMENT = "logged_self_agreement"


class PassBar(Strict):
    compare_to: str | None = None
    max_gap_points: float | None = Field(None, ge=0)
    min_value: float | None = Field(None, ge=0, le=100)

    @model_validator(mode="after")
    def _one_shape(self) -> PassBar:
        relative = self.compare_to is not None or self.max_gap_points is not None
        absolute = self.min_value is not None
        if relative == absolute:
            raise ValueError(
                "pass_bar must be either {compare_to, max_gap_points} (relative) or {min_value} (absolute)"
            )
        if relative and (self.compare_to is None or self.max_gap_points is None):
            raise ValueError("relative pass_bar needs both compare_to and max_gap_points")
        return self

    @property
    def is_relative(self) -> bool:
        return self.compare_to is not None


class Eval(Strict):
    primary_metric: Literal["accuracy", "macro_f1", "agreement", "field_accuracy"]
    pass_bar: PassBar
    concurrency: int = Field(16, ge=1, le=256)


class Budget(Strict):
    max_spend_usd: float = Field(gt=0)
    max_attempts: int = Field(3, ge=1, le=10)


# ---------------------------------------------------------------- root

CLASSIFY_METRICS = {"accuracy", "macro_f1", "agreement"}
EXTRACT_METRICS = {"field_accuracy"}


class TaskConfig(Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    description: str | None = None
    source: Source
    task: Task
    targets: Targets
    splits: Splits = Splits()
    baselines: list[Baseline] = []
    train: Train
    eval: Eval
    budget: Budget

    @model_validator(mode="after")
    def _cross_checks(self) -> TaskConfig:
        metric = self.eval.primary_metric
        allowed = CLASSIFY_METRICS if self.task.type == "classify" else EXTRACT_METRICS
        if metric not in allowed:
            raise ValueError(f"eval.primary_metric '{metric}' does not fit task type '{self.task.type}'")
        if metric == "agreement" and self.targets.from_ != "logs":
            raise ValueError("primary_metric 'agreement' is only for targets.from: logs")
        models = Counter(b.model for b in self.baselines)
        for b in self.baselines:
            if b.name is None:
                b.name = b.model if models[b.model] == 1 else default_baseline_name(b)
            if b.few_shot and self.task.type != "classify":
                raise ValueError("baselines[*].few_shot is only supported for classify tasks")
        if self.task.type != "classify" and (self.train.labels_in_prompt or self.train.class_weighting != "none"):
            raise ValueError("train.labels_in_prompt and train.class_weighting are only for classify tasks")
        if len({b.name for b in self.baselines}) != len(self.baselines):
            raise ValueError("baseline names must be unique; set `name` on baselines that share a model")
        bar = self.eval.pass_bar
        if bar.is_relative and bar.compare_to == LOGGED_SELF_AGREEMENT:
            if self.targets.from_ != "logs":
                raise ValueError(f"pass_bar.compare_to '{LOGGED_SELF_AGREEMENT}' is only for targets.from: logs")
        elif bar.is_relative:
            names = {b.name for b in self.baselines}
            if not self.baselines:
                raise ValueError("a relative pass_bar needs at least one baseline; use min_value instead")
            if bar.compare_to != "best_baseline" and bar.compare_to not in names:
                raise ValueError(f"pass_bar.compare_to '{bar.compare_to}' is not 'best_baseline' or a baseline")
        if self.task.input.kind == "image" and self.source.type not in ("hf_dataset", "image_folder"):
            raise ValueError("image inputs need source.type hf_dataset or image_folder")
        if self.source.type == "image_folder" and self.task.input.kind != "image":
            raise ValueError("source.type image_folder needs task.input.kind: image")
        if self.targets.from_ == "logs" and self.source.type != "logs":
            raise ValueError("targets.from: logs needs source.type: logs")
        if self.splits.by_time is None and self.source.type in MAIL_SOURCES:
            raise ValueError(f"email ({self.source.type}) runs must split by_time (spec 9.3)")
        if self.source.type in ("maildir", "mbox") and self.targets.from_ != "dataset":
            raise ValueError(
                f"{self.source.type} sources label each message with its folder; use targets.from: dataset"
            )
        if self.targets.from_ == "behavior":
            from ft.connectors.gmail import FEATURES
            from ft.connectors.rules import RuleError, Rules

            try:
                rule_labels = Rules(self.targets.rules or [], FEATURES).labels
            except RuleError as e:
                raise ValueError(f"targets.rules: {e}") from e
            declared = self.task.output.labels
            if isinstance(declared, list) and set(rule_labels) - set(declared):
                raise ValueError(
                    f"targets.rules produce labels {sorted(set(rule_labels) - set(declared))} not in task.output.labels"
                )
        return self


def default_baseline_name(b: Baseline) -> str:
    name = f"{b.model}-{b.reasoning_effort or 'default'}"
    return name + (f"-{b.few_shot}shot" if b.few_shot else "")


class ConfigError(Exception):
    pass


def load_task(path: Path) -> TaskConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return parse_task(raw, where=str(path))


def parse_task(raw: dict[str, Any], where: str = "task.yaml") -> TaskConfig:
    try:
        return TaskConfig.model_validate(raw)
    except ValidationError as e:
        lines = [f"{where} is invalid:"]
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from e


def dump_task(cfg: TaskConfig) -> dict[str, Any]:
    return cfg.model_dump(mode="json", by_alias=True, exclude_none=True)


def retrain_with_val(cfg: TaskConfig) -> bool:
    return cfg.train.retrain_with_val if cfg.train.retrain_with_val is not None else bool(cfg.splits.by_time)


def apply_patch(cfg: TaskConfig, patch: dict[str, Any]) -> TaskConfig:
    """Deep-merge a config patch (retry loop, M4) and re-validate."""
    merged = _deep_merge(dump_task(cfg), patch)
    return parse_task(merged, where="patched task.yaml")


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
