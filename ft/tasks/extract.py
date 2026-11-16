"""Extraction: a fixed JSON schema of typed fields, scored per field (spec 9.10).

Field types: text, money, number, date, and list_of {subfield: type}. Every field is required; a value
that isn't there is "" (or [] for lists), which keeps strict structured output simple.

Scoring, per example: each top-level field scores 1/0 on a normalized exact match; a list field scores
its item-level F1 (an item matches when all its normalized subfields match; both empty = 1). An
example's score is the mean over top-level fields, and field_accuracy is the mean over examples (x100).
Invalid output (not JSON, or missing a schema key) scores 0 on every field.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from ft.config import ListOf
from ft.providers.base import PromptSpec
from ft.tasks.base import Parsed, Scorer, ScoreRow, TaskType

TYPE_HINTS = {
    "text": "text, copied exactly as printed",
    "money": "amount, copied as printed",
    "number": "number, copied as printed",
    "date": "date, copied as printed",
}
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d %b %Y", "%b %d, %Y")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def normalize(value: Any, ftype: str) -> str:
    """text: casefold + collapse whitespace. money/number: digits only, leading zeros dropped (receipts mix
    "60.000", "60,000" and "Rp 60.000"). date: ISO if one of the common formats parses, else as text."""
    s = "" if value is None else str(value)
    if ftype in ("money", "number"):
        digits = re.sub(r"\D", "", s)
        return digits.lstrip("0") or ("0" if digits else "")
    s = " ".join(s.split()).casefold()
    if ftype == "date" and s:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt).date().isoformat()  # %b matches "jan" case-insensitively
            except ValueError:
                continue
    return s


class ExtractTask(TaskType):
    def __init__(self, cfg, labels=None):
        super().__init__(cfg, labels)
        self.schema: dict[str, str | ListOf] = cfg.task.output.schema_ or {}
        self.tuned_max_tokens = 1024
        self.baseline_max_tokens = 1024
        self.est_seconds_per_call = 2.0  # a JSON object, plus image prefill when the input is an image

    # ---- shape

    def conform(self, obj: Any) -> dict[str, Any]:
        """Coerce any dict to the schema: every key present, scalars as strings, list items as dicts of
        strings (a single dict becomes a one-item list). Used for targets and for parsed predictions."""
        obj = obj if isinstance(obj, dict) else {}
        out: dict[str, Any] = {}
        for name, ftype in self.schema.items():
            v = obj.get(name, "")
            if isinstance(ftype, ListOf):
                items = v if isinstance(v, list) else ([v] if isinstance(v, dict) else [])
                out[name] = [
                    {sub: _scalar(it.get(sub, "")) for sub in ftype.list_of} for it in items if isinstance(it, dict)
                ]
            else:
                out[name] = _scalar(v)
        return out

    def _describe(self) -> str:
        lines = []
        for name, ftype in self.schema.items():
            if isinstance(ftype, ListOf):
                subs = ", ".join(f"{s} ({TYPE_HINTS[t]})" for s, t in ftype.list_of.items())
                lines.append(f"- {name}: a list of items, each with {subs}. Use [] when there are none.")
            else:
                lines.append(f"- {name}: {TYPE_HINTS[ftype]}.")
        return "\n".join(lines)

    def json_schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for name, ftype in self.schema.items():
            if isinstance(ftype, ListOf):
                item = {
                    "type": "object",
                    "properties": {s: {"type": "string"} for s in ftype.list_of},
                    "required": list(ftype.list_of),
                    "additionalProperties": False,
                }
                props[name] = {"type": "array", "items": item}
            else:
                props[name] = {"type": "string"}
        return {"type": "object", "properties": props, "required": list(self.schema), "additionalProperties": False}

    # ---- prompts

    def baseline_prompt(self, context: str | None = None, examples: list[dict[str, Any]] | None = None) -> PromptSpec:
        parts = [self.instruction]
        if context:
            parts.append("Context:\n" + context.strip())
        parts.append(
            "Return JSON with exactly these fields. Copy values as they appear in the input; use an empty "
            'string ("") for a field that is not present.\n' + self._describe()
        )
        return PromptSpec(
            system="\n\n".join(parts), json_schema=self.json_schema(), max_output_tokens=self.baseline_max_tokens
        )

    def train_target(self, target: Any) -> str:
        """Compact JSON, keys in schema order (spec 9.5)."""
        return json.dumps(self.conform(target), ensure_ascii=False, separators=(",", ":"))

    def parse(self, raw: str | None, *, structured: bool) -> Parsed:
        if raw is None:
            return Parsed(None, False)
        text = _FENCE.sub("", raw.strip())
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")  # tolerate text around one JSON object
            try:
                obj = json.loads(text[start : end + 1]) if start >= 0 < end else None
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict) or any(k not in obj for k in self.schema):
            return Parsed(None, False)
        return Parsed(self.conform(obj), True)

    def scorer(self, rows: list[ScoreRow]) -> ExtractScorer:
        return ExtractScorer(rows, self)


def _scalar(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):  # e.g. CORD sometimes nests values; keep them readable
        return " ".join(_scalar(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _list_f1(pred: list[dict[str, str]], true: list[dict[str, str]], types: dict[str, str]) -> float:
    if not pred and not true:
        return 1.0
    norm = lambda it: tuple(normalize(it.get(s, ""), t) for s, t in types.items())  # noqa: E731
    p, t = Counter(norm(i) for i in pred), Counter(norm(i) for i in true)
    hit = sum((p & t).values())
    if hit == 0:
        return 0.0
    precision, recall = hit / sum(p.values()), hit / sum(t.values())
    return 2 * precision * recall / (precision + recall)


class ExtractScorer(Scorer):
    def __init__(self, rows: list[ScoreRow], task: ExtractTask):
        self.rows = rows
        self.task = task
        self.fields = list(task.schema)
        self.n = len(rows)
        # field_scores[i, j]: example i, field j, in [0, 1]
        self.field_scores = np.zeros((self.n, len(self.fields)))
        self.valid = np.array([r.valid for r in rows], dtype=bool)
        for i, r in enumerate(rows):
            if not r.valid:
                continue
            true, pred = task.conform(r.target), r.parsed
            for j, name in enumerate(self.fields):
                ftype = task.schema[name]
                if isinstance(ftype, ListOf):
                    self.field_scores[i, j] = _list_f1(pred[name], true[name], ftype.list_of)
                else:
                    self.field_scores[i, j] = float(normalize(pred[name], ftype) == normalize(true[name], ftype))

    def _field_accuracy(self, idx: np.ndarray) -> float:
        return float(self.field_scores[idx].mean() * 100) if len(idx) else float("nan")

    def metric_fn(self, name: str):
        if name == "field_accuracy":
            return self._field_accuracy
        raise ValueError(f"unknown extract metric '{name}'")

    def details(self) -> dict[str, Any]:
        per_field = []
        for j, name in enumerate(self.fields):
            ftype = self.task.schema[name]
            nonempty = sum(bool(self.task.conform(r.target)[name]) for r in self.rows)
            per_field.append(
                {
                    "field": name,
                    "type": "list" if isinstance(ftype, ListOf) else ftype,
                    "score": round(float(self.field_scores[:, j].mean() * 100), 2) if self.n else 0.0,
                    "present_in_target": nonempty,
                }
            )
        per_field.sort(key=lambda f: f["score"])
        errors = []
        for j, name in enumerate(self.fields):
            examples = []
            for i, r in enumerate(self.rows):
                if self.field_scores[i, j] >= 1.0:
                    continue
                expected = self.task.conform(r.target)[name]
                got = r.parsed[name] if r.valid else "(invalid output)"
                examples.append({"input": r.input, "expected": _short(expected), "predicted": _short(got)})
            if examples:
                errors.append({"field": name, "count": len(examples), "examples": examples[:3]})
        errors.sort(key=lambda e: -e["count"])
        return {
            "field_accuracy": round(self._field_accuracy(np.arange(self.n)), 2),
            "json_valid_rate": round(float(self.valid.mean() * 100), 2) if self.n else 0.0,
            "invalid_rate": round(float((~self.valid).mean() * 100), 2) if self.n else 0.0,
            "per_field": per_field,
            "field_errors": errors,
        }


def _short(v: Any, limit: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"
