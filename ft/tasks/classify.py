"""Classification: one label per input."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from ft.providers.base import PromptSpec
from ft.tasks.base import Parsed, Scorer, ScoreRow, TaskType


def normalize_label(s: str) -> str:
    return " ".join(s.split()).casefold()


class ClassifyTask(TaskType):
    tuned_max_tokens = 16

    def __init__(self, cfg, labels):
        super().__init__(cfg, labels)
        self._by_norm = {normalize_label(lbl): lbl for lbl in self.labels}
        # Label names can be long ("legal_and_compliance_escalation"); assume a dense ~2 chars/token.
        longest = max((len(lbl) for lbl in self.labels), default=0)
        self.tuned_max_tokens = max(16, longest // 2 + 8)
        self.baseline_max_tokens = max(64, longest // 2 + 32)

    def baseline_prompt(self, context: str | None = None, examples: list[dict[str, Any]] | None = None) -> PromptSpec:
        parts = [self.instruction]
        if context:
            parts.append("Context:\n" + context.strip())
        parts.append("Answer with exactly one label from this list:\n" + "\n".join(self.labels))
        if examples:
            shots = "\n\n".join(f"Message: {' '.join(e['input'].split())}\nLabel: {e['target']}" for e in examples)
            parts.append("Labeled examples:\n\n" + shots)
        schema = {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(self.labels)}},
            "required": ["label"],
            "additionalProperties": False,
        }
        return PromptSpec(system="\n\n".join(parts), json_schema=schema, max_output_tokens=self.baseline_max_tokens)

    def tuned_prompt(self) -> PromptSpec:
        system = self.instruction
        if self.cfg.train.labels_in_prompt:
            system += "\n\nAnswer with exactly one label from this list:\n" + "\n".join(self.labels)
        return PromptSpec(system=system, max_output_tokens=self.tuned_max_tokens)

    def train_target(self, target: Any) -> str:
        return str(target)

    def canonical(self, text: str) -> str | None:
        return self._by_norm.get(normalize_label(text))

    def parse(self, raw: str | None, *, structured: bool) -> Parsed:
        if raw is None:
            return Parsed(None, False)
        text = raw
        if structured:
            try:
                obj = json.loads(raw)
                text = obj.get("label", "") if isinstance(obj, dict) else str(obj)
            except (json.JSONDecodeError, TypeError):
                text = raw
        label = self.canonical(str(text))
        return Parsed(label, label is not None)

    def scorer(self, rows: list[ScoreRow]) -> ClassifyScorer:
        return ClassifyScorer(rows, self.labels)


class ClassifyScorer(Scorer):
    def __init__(self, rows: list[ScoreRow], labels: list[str]):
        self.rows = rows
        # Targets unseen in train (labels: auto) still need a slot; nothing can predict them.
        unseen = sorted({r.target for r in rows} - set(labels))
        self.labels = list(labels) + unseen
        labels = self.labels
        self.n = len(rows)
        index = {lbl: i for i, lbl in enumerate(labels)}
        self.k = len(labels)
        self.y_true = np.array([index[r.target] for r in rows], dtype=np.int64)
        self.y_pred = np.array([index[r.parsed] if r.valid else -1 for r in rows], dtype=np.int64)

    # ---- metrics in points (0-100), computed on a subset of row indices

    def _accuracy(self, idx: np.ndarray) -> float:
        return float(np.mean(self.y_true[idx] == self.y_pred[idx]) * 100) if len(idx) else float("nan")

    def _counts(self, idx: np.ndarray):
        t, p = self.y_true[idx], self.y_pred[idx]
        hit = t == p
        tp = np.bincount(t[hit], minlength=self.k)
        true_n = np.bincount(t, minlength=self.k)
        pred_n = np.bincount(p[p >= 0], minlength=self.k)
        return tp, true_n, pred_n

    def _macro_f1(self, idx: np.ndarray) -> float:
        tp, true_n, pred_n = self._counts(idx)
        denom = true_n + pred_n
        present = denom > 0
        if not present.any():
            return float("nan")
        f1 = np.where(present, 2 * tp / np.maximum(denom, 1), 0.0)
        return float(f1[present].mean() * 100)

    def metric_fn(self, name: str):
        if name in ("accuracy", "agreement"):
            return self._accuracy
        if name == "macro_f1":
            return self._macro_f1
        raise ValueError(f"unknown classify metric '{name}'")

    def details(self) -> dict[str, Any]:
        all_idx = np.arange(self.n)
        tp, true_n, pred_n = self._counts(all_idx)
        per_class = []
        for i, lbl in enumerate(self.labels):
            if true_n[i] == 0 and pred_n[i] == 0:
                continue
            prec = tp[i] / pred_n[i] if pred_n[i] else 0.0
            rec = tp[i] / true_n[i] if true_n[i] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            per_class.append(
                {
                    "label": lbl,
                    "support": int(true_n[i]),
                    "precision": round(prec * 100, 2),
                    "recall": round(rec * 100, 2),
                    "f1": round(f1 * 100, 2),
                }
            )
        per_class.sort(key=lambda r: (r["f1"], -r["support"]))

        pairs: Counter[tuple[str, str]] = Counter()
        examples: dict[tuple[str, str], list[str]] = {}
        for r in self.rows:
            if r.valid and r.parsed == r.target:
                continue
            key = (r.target, r.parsed if r.valid else "(invalid)")
            pairs[key] += 1
            ex = examples.setdefault(key, [])
            if len(ex) < 3:
                ex.append(r.input[:300])
        confusions = [
            {"true": t, "pred": p, "count": c, "examples": examples[(t, p)]} for (t, p), c in pairs.most_common(10)
        ]
        return {
            "accuracy": round(self._accuracy(all_idx), 2),
            "macro_f1": round(self._macro_f1(all_idx), 2),
            "invalid_rate": round(float(np.mean(self.y_pred == -1) * 100), 2) if self.n else 0.0,
            "per_class": per_class,
            "confusions": confusions,
        }
