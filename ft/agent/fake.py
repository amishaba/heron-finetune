"""A deterministic stand-in for the agent's model, for --dry-run and tests. No network, no spend.

It drives the same tools as the real agent with simple heuristics: the longest text column is the input,
the lowest-cardinality short column is the label, <500 rows or no short label column means decline, and
image tasks are declined when no model serves images. Good enough to exercise the whole loop.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from ft.agent.inspect import MAX_ROWS
from ft.providers.base import AgentTurn, ToolCall

MIN_ROWS = 500
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


class FakeAgent:
    model = "fake-agent"
    simulated = True

    def __init__(self):
        self.last: dict[str, Any] = {}
        self.queue: list[ToolCall] = []
        self.n = 0
        self.profile: dict[str, Any] | None = None
        self.draft: str | None = None
        self.plan_rationale: dict[str, str] = {}

    def start(self, system: str, user: str, tools: list[dict[str, Any]]) -> None:
        self.queue = [self._call("get_profile", {})]

    def _call(self, name: str, args: dict[str, Any]) -> ToolCall:
        self.n += 1
        return ToolCall(f"call_{self.n}", name, args)

    def step(self) -> AgentTurn:
        calls, self.queue = self.queue, []
        return AgentTurn(calls=calls, input_tokens=0, output_tokens=0)

    def nudge(self, text: str) -> None:
        self.queue = [self._decline("fake agent stalled", "use the real agent")]

    def add_results(self, results: list[tuple[str, str | list[str]]]) -> None:
        for _call_id, out in results:
            data = json.loads(out) if isinstance(out, str) else {}
            if self.profile is None and "tables" in data:
                self.profile = data
                self.queue = [self._decide()]
            elif "models" in data:
                self.queue = [self._after_models(data["models"])]
            elif "valid" in data and "submitted" not in data:
                if data["valid"]:
                    self.queue = [
                        self._call("submit_task", {"task_yaml": self.draft, "rationale": self.plan_rationale})
                    ]
                else:
                    why = "couldn't produce a valid task.yaml: " + "; ".join(data["errors"])[:300]
                    self.queue = [self._decline(why, "write task.yaml by hand")]

    def _decline(self, reason: str, suggestion: str, fit: str = "not_a_fit") -> ToolCall:
        return self._call("decline", {"fit": fit, "reason": reason, "suggestion": suggestion})

    # ---- heuristics

    def _decide(self) -> ToolCall:
        p = self.profile or {}
        tables = p.get("tables", {})
        rows = sum(t["rows"] for t in tables.values())
        if not tables:
            return self._decline("no tables with labels in the folder", "add a CSV/JSONL with inputs and answers")
        main_name = next((n for n, t in tables.items() if t["split_hint"] in ("train", None)), next(iter(tables)))
        cols = tables[main_name]["columns"]
        image_col = next(
            (c for c, s in cols.items() if any(str(v).lower().endswith(IMAGE_SUFFIXES) for v, _ in s["top_values"])),
            None,
        )
        if rows < MIN_ROWS:
            return self._decline(
                f"only {rows} labeled rows; a fine-tune needs at least {MIN_ROWS} to learn and to be measured",
                "prompt a frontier model with a few examples instead, or collect more labeled data",
            )
        if image_col and p.get("images"):
            self.kind = "image"
            return self._call("list_base_models", {"kind": "image"})
        text = {c: s for c, s in cols.items() if s["dtype"] == "string" and not s.get("json")}
        if len(text) < 2:
            return self._decline("no text input column with a separate label column", "add an input and a label column")
        self.input_col = max(text, key=lambda c: text[c]["length"]["mean"])
        labels = {
            c: s
            for c, s in text.items()
            if c != self.input_col
            and s["length"]["mean"] <= 30  # a label is a short code, not a sentence or a question
            and not any("?" in str(v) for v, _ in s["top_values"])
            and 2 <= s["cardinality"] <= max(2, rows // 5)
        }
        if not labels:
            return self._decline(
                "the answers are open-ended free text with no consistent target to grade automatically",
                "keep a frontier model and define a grading rubric before considering fine-tuning",
            )
        self.label_col = min(labels, key=lambda c: labels[c]["cardinality"])
        stats = labels[self.label_col]
        read = min(tables[main_name]["rows"], MAX_ROWS) or 1  # top_values counts the rows the profile read
        self.top_share = stats["top_values"][0][1] / read
        self.tables = tables
        self.kind = "text"
        return self._call("list_base_models", {"kind": "text"})

    def _after_models(self, models: list[dict[str, Any]]) -> ToolCall:
        if not models:
            return self._decline(
                f"labeled {self.kind} data suits fine-tuning, but no deployable base model here serves {self.kind} "
                "inputs",
                "use a frontier model for now",
                fit="no_available_model",
            )
        p = self.profile or {}
        folder = p["folder"]
        ext = {n.rsplit(".", 1)[-1] for n in self.tables}
        src_type = "csv" if ext == {"csv"} else "jsonl"
        hinted = any(t["split_hint"] for t in self.tables.values())
        test_rows = sum(t["rows"] for t in self.tables.values() if t["split_hint"] == "test")
        metric = "macro_f1" if self.top_share > 0.4 else "accuracy"
        name = (
            "".join(ch if ch.isalnum() else "-" for ch in folder.rsplit("/", 1)[-1].lower()).strip("-")[:40] or "task"
        )
        cfg = {
            "name": name,
            "description": f"Predict {self.label_col} from {self.input_col}.",
            "source": {"type": src_type, "path": folder},
            "task": {
                "type": "classify",
                "input": {"kind": "text", "field": self.input_col},
                "output": {"field": self.label_col, "labels": "auto"},
                "instruction": f"Classify the {self.input_col} into its {self.label_col}.",
            },
            "targets": {"from": "dataset"},
            "splits": {"use_predefined": hinted, "seed": 42, "test_max": max(2000, min(test_rows, 5000))},
            "baselines": [{"model": "gpt-6-luna", "reasoning_effort": "low"}],
            "train": {"provider": "together", "base_model": models[0]["id"],
                      "fallback_models": [m["id"] for m in models[1:2]], "lora": True, "epochs": 3,
                      "learning_rate": 1.0e-4},
            "eval": {"primary_metric": metric, "pass_bar": {"compare_to": "best_baseline", "max_gap_points": 2.0},
                     "concurrency": 16},
            "budget": {"max_spend_usd": 15, "max_attempts": 3},
        }  # fmt: skip
        self.draft = yaml.safe_dump(cfg, sort_keys=False)
        self.plan_rationale = {
            "task": f"one {self.label_col} label per {self.input_col}: classification",
            "label_source": "labels are already in the data",
            "base_model": f"{models[0]['id']} is the smallest deployable text model",
            "metric": f"{metric}: the most common label is {self.top_share:.0%} of rows",
            "split": "predefined files kept" if hinted else "val/test carved from the data",
            "baselines": "gpt-6-luna at low effort, the default baseline",
        }
        return self._call("validate_task", {"task_yaml": self.draft})
