"""models.yaml: base models the pipeline may train and deploy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BaseModelInfo:
    id: str
    kind: str
    size_b: float
    deployable: bool
    endpoint_config: str | None  # hardware name, used for the hourly price
    deploy_config: str | None = None  # config revision (cr_...) to pass when several exist
    # Verified that a deployed copy reads image inputs. Being vision-*tunable* isn't enough: on Together,
    # dedicated Qwen3.5-4B endpoints receive images but see noise (NOTES.md, CORD, 2026-10-17).
    serves_images: bool = False
    provider: str = "together"  # who trains and serves it (task.yaml train.provider must match)
    tokenizer_model: str | None = None  # Fireworks: overrides the tokenizer its recipe picks for the model
    renderer: str | None = None  # Fireworks: overrides the chat renderer the recipe infers
    notes: str = ""

    def accepts(self, input_kind: str) -> bool:
        return input_kind == "text" or (input_kind == "image" and self.serves_images)


def load_models(path: Path) -> dict[str, BaseModelInfo]:
    rows = yaml.safe_load(path.read_text())["models"]
    return {r["id"]: BaseModelInfo(**r) for r in rows}
