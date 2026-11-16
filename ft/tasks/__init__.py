from __future__ import annotations

from ft.config import TaskConfig
from ft.tasks.base import TaskType


def get_task_type(cfg: TaskConfig, labels: list[str] | None) -> TaskType:
    if cfg.task.type == "classify":
        from ft.tasks.classify import ClassifyTask

        return ClassifyTask(cfg, labels)
    from ft.tasks.extract import ExtractTask

    return ExtractTask(cfg, labels)
