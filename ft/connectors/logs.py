"""Logs mode (spec 15): read logged model calls and normalize them to one shape.

One JSON object per line. The native shape is OpenAI-compatible, so a future proxy can write it directly:

    {"id": "...", "timestamp": "...",
     "request": {"model": "...", "messages": [{"role": "system", "content": "..."}, {"role": "user", ...}]},
     "response": {"content": "card_arrival", "usage": {"input_tokens": 0, "output_tokens": 0}},
     "latency_ms": 412, "metadata": {}}

Also accepted: a raw OpenAI chat-completions response (`choices[0].message.content`, `prompt_tokens` /
`completion_tokens`), and Anthropic Messages shapes (`request.system` as a string or text blocks,
content as text blocks; `response.content` as a list of blocks).

Each log becomes a flat row with the fields a task can name: `last_user_message`, `system_prompt`,
`conversation` (every message as "role: text"), `response_content`, plus `logged_model`, `usage`,
`latency_ms`, `timestamp`. A truth sidecar (`{"id", "truth"|"label"|"target"}` per line) adds `truth`,
used only for evaluation, never as a training target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIELDS = ("last_user_message", "system_prompt", "conversation", "response_content")
TRUTH_KEYS = ("truth", "label", "target")


class LogsError(ValueError):
    pass


def _text(content: Any) -> str:
    """A message's content as text: a string, or a list of text blocks/parts (OpenAI or Anthropic)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)


def normalize(log: dict[str, Any], line_no: int = 0) -> dict[str, Any]:
    req = log.get("request") or {}
    resp = log.get("response") or {}
    messages = list(req.get("messages") or [])
    system = _text(req.get("system")) if req.get("system") is not None else ""
    if not system:
        system = "\n\n".join(_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer"))
    chat = [m for m in messages if m.get("role") not in ("system", "developer")]
    users = [m for m in chat if m.get("role") == "user"]
    if not users:
        raise LogsError(f"line {line_no}: no user message in request.messages")

    if "choices" in resp:  # raw OpenAI chat-completions response
        choice = (resp.get("choices") or [{}])[0]
        content = _text((choice.get("message") or {}).get("content"))
    else:
        content = _text(resp.get("content"))
    u = resp.get("usage") or {}
    usage = {
        "input_tokens": int(u.get("input_tokens", u.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(u.get("output_tokens", u.get("completion_tokens", 0)) or 0),
        "cache_read_tokens": int(u.get("cache_read_input_tokens", (u.get("prompt_tokens_details") or {}).get(
            "cached_tokens", u.get("cache_read_tokens", 0))) or 0),
    }  # fmt: skip
    last_user = _text(users[-1].get("content"))
    rid = str(log.get("id") or hashlib.sha1(f"{line_no}:{last_user}".encode()).hexdigest()[:12])
    return {
        "id": rid,
        "timestamp": log.get("timestamp"),
        "last_user_message": last_user,
        "system_prompt": system,
        "conversation": "\n".join(f"{m.get('role')}: {_text(m.get('content'))}" for m in chat),
        "response_content": content.strip(),
        "logged_model": req.get("model") or resp.get("model"),
        "usage": usage,
        "latency_ms": log.get("latency_ms"),
        "metadata": log.get("metadata") or {},
    }


def read_truth(path: Path) -> dict[str, str]:
    truth = {}
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        key = next((k for k in TRUTH_KEYS if k in obj), None)
        if "id" not in obj or key is None:
            raise LogsError(f"{path.name} line {i}: need 'id' and one of {TRUTH_KEYS}")
        truth[str(obj["id"])] = str(obj[key])
    return truth


def read_logs(path: Path, truth_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalized rows, plus stats (rows, skipped lines, models seen, truth coverage)."""
    rows, skipped = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(normalize(json.loads(line), i))
        except (json.JSONDecodeError, LogsError, AttributeError, TypeError) as e:
            skipped.append(f"line {i}: {e}"[:200])
    truth = read_truth(truth_path) if truth_path else {}
    for r in rows:
        if r["id"] in truth:
            r["truth"] = truth[r["id"]]
    models: dict[str, int] = {}
    for r in rows:
        models[str(r["logged_model"])] = models.get(str(r["logged_model"]), 0) + 1
    stats = {"rows": len(rows), "skipped": len(skipped), "skipped_examples": skipped[:5], "models": models,
             "with_truth": sum(1 for r in rows if "truth" in r)}  # fmt: skip
    return rows, stats
