"""OpenAI frontier provider (baselines and the planning agent).

Uses the Responses API with strict json_schema structured output. Facts checked 2026-10-16:
- reasoning effort: none | low | medium | high | xhigh | max; `max_output_tokens` includes reasoning tokens,
  and `status == "incomplete"` means the budget ran out (we mark that prediction invalid);
- a refusal item may not follow the schema (invalid);
- prompt caching is automatic at >= 1,024 tokens; cached tokens are in usage.input_tokens_details.
Retries: the SDK retries 429 / 5xx / connection errors with exponential backoff and honours retry-after.
Throttling: OpenAI counts input + max_output_tokens against tokens-per-minute (cached input included), so
long few-shot prompts at full concurrency hit 429s faster than retries recover. A client-side token
bucket keeps requests under FT_OPENAI_TPM (default 2M, the Build tier; we use 80% of it).
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from openai import APIError, AsyncOpenAI

from ft.images import data_url
from ft.providers.base import AgentTurn, BaselineSettings, Item, Prediction, PromptSpec, ToolCall, Usage

# Extra output budget for reasoning tokens on top of the answer itself (we pay only for what's used).
REASONING_BUDGET = {None: 4096, "none": 0, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384, "max": 32768}


IMAGE_TOKENS_ESTIMATE = 2500  # for throttling only: the "high"/"auto" detail patch budget


def user_input(item: Item) -> str | list[dict]:
    """Text as a plain string; an image as an input_image part (base64 data URL), plus any text."""
    if not item.image:
        return item.input
    content: list[dict] = [{"type": "input_image", "image_url": data_url(item.image)}]
    if item.input:
        content.append({"type": "input_text", "text": item.input})
    return [{"role": "user", "content": content}]


class TokenBucket:
    """Async token bucket: `take(n)` waits until n tokens fit under the per-minute rate."""

    def __init__(self, per_minute: float, burst_seconds: float = 10.0):
        self.rate = per_minute / 60
        self.capacity = self.rate * burst_seconds
        self.tokens = self.capacity
        self.t = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, n: float) -> None:
        n = min(n, self.capacity)
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                await asyncio.sleep((n - self.tokens) / self.rate)


def tpm_limit() -> float:
    return float(os.environ.get("FT_OPENAI_TPM", 2_000_000)) * 0.8


def counted_tokens(prompt: PromptSpec, item: Item, reasoning_effort: str | None) -> int:
    """What a request counts against TPM: ~4 chars/token of input, plus the whole output allowance."""
    return (
        (len(prompt.system) + len(item.input)) // 4
        + prompt.max_output_tokens
        + REASONING_BUDGET.get(reasoning_effort, 4096)
    )


class OpenAIFrontier:
    def __init__(self, max_retries: int = 6, timeout_s: float = 120.0):
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    async def _one(self, client: AsyncOpenAI, item: Item, prompt: PromptSpec, settings: BaselineSettings) -> Prediction:
        kwargs: dict = {
            "model": settings.model,
            "instructions": prompt.system,
            "input": user_input(item),
            "max_output_tokens": prompt.max_output_tokens + REASONING_BUDGET.get(settings.reasoning_effort, 4096),
            "store": False,
        }
        if settings.cache_key:
            kwargs["prompt_cache_key"] = settings.cache_key
        if settings.reasoning_effort:
            kwargs["reasoning"] = {"effort": settings.reasoning_effort}
        if prompt.json_schema:
            kwargs["text"] = {
                "format": {"type": "json_schema", "name": "answer", "strict": True, "schema": prompt.json_schema}
            }
        t0 = time.monotonic()
        try:
            resp = await client.responses.create(**kwargs)
        except APIError as e:
            return Prediction(
                id=item.id,
                prediction_raw=None,
                latency_ms=(time.monotonic() - t0) * 1000,
                error=f"{type(e).__name__}: {e}"[:500],
            )
        latency = (time.monotonic() - t0) * 1000
        u = resp.usage
        in_details = getattr(u, "input_tokens_details", None) if u else None
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            cache_read_tokens=getattr(in_details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(in_details, "cache_write_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
        )
        error = None
        raw: str | None = resp.output_text or None
        if resp.status == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
            error, raw = f"incomplete: {reason}", None
        elif any(getattr(c, "type", None) == "refusal" for o in resp.output for c in getattr(o, "content", None) or []):
            error, raw = "refusal", None
        return Prediction(id=item.id, prediction_raw=raw, latency_ms=latency, usage=usage, error=error)

    async def predict(self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings) -> list[Prediction]:
        sem = asyncio.Semaphore(settings.concurrency)
        bucket = TokenBucket(tpm_limit())
        # A fresh client per call: each asyncio.run() has its own event loop, and pooled connections
        # can't cross loops.
        async with AsyncOpenAI(max_retries=self.max_retries, timeout=self.timeout_s) as client:

            async def guarded(it: Item) -> Prediction:
                async with sem:
                    await bucket.take(counted_tokens(prompt, it, settings.reasoning_effort))
                    return await self._one(client, it, prompt, settings)

            return list(await asyncio.gather(*(guarded(it) for it in items)))

    async def batch_predict(self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings):
        raise NotImplementedError("Batch API calls arrive with frontier labeling (M3/M4)")


class OpenAIAgent:
    """The planning agent's model (spec 12). Responses API with
    function tools, store=False: every turn replays the conversation, including the model's output items
    (encrypted reasoning, function calls) untouched, as the reasoning guide asks for tool loops."""

    simulated = False

    def __init__(self, model: str = "gpt-6-luna", effort: str = "medium", max_retries: int = 6, timeout_s: float = 300):
        from openai import OpenAI

        self.model = model
        self.effort = effort
        self.client = OpenAI(max_retries=max_retries, timeout=timeout_s)
        self.items: list = []

    def start(self, system: str, user: str, tools: list[dict]) -> None:
        self.system = system
        self.tools = [{"type": "function", "strict": True, **t} for t in tools]
        self.items = [{"role": "user", "content": user}]

    def step(self) -> AgentTurn:
        resp = self.client.responses.create(
            model=self.model,
            instructions=self.system,
            input=self.items,
            tools=self.tools,
            reasoning={"effort": self.effort},
            store=False,
            include=["reasoning.encrypted_content"],
        )
        self.items += [o.model_dump(exclude_none=True) for o in resp.output]
        calls = [
            ToolCall(o.call_id, o.name, json.loads(o.arguments or "{}"))
            for o in resp.output
            if o.type == "function_call"
        ]
        u = resp.usage
        details = getattr(u, "input_tokens_details", None)
        return AgentTurn(
            calls=calls,
            text=resp.output_text or "",
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
        )

    def add_results(self, results: list[tuple[str, str | list[str]]]) -> None:
        for call_id, out in results:
            if isinstance(out, list):  # view_images: image parts in the tool output
                out = [{"type": "input_image", "image_url": data_url(p, max_side=1024)} for p in out]
            self.items.append({"type": "function_call_output", "call_id": call_id, "output": out})

    def nudge(self, text: str) -> None:
        self.items.append({"role": "user", "content": text})
