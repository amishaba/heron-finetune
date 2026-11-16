"""serve: deploy an endpoint, wait until ready, and ALWAYS tear it down (spec 9.8).

Endpoints bill per minute while idle, so a leaked endpoint is a serious bug. Guarantees:
  - endpoint IDs go into state.json the moment `create` returns;
  - teardown runs in `finally` (covers exceptions and Ctrl-C);
  - SIGTERM is turned into SystemExit so `finally` runs for it too;
  - an atexit hook tears down anything still registered;
  - teardown is idempotent; if it fails, the endpoint stays "live" in state.json for `ft teardown`.
"""

from __future__ import annotations

import atexit
import contextlib
import re
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from rich.console import Console

from ft.events import now_iso
from ft.io import write_json
from ft.providers.base import Endpoint, ModelRef
from ft.spend import ENDPOINT_PROVISION_MINUTES, estimate_endpoint_usd

NAME = "serve"
READY_TIMEOUT_S = 20 * 60

_live: dict[str, tuple[object, Endpoint, object]] = {}  # endpoint id → (ctx, endpoint, attempt)
_hooks_installed = False
_err = Console(stderr=True)


def endpoint_name(run_id: str, attempt: int) -> str:
    """Lowercase letters, digits and hyphens; unique per attempt; '-a<n>' suffix always kept."""
    suffix = f"-a{attempt}"
    base = re.sub(r"[^a-z0-9-]+", "-", f"ft-{run_id}".lower()).strip("-")
    return base[: 60 - len(suffix)].rstrip("-") + suffix


def _install_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    atexit.register(teardown_all_live)

    def _on_sigterm(signum, frame):
        raise SystemExit(128 + signum)  # unwinds the stack so `finally` blocks tear down

    with contextlib.suppress(ValueError):  # not in the main thread
        signal.signal(signal.SIGTERM, _on_sigterm)
    _hooks_installed = True


def teardown_all_live() -> None:
    for eid in list(_live):
        ctx, ep, attempt = _live[eid]
        teardown(ctx, ep, attempt)


def hourly_rate(ctx, base_model: str, endpoint: Endpoint | None = None) -> float:
    info = ctx.models.get(base_model)
    config = (info.endpoint_config if info else None) or (endpoint.hardware_config if endpoint else None)
    return ctx.prices.endpoint_per_hour(config)


def find_by_name(ctx, name: str) -> Endpoint | None:
    try:
        for ep in ctx.providers.server.list_ft_endpoints():
            if ep.name == name:
                ctx.state.record_endpoint(name, id=ep.id, model=ep.model)
                return ep
    except Exception as e:  # noqa: BLE001
        ctx.events.emit("error", stage=NAME, message=f"couldn't list endpoints while cleaning up {name}: {e}")
    return None


def teardown(ctx, ep: Endpoint, attempt: int) -> None:
    """Idempotent. Records time billed and cost. Never raises (it runs inside `finally`)."""
    rec = ctx.state.data["endpoints"].get(ep.name, {})
    if rec.get("torn_down_at"):
        _live.pop(ep.id, None)
        return
    try:
        ctx.providers.server.teardown(ep.id)
    except Exception as e:  # noqa: BLE001
        ctx.events.emit("error", stage=NAME, attempt=attempt, message=f"TEARDOWN FAILED for {ep.id}: {e}")
        _err.print(
            f"[bold red]TEARDOWN FAILED for endpoint {ep.id} ({ep.name}): {e}\n"
            f"It may still be billing. Run `ft teardown {ctx.run_id}` now.[/]"
        )
        return
    now = datetime.now(UTC)
    requested = datetime.fromisoformat(rec["requested_at"])
    minutes = (now - requested).total_seconds() / 60
    if ep.simulated:
        minutes += ENDPOINT_PROVISION_MINUTES  # fakes are instant; bill as if provisioning took the usual time
    rate = rec.get("hourly_rate_usd") or 0.0
    usd = minutes * rate / 60
    ctx.state.record_endpoint(
        ep.name, torn_down_at=now.isoformat(timespec="seconds"), minutes_billed=round(minutes, 2), usd=round(usd, 4)
    )
    ctx.spend.record(NAME, ctx.cfg.train.provider, usd, f"endpoint {ep.name}: {minutes:.1f} min", attempt=attempt)
    ctx.events.emit("endpoint_teardown", stage=NAME, attempt=attempt, endpoint_id=ep.id, minutes=round(minutes, 2))
    write_json(ctx.attempt_dir(attempt) / "endpoint.json", ctx.state.data["endpoints"][ep.name])
    _live.pop(ep.id, None)


@contextmanager
def deployed(ctx, model: ModelRef, attempt: int, eval_minutes: float) -> Iterator[Endpoint]:
    _install_hooks()
    server = ctx.providers.server
    name = endpoint_name(ctx.run_id, attempt)
    rate = hourly_rate(ctx, model.base_model)
    ctx.spend.check(
        NAME,
        ctx.cfg.train.provider,
        estimate_endpoint_usd(eval_minutes, rate),
        f"endpoint {name} for ~{ENDPOINT_PROVISION_MINUTES + eval_minutes:.0f} min at ${rate:.2f}/h",
        attempt=attempt,
    )
    ep: Endpoint | None = None
    try:
        requested_at = now_iso()
        old = ctx.state.data["endpoints"].get(name)
        if old and old.get("torn_down_at"):  # re-deploy after a failed predict: keep the history
            ctx.state.data["endpoints"][f"{name}@{old.get('requested_at')}"] = ctx.state.data["endpoints"].pop(name)
        elif old:
            raise RuntimeError(f"endpoint {name} is still recorded as live; run `ft teardown {ctx.run_id}` first")
        # Saved before the call, so `ft teardown` can find the endpoint by name if we die mid-create.
        ctx.state.record_endpoint(name, attempt=attempt, requested_at=requested_at, hourly_rate_usd=rate)
        try:
            ep = server.create(model, name)
        except BaseException:
            ep = find_by_name(ctx, name)  # created but not reported back (timeout, crash in the CLI)
            raise
        ctx.state.record_endpoint(
            name,
            id=ep.id,
            deployment_id=ep.deployment_id,
            model=ep.model,
            attempt=attempt,
            requested_at=requested_at,
            hourly_rate_usd=rate,
            hardware_config=ep.hardware_config,
            simulated=ep.simulated,
        )
        _live[ep.id] = (ctx, ep, attempt)
        ctx.events.emit("endpoint_deploy", stage=NAME, attempt=attempt, endpoint_id=ep.id, name=name)
        deadline = time.monotonic() + READY_TIMEOUT_S
        while not server.is_ready(ep):
            if time.monotonic() > deadline:
                raise TimeoutError(f"endpoint {ep.id} not ready after {READY_TIMEOUT_S // 60} min")
            time.sleep(server.poll_seconds)
        ctx.state.record_endpoint(name, ready_at=now_iso())
        yield ep
    finally:
        if ep is not None:
            teardown(ctx, ep, attempt)
