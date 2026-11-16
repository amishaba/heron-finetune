"""Provider plumbing that doesn't need the network: retries, failure gate, think-stripping,
endpoint cleanup when create fails midway, and parsing the Together CLI's JSON."""

import httpx
import pytest
from rich.console import Console

from ft.context import RunContext
from ft.pipeline import run_pipeline
from ft.providers import together_provider as tp
from ft.providers.base import Endpoint, ModelRef, Prediction
from ft.providers.fake import FakeServer
from ft.stages import TooManyFailures, run_predictions
from tests.conftest import TINY

quiet = Console(quiet=True)


class _Ctx:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


class _Item:
    def __init__(self, i):
        self.id = str(i)


def _flaky(fail_ids: set[str], error="RateLimitError: slow down", heal=True):
    calls = {"n": 0}

    async def call(part):
        calls["n"] += 1
        out = []
        for it in part:
            bad = it.id in fail_ids and (calls["n"] == 1 or not heal)
            out.append(
                Prediction(id=it.id, prediction_raw=None if bad else "x", latency_ms=1.0, error=error if bad else None)
            )
        return out

    return call, calls


def test_transient_failures_are_retried_once():
    items = [_Item(i) for i in range(100)]
    call, calls = _flaky({"3", "7"})
    preds = run_predictions(_Ctx(), call, items, "t")
    assert calls["n"] == 2 and all(p.error is None for p in preds)
    assert [p.id for p in preds] == [it.id for it in items]


def test_too_many_failures_stops_the_stage():
    items = [_Item(i) for i in range(100)]
    call, _ = _flaky({str(i) for i in range(5)}, heal=False)
    with pytest.raises(TooManyFailures, match="5/100") as exc:
        run_predictions(_Ctx(), call, items, "t")
    assert len(exc.value.predictions) == 100  # handed back so the caller can record spend


def test_model_outcomes_are_not_retried():
    items = [_Item(i) for i in range(10)]
    call, calls = _flaky({"1"}, error="refusal", heal=False)
    preds = run_predictions(_Ctx(), call, items, "t")
    assert calls["n"] == 1 and preds[1].error == "refusal"  # counts as wrong, not as an outage


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<think>\nhmm, cards...\n</think>\n\ncard_arrival", "card_arrival"),
        ("card_arrival", "card_arrival"),
        ("reasoning without an opening tag</think> lost_card", "lost_card"),
        (None, None),
    ],
)
def test_strip_thinking(raw, expected):
    assert tp.strip_thinking(raw) == expected


class _CrashingServer(FakeServer):
    """Creates the endpoint on the provider, then dies before returning its id."""

    def create(self, model, name):
        super().create(model, name)
        raise TimeoutError("CLI timed out after the endpoint was created")

    def list_ft_endpoints(self):
        return [
            Endpoint(id=f"endpoint-fake-{n}", name=n, model="m", simulated=True)
            for n in [d.removeprefix("endpoint-fake-") for d in self.deployed]
        ]


def test_endpoint_created_but_not_reported_is_found_and_torn_down():
    ctx = RunContext.create(TINY, dry_run=True, yes=True, run_id="t")
    ctx.providers.server = _CrashingServer(ctx.fake_opts)
    with pytest.raises(TimeoutError):
        run_pipeline(ctx, quiet)
    ((name, rec),) = ctx.state.data["endpoints"].items()
    assert rec["requested_at"] and rec["torn_down_at"]  # name recorded before create, then cleaned up
    assert ctx.providers.server.torn_down == ctx.providers.server.deployed


def _fake_cli(responses, calls):
    def fake_tg(*args, project=None, timeout=300):
        calls.append(args)
        for key, value in responses.items():
            if args[: len(key)] == key:
                out = value(args) if callable(value) else value
                if isinstance(out, Exception):
                    raise out
                return out
        raise AssertionError(f"unexpected tg call {args}")

    return fake_tg


DEPLOYED = {
    "endpoint": {"id": "ep_123", "name": "proj-slug/ft-run-a1"},
    "deployment": {"id": "dep_9", "hardware": {"name": "1xnvidia-h100-80gb"}},
}


def test_together_server_deploys_then_routes_traffic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tp,
        "tg",
        _fake_cli(
            {
                ("beta", "endpoints", "deploy"): DEPLOYED,
                ("beta", "endpoints", "get"): {"id": "ep_123", "trafficSplit": []},
                ("beta", "endpoints", "update"): {"ok": True},
            },
            calls,
        ),
    )
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    server = tp.TogetherServer({"Qwen/Qwen3.5-9B": "cr_9b"})
    ep = server.create(ModelRef(model_object_id="ml_abc", base_model="Qwen/Qwen3.5-9B"), "ft-run-a1")
    assert (ep.id, ep.model, ep.deployment_id) == ("ep_123", "proj-slug/ft-run-a1", "dep_9")
    deploy = calls[0]
    assert deploy[:4] == ("beta", "endpoints", "deploy", "ml_abc")
    assert "--traffic-weight" not in deploy  # applying it inside deploy races on the etag (409)
    assert "--inactive-timeout" in deploy and deploy[deploy.index("--config") + 1] == "cr_9b"
    assert calls[-1] == ("beta", "endpoints", "update", "dep_9", "--traffic-weight", "1")


def test_routing_skipped_when_deployment_already_gets_traffic(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tp,
        "tg",
        _fake_cli(
            {
                ("beta", "endpoints", "deploy"): DEPLOYED,
                ("beta", "endpoints", "get"): {"trafficSplit": [{"deploymentId": "dep_9", "weight": 1}]},
            },
            calls,
        ),
    )
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    tp.TogetherServer().create(ModelRef(model_object_id="ml_abc", base_model="b"), "ft-run-a1")
    assert not any(c[:3] == ("beta", "endpoints", "update") for c in calls)


def test_conflicts_are_retried(monkeypatch):
    monkeypatch.setattr(tp, "CONFLICT_RETRY_DELAYS_S", (0, 0, 0))
    attempts = {"n": 0}

    def rm(args):
        attempts["n"] += 1
        return tp.TogetherError('{"message": "Conflict", "status": 409}') if attempts["n"] < 3 else {"ok": 1}

    monkeypatch.setattr(tp, "tg", _fake_cli({("beta", "endpoints", "rm"): rm}, []))
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    tp.TogetherServer().teardown("ep_1")
    assert attempts["n"] == 3


def test_persistent_conflict_still_raises(monkeypatch):
    monkeypatch.setattr(tp, "CONFLICT_RETRY_DELAYS_S", (0, 0))
    conflict = tp.TogetherError("Conflict 409")
    monkeypatch.setattr(tp, "tg", _fake_cli({("beta", "endpoints", "rm"): conflict}, []))
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    with pytest.raises(tp.TogetherError):
        tp.TogetherServer().teardown("ep_1")


def test_redeploy_in_same_attempt_keeps_history_and_tears_down_new_endpoint():
    from ft.providers.fake import FakeOptions
    from ft.stages import predict

    ctx = RunContext.create(
        TINY, dry_run=True, yes=True, run_id="t", fake_opts=FakeOptions(tuned_q_by_attempt={1: 1.0})
    )
    run_pipeline(ctx, quiet)
    ctx.attempt = 1
    predict.run(ctx)  # deploy again under the same name
    eps = ctx.state.data["endpoints"]
    assert len(eps) == 2 and all(e.get("torn_down_at") for e in eps.values())
    assert len(ctx.providers.server.deployed) == 2 and set(ctx.providers.server.deployed) <= set(
        ctx.providers.server.torn_down
    )


def test_together_server_retries_adapter_with_merge(monkeypatch):
    calls = []

    def fake_tg(*args, project=None, timeout=300):
        calls.append(args)
        if args[2] == "get":
            return {"trafficSplit": [{"deploymentId": "dep_1", "weight": 1}]}
        if "--merge" not in args:
            raise tp.TogetherError("ml_abc is a LoRA adapter.\n\nChoose how to deploy it")
        return {"endpoint": {"id": "ep_1", "name": "s/ft-x-a1"}, "deployment": {"id": "dep_1"}}

    monkeypatch.setattr(tp, "tg", fake_tg)
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    ep = tp.TogetherServer().create(ModelRef(model_object_id="ml_abc", base_model="b"), "ft-x-a1")
    assert ep.id == "ep_1" and "--merge" in calls[1]


def test_teardown_of_missing_endpoint_is_success(monkeypatch):
    def fake_tg(*args, project=None, timeout=300):
        raise tp.TogetherError("endpoint ep_1 not found (404)")

    monkeypatch.setattr(tp, "tg", fake_tg)
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    tp.TogetherServer().teardown("ep_1")  # idempotent: already gone is fine


def test_readiness_reads_deployment_state(monkeypatch):
    state = {"s": "DEPLOYMENT_STATE_PROVISIONING"}

    def fake_tg(*args, project=None, timeout=300):
        return {"id": "ep_1", "deployments": [{"state": state["s"], "readyReplicas": 1}]}

    monkeypatch.setattr(tp, "tg", fake_tg)
    monkeypatch.setenv("TOGETHER_PROJECT_ID", "proj_x")
    server = tp.TogetherServer()
    ep = Endpoint(id="ep_1", name="ft-x-a1", model="s/ft-x-a1")
    assert not server.is_ready(ep)
    state["s"] = "DEPLOYMENT_STATE_READY"
    assert server.is_ready(ep)
    state["s"] = "DEPLOYMENT_STATE_FAILED"
    with pytest.raises(tp.TogetherError, match="failed"):
        server.is_ready(ep)


def test_loss_curve_reads_train_loss_metric():
    class _Metrics:
        metrics = [{"train/loss": 2.6, "train/global_step": 1.0}, {"train/loss": 0.41, "train/global_step": 2.0}]

    class _FT:
        def list_metrics(self, job_id, resolution=100):
            return _Metrics()

    trainer = tp.TogetherTrainer.__new__(tp.TogetherTrainer)
    trainer.client = type("C", (), {"fine_tuning": _FT()})()
    assert trainer._loss_curve("ft-1") == [2.6, 0.41]


def test_token_bucket_throttles_to_rate():
    import asyncio
    import time

    from ft.providers.openai_provider import TokenBucket

    async def go():
        b = TokenBucket(per_minute=60_000, burst_seconds=10)  # 1,000 tokens/s, 10,000 burst
        t0 = time.monotonic()
        await b.take(10_000)  # the burst is free
        assert time.monotonic() - t0 < 0.05
        await b.take(400)  # then it waits for the refill
        return time.monotonic() - t0

    assert 0.35 < asyncio.run(go()) < 0.8


def test_logprobs_are_normalized_from_either_shape():
    from ft.providers.base import normalize_logprobs

    openai_shape = {"content": [{"token": "refund", "logprob": -0.01, "top_logprobs": [
        {"token": "refund", "logprob": -0.01}, {"token": "lost", "logprob": -4.7}]}]}  # fmt: skip
    together_shape = {"tokens": ["ref", "und"], "token_logprobs": [-0.02, -0.001],
                      "top_logprobs": [{"ref": -0.02, "lost": -4.1}, {"und": -0.001}]}  # fmt: skip
    a = normalize_logprobs(openai_shape)
    b = normalize_logprobs(together_shape)
    assert a == [{"t": "refund", "lp": -0.01, "top": {"refund": -0.01, "lost": -4.7}}]
    assert b[0] == {"t": "ref", "lp": -0.02, "top": {"ref": -0.02, "lost": -4.1}} and len(b) == 2
    assert normalize_logprobs(None) is None and normalize_logprobs({"content": None}) is None


def test_tuned_predictions_keep_logprobs(isolated_runs):
    from rich.console import Console

    from ft.context import RunContext
    from ft.io import read_jsonl
    from ft.pipeline import run_pipeline
    from tests.conftest import TINY

    ctx = RunContext.create(TINY, dry_run=True, yes=True, run_id="lp")
    run_pipeline(ctx, Console(quiet=True))
    rows = read_jsonl(ctx.attempt_dir(1) / "test.predictions.jsonl")
    assert all(r["logprobs"][0]["lp"] < 0 for r in rows)


def test_concurrent_logprobs_rejections_keep_thinking_disabled():
    """A burst of rejected logprobs requests must not also switch thinking back on (seen live)."""
    import asyncio

    from openai import BadRequestError

    from ft.providers.base import Endpoint, Item, PromptSpec
    from ft.providers.together_provider import TogetherTuned

    sent = []

    class FakeCompletions:
        async def create(self, extra_body=None, **kw):
            sent.append(dict(extra_body or {}))
            await asyncio.sleep(0)
            if "logprobs" in (extra_body or {}):
                req = httpx.Request("POST", "http://x")
                raise BadRequestError("logprobs not supported", response=httpx.Response(400, request=req), body=None)

            class R:
                usage = None
                choices = [type("C", (), {"message": type("M", (), {"content": "refund"})(), "logprobs": None})()]

            return R()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    tuned = TogetherTuned()
    ep = Endpoint(id="e", name="n", model="m")

    async def burst():
        return await asyncio.gather(
            *(
                tuned._one(FakeClient(), ep, Item(id=str(i), input="x", target="refund"), PromptSpec(system="s"))
                for i in range(16)
            )  # fmt: skip
        )

    preds = asyncio.run(burst())
    assert all(p.prediction_raw == "refund" for p in preds)
    assert tuned.disable_thinking and tuned.logprobs_k is None
    assert all(b.get("chat_template_kwargs") == {"enable_thinking": False} for b in sent)
    assert len(tuned.notes) == 1 and "logprobs rejected" in tuned.notes[0]
