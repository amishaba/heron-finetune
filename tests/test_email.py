"""Email cleaner (M5) without touching Gmail: parsing, behavior rules, redaction, a full dry run on a
synthetic export, and `ft email apply` against a fake Gmail service."""

import base64
import json
import random

import pytest
import yaml
from rich.console import Console

from ft.connectors import gmail
from ft.connectors.rules import RuleError, Rules, parse
from ft.context import REPO_ROOT, RunContext
from ft.pipeline import run_pipeline
from ft.providers.base import Prediction
from ft.redact import Redactor, sender_token

quiet = Console(quiet=True)
EMAIL_TASK = REPO_ROOT / "examples" / "email" / "task.yaml"


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    d = tmp_path / "ftconfig"
    monkeypatch.setenv("FT_CONFIG_DIR", str(d))
    return d


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


def api_message(i: int, *, sender, subject, body, labels, html=False, unsubscribe=False, thread=None, ms=None):
    headers = [
        {"name": "From", "value": sender},
        {"name": "To", "value": "me@example.org, team@example.org"},
        {"name": "Subject", "value": subject},
    ]
    if unsubscribe:
        headers.append({"name": "List-Unsubscribe", "value": "<mailto:unsub@shop.example>"})
    part = {"mimeType": "text/html" if html else "text/plain", "body": {"data": _b64(body)}}
    return {
        "id": f"m{i:04d}",
        "threadId": thread or f"t{i:04d}",
        "labelIds": labels,
        "snippet": body[:80],
        "internalDate": str(ms or 1_767_225_600_000 + i * 3_600_000),
        "payload": {"mimeType": "multipart/alternative", "headers": headers, "parts": [part]},
    }


# ---------------------------------------------------------------- parsing


def test_parse_message_features_and_body():
    m = api_message(
        1,
        sender='"Ana Silva" <Ana.Silva@acme.example>',
        subject="Q3 plan",
        body="<p>Hi,<br>see the <b>plan</b>.</p>",
        labels=["INBOX", "STARRED", "CATEGORY_PERSONAL"],
        html=True,
        unsubscribe=True,
    )
    r = gmail.parse_message(m)
    assert r["from_address"] == "ana.silva@acme.example" and r["from_name"] == "Ana Silva"
    assert r["n_recipients"] == 2 and r["category"] == "primary"
    assert r["features"] == {"opened": True, "starred": True, "archived": False, "has_unsubscribe": True}
    assert "plan" in r["body"] and "<b>" not in r["body"]  # HTML converted to text


def test_replied_after_needs_a_later_sent_message_in_the_thread():
    rows = [
        gmail.parse_message(
            api_message(i, sender="x@y.example", subject="s", body="b", labels=[], thread="T", ms=1000 * (i + 1))
        )
        for i in range(3)
    ]
    gmail.add_replied_after(rows, [("T", 2500), ("OTHER", 9_999_999)])
    assert [r["features"]["replied_after"] for r in rows] == [True, True, False]


# ---------------------------------------------------------------- rules


def test_rules_precedence_parentheses_and_first_match():
    f = {"opened": True, "starred": False, "replied_after": False, "archived": True, "has_unsubscribe": True}
    assert parse("opened AND NOT starred OR replied_after").eval(f) is True
    assert parse("opened AND (starred OR replied_after)").eval(f) is False
    assert parse("NOT NOT opened").eval(f) is True
    rules = Rules(
        [{"important": "replied_after OR starred"}, {"read": "opened"}, {"skip": "NOT opened"}], gmail.FEATURES
    )
    assert rules.label(f) == "read" and rules.label({**f, "starred": True}) == "important"
    assert rules.label({**f, "opened": False}) == "skip"


@pytest.mark.parametrize("bad", ["opened AND", "(opened", "opened)", "AND opened", "opened; import os", "__import__"])
def test_malformed_or_unknown_rules_fail_early(bad):
    with pytest.raises(RuleError):
        Rules([{"x": bad}], gmail.FEATURES)


# ---------------------------------------------------------------- redaction


def test_redaction_and_sender_tokens(config_dir):
    r = Redactor()
    text = r.redact("Call Maria Lopez at +1 415 555 0134 or maria.lopez@gmail.com about the Lisbon office.")
    assert "Maria" not in text and "555" not in text and "maria.lopez@" not in text
    assert "<PERSON>" in text and "<PHONE_NUMBER>" in text and "<EMAIL_ADDRESS>" in text
    tok, domain = sender_token("Maria.Lopez@Gmail.com")
    assert domain == "gmail.com" and tok.startswith("sender_") and tok == sender_token("maria.lopez@gmail.com")[0]
    assert (config_dir / "salt").exists()


def test_render_matches_spec_format(config_dir):
    row = gmail.parse_message(
        api_message(
            2,
            sender="Bob <bob@shop.example>",
            subject="Sale for Bob Jones",
            body="Hello Bob Jones, 50% off today.",
            labels=["CATEGORY_PROMOTIONS"],
            unsubscribe=True,
        )
    )
    text = gmail.render(row, Redactor())
    lines = text.splitlines()
    assert lines[0].startswith("From: sender_") and lines[0].endswith("(shop.example)")
    assert lines[1:4] == ["Recipients: 2", "Category: promotions", "Unsubscribe link: yes"]
    assert lines[4].startswith("Subject: ") and "Bob Jones" not in text and "bob@" not in text


# ---------------------------------------------------------------- end to end (dry run)

PEOPLE = ["Ana Silva", "Tom Becker", "Priya Nair", "Lena Park"]
SHOPS = ["deals@shop.example", "news@media.example", "noreply@social.example"]


def synthetic_export(path, n=360, seed=3):
    """A year of fake mail: colleagues (often replied to), receipts (read), promotions (mostly unopened)."""
    rng = random.Random(seed)
    msgs, sent = [], []
    for i in range(n):
        kind = rng.choices(["colleague", "receipt", "promo"], [0.2, 0.3, 0.5])[0]
        if kind == "colleague":
            who = rng.choice(PEOPLE)
            m = api_message(
                i,
                sender=f"{who} <{who.split()[0].lower()}@work.example>",
                subject=f"Re: project {rng.randint(1, 40)}",
                body=f"Hi, {who} here. Can you review the doc before Friday? Call me at +1 415 555 01{i % 90:02d}.",
                labels=["INBOX", "CATEGORY_PERSONAL"] + (["STARRED"] if rng.random() < 0.2 else []),
            )
            if rng.random() < 0.6:
                sent.append((m["threadId"], int(m["internalDate"]) + 600_000))
        elif kind == "receipt":
            m = api_message(
                i,
                sender="Store <orders@store.example>",
                subject=f"Your order #{1000 + i}",
                body=f"Thanks for your order #{1000 + i}. Total ${rng.randint(5, 90)}.",
                labels=["CATEGORY_UPDATES"] + ([] if rng.random() < 0.8 else ["UNREAD"]),
            )
        else:
            m = api_message(
                i,
                sender=f"Shop <{rng.choice(SHOPS)}>",
                subject=f"{rng.randint(10, 70)}% off this week",
                body="Huge sale on everything. Unsubscribe anytime.",
                unsubscribe=True,
                labels=["CATEGORY_PROMOTIONS"] + (["UNREAD"] if rng.random() < 0.85 else []),
            )
        msgs.append(m)
    rows = [gmail.parse_message(m) for m in msgs]
    gmail.add_replied_after(rows, sent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def _email_task(tmp_path, export):
    raw = yaml.safe_load(EMAIL_TASK.read_text())
    raw["source"]["export"] = str(export)
    raw["splits"]["test_max"] = 100
    p = tmp_path / "email.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_email_dry_run_end_to_end(tmp_path):
    export = tmp_path / "export" / "messages.jsonl"
    synthetic_export(export)
    ctx = RunContext.create(_email_task(tmp_path, export), dry_run=True, yes=True, run_id="email")
    run_pipeline(ctx, quiet)

    assert ctx.labels() == ["important", "read", "skip"]
    meta = json.loads((ctx.data_dir / "splits.json").read_text())
    assert meta["method"] == "by_time"
    dates = {
        s: [json.loads(line)["meta"]["date"] for line in (ctx.data_dir / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "val", "test")
    }
    assert max(dates["train"]) <= min(dates["val"]) and max(dates["val"]) <= min(dates["test"])

    # nothing identifying in what would be uploaded
    upload = (ctx.attempt_dir(1) / "train.together.jsonl").read_text()
    for name in PEOPLE:
        assert name not in upload and name.split()[0].lower() + "@" not in upload
    assert "555" not in upload and "sender_" in upload

    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert report["primary_metric"] == "macro_f1"
    assert report["privacy"]["redaction"]["entities_replaced"].get("PERSON", 0) > 0
    assert any(s.startswith("Together: training file") for s in report["privacy"]["sent"])
    assert "what left this machine" in (ctx.run_dir / "report.html").read_text()


# ---------------------------------------------------------------- apply


class FakeGmail:
    """Just enough of the Gmail API for apply; records every modification."""

    def __init__(self, messages):
        self._messages = {m["id"]: m for m in messages}
        self.label_list = [{"id": "INBOX", "name": "INBOX"}]
        self.modifications = []

    def users(self):
        return self

    def labels(self):
        return self

    def messages(self):
        return self

    def list(self, userId, **kw):
        if "q" not in kw:  # labels().list()
            return _Exec({"labels": list(self.label_list)})
        return _Exec({"messages": [{"id": i} for i in self._messages]})

    def create(self, userId, body):
        new = {"id": f"Label_{len(self.label_list)}", "name": body["name"]}
        self.label_list.append(new)
        return _Exec(new)

    def get(self, userId, id, format):
        return _Req(self._messages[id])

    def batchModify(self, userId, body):
        self.modifications.append(body)
        return _Exec({})

    def new_batch_http_request(self, callback):
        return _Batch(callback)


class _Exec:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _Req(_Exec):
    uri = "https://gmail.example/get?format=full"


class _Batch:
    def __init__(self, cb):
        self.cb, self.reqs = cb, []

    def add(self, req, request_id):
        self.reqs.append((request_id, req))

    def execute(self):
        for rid, req in self.reqs:
            self.cb(rid, req.execute(), None)


def test_apply_only_adds_labels_and_tears_down(tmp_path, config_dir, monkeypatch):
    from ft.email_apply import apply, load_checkpoint

    export = tmp_path / "export" / "messages.jsonl"
    synthetic_export(export)
    ctx = RunContext.create(_email_task(tmp_path, export), dry_run=True, yes=True, run_id="email")
    run_pipeline(ctx, quiet)

    new_mail = [
        api_message(
            900 + i,
            sender="Shop <deals@shop.example>",
            subject="Flash sale",
            body="Buy now",
            labels=["INBOX", "UNREAD", "CATEGORY_PROMOTIONS"],
            ms=1_900_000_000_000 + i,
        )
        for i in range(4)
    ]
    service = FakeGmail(new_mail)

    async def fake_predict(endpoint, items, prompt, concurrency):
        return [
            Prediction(id=it.id, prediction_raw="skip" if i % 2 else "important", latency_ms=5.0)
            for i, it in enumerate(items)
        ]

    monkeypatch.setattr(ctx.providers.tuned, "predict", fake_predict)
    result = apply(ctx, service=service, since_ms=1_800_000_000_000, log=lambda m: None)

    assert result["messages"] == 4 and result["predictions"] == {"important": 2, "skip": 2}
    assert {lbl["name"] for lbl in service.label_list} >= {"ft/Probably skip", "ft/Important"}
    for mod in service.modifications:  # only ever adds labels
        assert set(mod) == {"ids", "addLabelIds"}
    assert sum(len(m["ids"]) for m in service.modifications) == 4
    assert load_checkpoint("email") == 1_900_000_000_003
    assert ctx.state.live_endpoints() == {}  # endpoint torn down
    log = (config_dir / "email_apply_log.jsonl").read_text()
    assert "Buy now" not in log and "Flash sale" not in log  # ids and counts only
