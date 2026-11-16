"""Gmail connector (spec 14.2-14.4, 14.6).

`ft email pull` (gmail.readonly) exports the last N months to data/email/messages.jsonl. The pipeline's
load stage then renders, redacts and labels that local export, so runs never touch Gmail themselves.
`ft email apply` (gmail.modify, requested separately) only ever ADDS labels. It never archives,
deletes or marks anything read.

Behavior features per message:
  opened         lacks the UNREAD label
  starred        has STARRED
  replied_after  the thread has a message from the user (SENT) dated after this one
  archived       not in INBOX
  has_unsubscribe  a List-Unsubscribe header exists
"""

from __future__ import annotations

import base64
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import Any

from ft.redact import Redactor, config_dir, sender_token

SCOPE_READ = "https://www.googleapis.com/auth/gmail.readonly"
SCOPE_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
FEATURES = {"opened", "starred", "replied_after", "archived", "has_unsubscribe"}
CATEGORIES = {
    "CATEGORY_PERSONAL": "primary",
    "CATEGORY_SOCIAL": "social",
    "CATEGORY_PROMOTIONS": "promotions",
    "CATEGORY_UPDATES": "updates",
    "CATEGORY_FORUMS": "forums",
}
BODY_CHARS = 2000
RENDER_WORDS = 300
APPLY_LABELS = {"skip": "ft/Probably skip", "important": "ft/Important"}
MESSAGE_FIELDS = (
    "id,threadId,labelIds,snippet,internalDate,"
    "payload(headers,mimeType,body/data,parts(mimeType,body/data,parts(mimeType,body/data,parts(mimeType,body/data))))"
)

SETUP_HELP = """\
Gmail access needs an OAuth client (one-time, ~5 minutes):
  1. https://console.cloud.google.com/ -> create a project (e.g. "ft-email").
  2. APIs & Services -> Library -> "Gmail API" -> Enable.
  3. APIs & Services -> OAuth consent screen -> External -> fill app name + your email;
     add yourself under "Test users".
  4. APIs & Services -> Credentials -> Create credentials -> OAuth client ID -> "Desktop app".
  5. Download the JSON and save it as: {path}
Then run `ft email pull` again; a browser window asks you to allow read-only access."""


class GmailError(RuntimeError):
    pass


# ---------------------------------------------------------------- auth


def credentials_path() -> Path:
    return config_dir() / "credentials.json"


def get_service(scope: str):
    """OAuth desktop flow; one token file per scope, so modify access is granted separately."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token = config_dir() / ("gmail_token.json" if scope == SCOPE_READ else "gmail_modify_token.json")
    creds = Credentials.from_authorized_user_file(str(token), [scope]) if token.exists() else None
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not credentials_path().exists():
            raise GmailError(SETUP_HELP.format(path=credentials_path()))
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path()), [scope])
        creds = flow.run_local_server(port=0, open_browser=True)
    token.write_text(creds.to_json())
    token.chmod(0o600)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------- parsing (pure)


def _decode(data: str | None) -> str:
    if not data:
        return ""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _walk(part: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield part
    for p in part.get("parts") or []:
        yield from _walk(p)


def extract_body(payload: dict[str, Any]) -> str:
    """Plain text if present, else HTML converted to text; first BODY_CHARS characters."""
    parts = list(_walk(payload))
    plain = [_decode(p.get("body", {}).get("data")) for p in parts if p.get("mimeType") == "text/plain"]
    text = next((t for t in plain if t.strip()), "")
    if not text:
        html = next((_decode(p.get("body", {}).get("data")) for p in parts if p.get("mimeType") == "text/html"), "")
        if html:
            import html2text

            h = html2text.HTML2Text()
            h.ignore_links, h.ignore_images, h.body_width = True, True, 0
            text = h.handle(html)
    from ft.connectors.mailfiles import strip_header_blocks

    text, _removed = strip_header_blocks(text)
    return " ".join(text.split())[:BODY_CHARS]


def parse_message(msg: dict[str, Any]) -> dict[str, Any]:
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    name, address = parseaddr(headers.get("from", ""))
    recipients = getaddresses([headers.get(k, "") for k in ("to", "cc")])
    labels = set(msg.get("labelIds") or [])
    internal_ms = int(msg.get("internalDate", 0))
    category = next((v for k, v in CATEGORIES.items() if k in labels), "primary")
    return {
        "id": msg["id"],
        "thread_id": msg.get("threadId"),
        "internal_ms": internal_ms,
        "date": datetime.fromtimestamp(internal_ms / 1000, UTC).isoformat(timespec="seconds"),
        "from_name": name,
        "from_address": address.lower(),
        "n_recipients": len([a for _, a in recipients if a]),
        "subject": headers.get("subject", ""),
        "snippet": msg.get("snippet", ""),
        "body": extract_body(msg.get("payload", {})),
        "category": category,
        "label_ids": sorted(labels),
        "features": {
            "opened": "UNREAD" not in labels,
            "starred": "STARRED" in labels,
            "archived": "INBOX" not in labels,
            "has_unsubscribe": "list-unsubscribe" in headers,
        },
    }


def add_replied_after(rows: list[dict[str, Any]], sent: list[tuple[str, int]]) -> None:
    """replied_after: the user sent something in the same thread after this message arrived."""
    replies: dict[str, list[int]] = defaultdict(list)
    for thread_id, ms in sent:
        replies[thread_id].append(ms)
    for r in rows:
        r["features"]["replied_after"] = any(ms > r["internal_ms"] for ms in replies.get(r["thread_id"], []))


def render(row: dict[str, Any], redactor: Redactor) -> str:
    """The model's input (spec 14.3). Sender becomes a salted token; subject and body are redacted."""
    token, domain = sender_token(row["from_address"])
    words = row["body"].split()[:RENDER_WORDS]
    return "\n".join(
        [
            f"From: {token} ({domain})",
            f"Recipients: {row['n_recipients']}",
            f"Category: {row['category']}",
            f"Unsubscribe link: {'yes' if row['features']['has_unsubscribe'] else 'no'}",
            f"Subject: {redactor.redact(row['subject'])}",
            "",
            redactor.redact(" ".join(words)),
        ]
    )


# ---------------------------------------------------------------- API calls


def _list_ids(service, q: str) -> list[str]:
    ids, token = [], None
    while True:
        resp = service.users().messages().list(userId="me", q=q, maxResults=500, pageToken=token).execute()
        ids += [m["id"] for m in resp.get("messages", [])]
        token = resp.get("nextPageToken")
        if not token:
            return ids


def _batch_get(service, ids: list[str], fmt: str, fields: str | None, log: Callable[[str], None]) -> list[dict]:
    """messages.get in batches of 50, retrying rate-limited ones with backoff."""
    out: dict[str, dict] = {}
    pending = list(ids)
    delay = 2.0
    while pending:
        failed: list[str] = []
        for start in range(0, len(pending), 50):
            chunk = pending[start : start + 50]

            def cb(request_id, response, exception, _failed=failed):
                if exception is not None:
                    _failed.append(request_id)
                else:
                    out[request_id] = response

            batch = service.new_batch_http_request(callback=cb)
            for mid in chunk:
                req = service.users().messages().get(userId="me", id=mid, format=fmt)
                if fields:
                    req.uri += f"&fields={fields}"
                batch.add(req, request_id=mid)
            batch.execute()
            if len(ids) > 500 and (start // 50) % 10 == 0:
                log(f"fetched {len(out)}/{len(ids)}")
            time.sleep(0.25)  # stay under the per-user quota
        if failed:
            if delay > 64:
                raise GmailError(f"{len(failed)} messages kept failing (rate limits?)")
            log(f"retrying {len(failed)} rate-limited messages in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
        pending = failed
    return [out[i] for i in ids if i in out]


def pull(months: int, min_age_days: int, out_path: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    service = get_service(SCOPE_READ)
    now = time.time()
    after = int(now - months * 30.44 * 86400)
    before = int(now - min_age_days * 86400)
    log(f"listing messages from the last {months} months (skipping the last {min_age_days} days)")
    ids = _list_ids(service, f"after:{after} before:{before} -in:chats -in:drafts -in:sent")
    log(f"{len(ids)} messages; listing your sent mail for reply detection")
    sent_ids = _list_ids(service, f"in:sent after:{after}")
    sent = [(m["threadId"], int(m["internalDate"])) for m in _batch_get(service, sent_ids, "minimal", None, log)]
    rows = [parse_message(m) for m in _batch_get(service, ids, "full", MESSAGE_FIELDS, log)]
    add_replied_after(rows, sent)
    rows.sort(key=lambda r: r["internal_ms"])

    from ft.io import write_json, write_jsonl

    write_jsonl(out_path, rows)
    meta = {
        "pulled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "months": months,
        "min_age_days": min_age_days,
        "messages": len(rows),
        "sent_messages_seen": len(sent),
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    out_path.chmod(0o600)
    return meta


# ---------------------------------------------------------------- export → pipeline records


def load_export(path: Path, rules, redactor: Redactor) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Render + redact every exported message and label it with the behavior rules."""
    if not path.exists():
        raise GmailError(f"no Gmail export at {path}; run `ft email pull` first")
    from ft.io import read_jsonl

    records, dropped = [], {"no_rule_matched": 0}
    for row in read_jsonl(path):
        target = rules.label(row["features"])
        if target is None:
            dropped["no_rule_matched"] += 1
            continue
        records.append(
            {
                "id": row["id"],
                "input": render(row, redactor),
                "target": target,
                "meta": {
                    "date": row["date"],
                    "thread_id": row["thread_id"],
                    "category": row["category"],
                    **{f"f_{k}": v for k, v in row["features"].items()},
                },
            }
        )
    return records, dropped


# ---------------------------------------------------------------- apply (labels only)


def fetch_since(service, after_ms: int, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """New mail (not sent/drafts/chats) received after `after_ms`."""
    ids = _list_ids(service, f"after:{after_ms // 1000} -in:chats -in:drafts -in:sent")
    rows = [parse_message(m) for m in _batch_get(service, ids, "full", MESSAGE_FIELDS, log)]
    rows = [r for r in rows if r["internal_ms"] > after_ms]
    for r in rows:
        r["features"]["replied_after"] = False  # unknown for new mail; not used for prediction
    return sorted(rows, key=lambda r: r["internal_ms"])


def ensure_labels(service, names: Iterable[str]) -> dict[str, str]:
    existing = {lbl["name"]: lbl["id"] for lbl in service.users().labels().list(userId="me").execute()["labels"]}
    for name in names:
        if name not in existing:
            created = (
                service.users()
                .labels()
                .create(
                    userId="me",
                    body={"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"},
                )
                .execute()
            )
            existing[name] = created["id"]
    return {n: existing[n] for n in names}


def add_labels(service, label_id: str, message_ids: list[str]) -> None:
    """ONLY adds a label. There is deliberately no code path that removes labels (e.g. UNREAD/INBOX)."""
    for start in range(0, len(message_ids), 1000):
        body = {"ids": message_ids[start : start + 1000], "addLabelIds": [label_id]}
        assert set(body) == {"ids", "addLabelIds"}, "apply must never remove labels, archive or delete"
        service.users().messages().batchModify(userId="me", body=body).execute()


def export_path() -> Path:
    from ft.context import REPO_ROOT

    return REPO_ROOT / "data" / "email" / "messages.jsonl"


def summarize_export(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}
