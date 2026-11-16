"""Local mail files as a dataset: the folder a message was filed in is its label.

Two source types share everything after reading:
- maildir: a directory tree of one-message-per-file folders. Handles standard Maildir (messages in
  cur/ and new/, Maildir++ subfolders named `.Work.Projects`) and plain folder trees such as the Enron
  corpus or MH (`<folder>/<n>`). The folder label is the path below the root, e.g. "projects/genco".
- mbox: one .mbox file, or a directory of them (Thunderbird, Apple Mail exports). The folder is the
  file's path below the root, or, with `folder_from: header`, a header such as Google Takeout's
  X-Gmail-Labels (comma-separated; a message keeps a label only if exactly one non-excluded label is left).

Then, in order: excluded folders are dropped (glob patterns on the folder path, case-insensitive);
copies of one message are collapsed (same sender, date, subject and body), and a message filed in two
kept folders is dropped as ambiguous; folders with fewer than `min_per_folder` messages are dropped.
Each step's drops are counted, never silent.
"""

from __future__ import annotations

import csv
import hashlib
import mailbox
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ft.redact import sender_token

BODY_CHARS = 4000
RENDER_WORDS = 300
# Bookkeeping files that live next to messages in Maildir/MH/Dovecot/Courier trees.
NOT_MESSAGES = {"maildirfolder", "subscriptions", ".mh_sequences", ".DS_Store"}

_parser = BytesParser(policy=policy.default)


class MailFilesError(RuntimeError):
    pass


# ---------------------------------------------------------------- reading


def _is_bookkeeping(name: str) -> bool:
    return name in NOT_MESSAGES or name.startswith((".", "dovecot", "courier"))


def iter_maildir(root: Path) -> Iterator[tuple[list[str], bytes]]:
    """(folders, raw message) for every message file under root."""
    if not root.is_dir():
        raise MailFilesError(f"maildir path {root} is not a directory")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "tmp")
        here = Path(dirpath)
        parts = list(here.relative_to(root).parts)
        # Standard Maildir: cur/ and new/ belong to the folder that contains them.
        if parts and parts[-1] in ("cur", "new") and (here.parent / "cur").is_dir() and (here.parent / "new").is_dir():
            parts = parts[:-1]
        # Maildir++ subfolders: ".Work.Projects" -> "Work/Projects".
        parts = [p for part in parts for p in (part[1:].split(".") if part.startswith(".") else [part])]
        folder = "/".join(parts) or ("INBOX" if (root / "cur").is_dir() else "(root)")
        for name in sorted(filenames):
            if not _is_bookkeeping(name):
                yield [folder], (here / name).read_bytes()


def _mbox_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise MailFilesError(f"mbox path {root} does not exist")
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not _is_bookkeeping(p.name) and p.suffix != ".msf":  # .msf: Thunderbird index
            with p.open("rb") as f:
                if p.suffix == ".mbox" or f.read(5) == b"From ":
                    out.append(p)
    if not out:
        raise MailFilesError(f"no mbox files under {root}")
    return out


def header_labels(value: str) -> list[str]:
    """'Inbox,"Work, 2024",Opened' -> ['Inbox', 'Work, 2024', 'Opened'] (Takeout quotes labels with commas)."""
    return [v.strip() for v in next(csv.reader([value or ""]), []) if v.strip()]


def iter_mbox(root: Path, folder_from: str, folder_header: str) -> Iterator[tuple[list[str], bytes]]:
    base = root.parent if root.is_file() else root
    for path in _mbox_files(root):
        file_folder = str(path.relative_to(base).with_suffix("")).replace(os.sep, "/")
        box = mailbox.mbox(str(path), create=False)
        try:
            for key in box.iterkeys():
                raw = box.get_bytes(key)
                if folder_from == "file":
                    yield [file_folder], raw
                else:
                    msg = _parser.parsebytes(raw, headersonly=True)
                    yield header_labels(str(msg.get(folder_header, ""))), raw
        finally:
            box.close()


# ---------------------------------------------------------------- parsing (pure)


def _text(part: EmailMessage) -> str:
    try:
        return part.get_content()
    except (LookupError, UnicodeDecodeError, AssertionError):  # unknown or lying charset
        payload = part.get_payload(decode=True) or b""
        return payload.decode("latin-1", errors="replace")


HEADER_LINE = re.compile(r"^[A-Za-z][\w-]*:\s")
TRANSPORT_HEADER = re.compile(
    r"^(return-path|received|received-spf|delivered-to|dkim-signature|domainkey-signature|authentication-results"
    r"|arc-[\w-]+|message-id|mime-version|content-type|content-transfer-encoding|x-[\w-]+):",
    re.IGNORECASE,
)
HUMAN_HEADER = re.compile(r"^(from|to|cc|subject|date|sent):", re.IGNORECASE)


def strip_header_blocks(text: str) -> tuple[str, int]:
    """Remove raw RFC 822 header blocks pasted into bodies (forwards often start with Return-Path,
    Received, Message-ID, X-... lines). A block starts at a transport header and runs to the next blank
    line, which also catches wrapped lines without indentation (Enron re-wrapped them). It counts only if
    at least two lines look like headers, so prose such as "Received: your note" survives. The readable
    From/To/Cc/Subject/Date lines inside a block are kept. Returns (text, blocks removed)."""
    lines = text.splitlines()
    out: list[str] = []
    removed = i = 0
    while i < len(lines):
        bare = lines[i].lstrip("> \t")
        if TRANSPORT_HEADER.match(bare):
            j = i
            while j < len(lines) and lines[j].lstrip("> \t").strip():
                j += 1
            block = [line.lstrip("> \t") for line in lines[i:j]]
            if sum(bool(HEADER_LINE.match(line)) for line in block) >= 2:
                out += [line for line in block if HUMAN_HEADER.match(line)]
                removed += 1
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), removed


def extract_body(msg: EmailMessage) -> str:
    return _body(msg)[0]


def _body(msg: EmailMessage) -> tuple[str, int]:
    """Plain text if present, else HTML converted to text; raw header blocks removed, whitespace
    collapsed, first BODY_CHARS. Returns (text, header blocks removed)."""
    plain = html = ""
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain" and not plain.strip():
            plain = _text(part)
        elif ctype == "text/html" and not html:
            html = _text(part)
    text = plain
    if not text.strip() and html:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width = True, True, 0
        text = h.handle(html)
    text, removed = strip_header_blocks(text)
    return " ".join(text.split())[:BODY_CHARS], removed


def parse(raw: bytes) -> dict[str, Any] | None:
    """Normalized message, or None when there's no usable date (time splits need one)."""
    msg = _parser.parsebytes(raw)
    try:
        when = parsedate_to_datetime(str(msg.get("date", "")))
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    _name, address = parseaddr(str(msg.get("from", "")))
    recipients = getaddresses([str(msg.get(k, "")) for k in ("to", "cc")])
    body, header_blocks = _body(msg)
    return {
        "date": when.astimezone(UTC).isoformat(timespec="seconds"),
        "from_address": address.lower(),
        "n_recipients": len([a for _, a in recipients if a]),
        "subject": " ".join(str(msg.get("subject", "")).split()),
        "body": body,
        "header_blocks_stripped": header_blocks,
        "has_unsubscribe": "list-unsubscribe" in msg,
    }


def content_key(row: dict[str, Any]) -> str:
    """Copies of one message (Enron files the same mail under several folders, each copy with its own
    Message-ID) share sender, date, subject and body."""
    ident = "\x1f".join([row["from_address"], row["date"], row["subject"], row["body"]])
    return hashlib.sha1(ident.encode()).hexdigest()[:12]


def excluded(folder: str, patterns: list[str]) -> bool:
    f = folder.casefold()
    return any(fnmatchcase(f, p.casefold()) for p in patterns)


# ---------------------------------------------------------------- source -> filed messages


def read_messages(src, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse, exclude, de-duplicate and size-filter. Returns (rows sorted by date, stats)."""
    is_maildir = src.type == "maildir"
    stream = iter_maildir(root) if is_maildir else iter_mbox(root, src.folder_from, src.folder_header)
    dropped: Counter[str] = Counter()
    excluded_folders: Counter[str] = Counter()
    copies: dict[str, dict[str, Any]] = {}
    folders_of: dict[str, set[str]] = defaultdict(set)
    seen = 0
    for folders, raw in stream:
        seen += 1
        kept = [f for f in folders if not excluded(f, src.exclude_folders)]
        for f in folders:
            if f not in kept:
                excluded_folders[f] += 1
        if not kept:
            dropped["excluded_folder"] += 1
            continue
        if len(set(kept)) > 1:  # e.g. two user labels in a Takeout export
            dropped["several_labels"] += 1
            continue
        try:
            row = parse(raw)
        except Exception:  # noqa: BLE001 - a malformed file must not stop the load
            dropped["unparseable"] += 1
            continue
        if row is None:
            dropped["no_date"] += 1
            continue
        if not row["subject"] and not row["body"]:
            dropped["empty"] += 1
            continue
        key = content_key(row)
        if key in copies:
            dropped["duplicate_copy"] += 1
        else:
            copies[key] = row
        folders_of[key].add(kept[0])

    rows = []
    for key, row in copies.items():
        if len(folders_of[key]) > 1:
            dropped["filed_in_several_folders"] += 1
            continue
        rows.append({**row, "id": key, "folder": next(iter(folders_of[key]))})
    counts = Counter(r["folder"] for r in rows)
    small = {f: n for f, n in counts.items() if n < src.min_per_folder}
    if small:
        dropped["small_folder"] += sum(small.values())
        rows = [r for r in rows if r["folder"] not in small]
    rows.sort(key=lambda r: (r["date"], r["id"]))
    stats = {
        "files_seen": seen,
        "messages": len(rows),
        "folders": dict(Counter(r["folder"] for r in rows).most_common()),
        "small_folders_dropped": dict(sorted(small.items(), key=lambda kv: -kv[1])),
        "excluded_folders": dict(excluded_folders.most_common()),
        "dropped": dict(dropped),
        "header_blocks_stripped": sum(r["header_blocks_stripped"] for r in rows),
        "date_range": [rows[0]["date"], rows[-1]["date"]] if rows else None,
    }
    return rows, stats


# ---------------------------------------------------------------- rendering


class NoRedaction:
    """Stand-in for Redactor when a source sets `redact: false` (public corpora)."""

    def __init__(self):
        self.counts: Counter[str] = Counter()

    def redact(self, text: str) -> str:
        return text


def render(row: dict[str, Any], redactor, *, redact_sender: bool = True) -> str:
    """The model's input. Same shape as the Gmail connector's, minus Gmail-only fields."""
    if redact_sender:
        token, domain = sender_token(row["from_address"])
        sender = f"{token} ({domain})"
    else:
        sender = row["from_address"]
    words = row["body"].split()[:RENDER_WORDS]
    return "\n".join(
        [
            f"From: {sender}",
            f"Recipients: {row['n_recipients']}",
            f"Subject: {redactor.redact(row['subject'])}",
            "",
            redactor.redact(" ".join(words)),
        ]
    )


def fields_rendered(redacting: bool) -> list[str]:
    r = " (redacted)" if redacting else ""
    return [
        "sender token + domain" if redacting else "sender address",
        "recipient count",
        f"subject{r}",
        f"first {RENDER_WORDS} words of body{r}",
    ]


def summarize(stats: dict[str, Any]) -> str:
    return f"{stats['messages']} messages in {len(stats['folders'])} folders" + (
        f" (from {stats['date_range'][0][:10]} to {stats['date_range'][1][:10]})" if stats["date_range"] else ""
    )
