"""maildir / mbox sources: folder trees (Enron-style, standard Maildir, Maildir++), mbox files and
Google Takeout labels, exclusions, de-duplication, size filtering, and a full dry run."""

import json
import mailbox
import random
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime

import pytest
import yaml
from rich.console import Console

from ft.config import ConfigError, MaildirSource, MboxSource, load_task, parse_task
from ft.connectors import mailfiles
from ft.context import REPO_ROOT, RunContext
from ft.pipeline import run_pipeline

quiet = Console(quiet=True)
START = datetime(2001, 1, 1, 9, 0, tzinfo=UTC)
TOPICS = {
    "projects": ("Budget for the {n} project", "Revised budget for phase {n}. Call Maria Lopez at 713-555-0142."),
    "legal": ("Contract review {n}", "Please review clause {n} of the master agreement. -- Tom Becker"),
    "travel": ("Flight itinerary {n}", "Your flight to Portland departs at {n}:15. Sent to john.doe@example.com."),
}


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FT_CONFIG_DIR", str(tmp_path / "ftconfig"))


def message(i: int, folder: str, *, sender=None, when=None, html=False) -> bytes:
    subject, body = TOPICS.get(folder, ("Note {n}", "Misc note number {n}."))
    m = EmailMessage()
    m["From"] = sender or f"person{i % 7}@enron.com"
    m["To"] = "me@enron.com, team@enron.com"
    m["Subject"] = subject.format(n=i)
    m["Date"] = format_datetime(when or START + timedelta(hours=i))
    m["Message-ID"] = f"<{random.random()}@example>"  # Enron gives every copy its own id
    if html:
        m.set_content(f"<p>{body.format(n=i)}</p>", subtype="html")
    else:
        m.set_content(body.format(n=i))
    return bytes(m)


def enron_tree(root, per_folder=30):
    """root/<folder>/<n>. files, plus system folders holding copies of filed mail."""
    n = 0
    for folder in ("projects", "legal", "travel"):
        for _ in range(per_folder):
            n += 1
            raw = message(n, folder, html=n % 5 == 0)
            for where in (folder, "all_documents"):
                d = root / where
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{n}.").write_bytes(raw)
    for k in range(3):  # too small to keep
        (root / "misc").mkdir(exist_ok=True)
        (root / "misc" / f"{k + 1}.").write_bytes(message(500 + k, "misc"))
    for k in range(5):
        (root / "inbox").mkdir(exist_ok=True)
        (root / "inbox" / f"{k + 1}.").write_bytes(message(600 + k, "inbox"))
    both = message(700, "projects")  # filed under two hand-made folders: ambiguous
    (root / "projects" / "700.").write_bytes(both)
    (root / "legal" / "700.").write_bytes(both)
    (root / "legal" / "701.").write_bytes(b"From: x@y.com\nSubject: no date\n\nbody")
    return root


SYSTEM = ["inbox", "sent*", "all_documents", "deleted_items", "discussion_threads"]


def src(type_="maildir", **kw):
    cls = MaildirSource if type_ == "maildir" else MboxSource
    return cls(**{"type": type_, "path": "unused", "exclude_folders": SYSTEM, "min_per_folder": 10, **kw})


def test_enron_style_tree(tmp_path):
    rows, stats = mailfiles.read_messages(src(), enron_tree(tmp_path / "mail"))
    assert stats["folders"] == {"projects": 30, "legal": 30, "travel": 30}
    d = stats["dropped"]
    assert d["excluded_folder"] == 95  # 90 copies in all_documents + 5 inbox
    assert d["filed_in_several_folders"] == 1 and d["no_date"] == 1
    assert d["small_folder"] == 3 and stats["small_folders_dropped"] == {"misc": 3}
    assert d["duplicate_copy"] == 1  # the second copy of the ambiguous one
    assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)
    assert all("<p>" not in r["body"] for r in rows) and all(r["body"] for r in rows)  # HTML became text


def test_standard_maildir_and_maildir_plus_plus(tmp_path):
    root = tmp_path / "Maildir"
    for sub in ("cur", "new", "tmp", ".Work.Projects/cur", ".Work.Projects/new", ".Work.Projects/tmp"):
        (root / sub).mkdir(parents=True)
    for i in range(12):
        (root / ("cur" if i % 2 else "new") / f"{i}.M1P1.host:2,S").write_bytes(message(i, "travel"))
        (root / ".Work.Projects" / "cur" / f"{i}.M2P1.host:2,S").write_bytes(message(100 + i, "projects"))
    (root / ".Work.Projects" / "maildirfolder").write_bytes(b"")
    (root / "dovecot-uidlist").write_bytes(b"3 V1 N1\n")
    (root / "tmp" / "half-written").write_bytes(b"From: x\n")
    _rows, stats = mailfiles.read_messages(src(exclude_folders=[]), root)
    assert stats["folders"] == {"INBOX": 12, "Work/Projects": 12}
    assert stats["dropped"] == {}


def test_mbox_directory_and_takeout_labels(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    for folder in ("legal", "travel"):
        box = mailbox.mbox(str(d / f"{folder}.mbox"))
        for i in range(11):
            box.add(message(i + (100 if folder == "legal" else 0), folder))
        box.close()
    _rows, stats = mailfiles.read_messages(src("mbox"), d)
    assert stats["folders"] == {"legal": 11, "travel": 11}

    takeout = tmp_path / "All mail Including Spam and Trash.mbox"
    box = mailbox.mbox(str(takeout))
    labels = ['Inbox,Opened,"Trips, 2024"', "Opened,Receipts", "Inbox,Opened", 'Receipts,"Trips, 2024"']
    for i in range(48):
        m = mailbox.mboxMessage(message(i, "travel"))
        m["X-Gmail-Labels"] = labels[i % 4]
        box.add(m)
    box.close()
    cfg = src("mbox", folder_from="header")
    cfg.exclude_folders = ["inbox", "opened", "sent", "important"]
    cfg.min_per_folder = 5
    _rows, stats = mailfiles.read_messages(cfg, takeout)
    assert stats["folders"] == {"Trips, 2024": 12, "Receipts": 12}
    assert stats["dropped"] == {"excluded_folder": 12, "several_labels": 12}
    assert mailfiles.header_labels('a,"b, c" , d') == ["a", "b, c", "d"]


def test_mail_sources_need_time_split_and_dataset_targets():
    raw = yaml.safe_load((REPO_ROOT / "examples" / "enron" / "task.yaml").read_text())
    parse_task(raw)
    no_time = {**raw, "splits": {k: v for k, v in raw["splits"].items() if k != "by_time"}}
    with pytest.raises(ConfigError, match="by_time"):
        parse_task(no_time)
    with pytest.raises(ConfigError, match="targets.from: dataset"):
        parse_task({**raw, "targets": {"from": "frontier", "frontier_model": "gpt-6-luna"}})


def test_maildir_dry_run_end_to_end(tmp_path):
    enron_tree(tmp_path / "mail", per_folder=40)
    raw = yaml.safe_load((REPO_ROOT / "examples" / "enron" / "task.yaml").read_text())
    raw["source"].update(path=str(tmp_path / "mail"), min_per_folder=10, redact=True)
    raw["splits"]["test_max"] = 30
    p = tmp_path / "enron.yaml"
    p.write_text(yaml.safe_dump(raw))
    ctx = RunContext.create(p, dry_run=True, yes=True, run_id="mail")
    run_pipeline(ctx, quiet)

    assert ctx.labels() == ["legal", "projects", "travel"]
    dates = {
        s: [json.loads(line)["meta"]["date"] for line in (ctx.data_dir / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "val", "test")
    }
    assert max(dates["train"]) <= min(dates["val"]) and max(dates["val"]) <= min(dates["test"])
    upload = (ctx.attempt_dir(1) / "train.together.jsonl").read_text()
    assert "Maria Lopez" not in upload and "555-0142" not in upload and "john.doe@" not in upload
    assert "person1@enron.com" not in upload and "sender_" in upload
    report = json.loads((ctx.run_dir / "report.json").read_text())
    assert report["privacy"]["redaction"]["redacted"] is True
    stats = json.loads((ctx.data_dir / "load_stats.json").read_text())
    assert stats["small_folders_dropped"] == {"misc": 3}


def test_redact_false_keeps_text(tmp_path):
    rows, _ = mailfiles.read_messages(src(), enron_tree(tmp_path / "mail"))
    r = next(r for r in rows if r["folder"] == "projects")
    text = mailfiles.render(r, mailfiles.NoRedaction(), redact_sender=False)
    assert "Maria Lopez" in text and r["from_address"] in text
    assert load_task(REPO_ROOT / "examples" / "enron" / "task.yaml").source.type == "maildir"


ENRON_FORWARD = """Return-Path: <pat.doe@example.com>
Received: from  rly-xa04.mx.aol.com (rly-xa04.mail.aol.com [172.20.105.73])
by air-xa03.mail.aol.com (v76_r1.8) with ESMTP; Wed, 18 Oct 2000 13:53:51
-0400
Received: from nahou-msmsw03px.corp.enron.com ([172.28.10.39]) by
postmaster.enron.com (8.8.8/8.8.8/postmaster-1.00) with ESMTP id MAA18325 for
From: pat.doe@example.com
Subject: part-time work
Message-ID: <OF1EB17063.BB14093D-ON8625697C.00624EBF@enron.com>
X-MIMETrack: Serialize by Router on ENE-MTA01/Enron(Release 5.0.3 (Intl)|21
March 2000) at 10/18/2000 12:50:19 PM
MIME-Version: 1.0
X-Mailer: Unknown

Received: your note about the part-time role, thanks.
Can the student start in January?
"""


def test_raw_header_blocks_are_stripped_but_prose_and_human_headers_stay():
    text, removed = mailfiles.strip_header_blocks(ENRON_FORWARD)
    assert removed == 1
    for gone in ("Return-Path", "rly-xa04", "air-xa03", "-0400", "MAA18325", "Message-ID", "March 2000", "X-Mailer"):
        assert gone not in text
    assert "From: pat.doe@example.com" in text and "Subject: part-time work" in text
    assert "Received: your note about the part-time role" in text and "start in January" in text
    quoted = "\n".join("> " + line for line in ENRON_FORWARD.splitlines())
    assert mailfiles.strip_header_blocks(quoted)[1] == 1


def test_loader_counts_stripped_header_blocks(tmp_path):
    root = tmp_path / "mail" / "ut"
    root.mkdir(parents=True)
    for i in range(10):
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = "a@aol.com", "b@enron.com", f"Fwd: part-time work {i}"
        m["Date"] = format_datetime(START + timedelta(days=i))
        m.set_content(ENRON_FORWARD)
        (root / f"{i + 1}.").write_bytes(bytes(m))
    rows, stats = mailfiles.read_messages(src(min_per_folder=1), tmp_path / "mail")
    assert stats["header_blocks_stripped"] == 10
    assert all("Received: from" not in r["body"] and "start in January" in r["body"] for r in rows)
