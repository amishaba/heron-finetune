"""The static site (demo/build_static.py): real UI, contact note, and nothing private in the public files."""

import json
import subprocess

import pytest

from demo.build_featured import ADDRESS_RE, check_public
from demo.build_static import CONTACT, ROOT, build


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "web"
    build(out)
    return out


def test_every_featured_run_has_its_files(site):
    runs = json.loads((site / "results" / "featured.json").read_text())
    assert len(runs) == 4
    for run in runs:
        assert json.loads((site / "results" / f"{run['slug']}.report.json").read_text())["verdict"]
        assert "<html" in (site / "reports" / f"{run['slug']}.html").read_text().lower()


def test_no_built_file_is_gitignored(site):
    """Vercel deploys what's committed, so an ignored file works locally and is missing from the live site."""
    paths = [f"web/{p.relative_to(site)}" for p in site.rglob("*") if p.is_file()]
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths], cwd=ROOT, capture_output=True, text=True
    ).stdout
    assert not ignored


def test_page_is_static_with_contact_and_repo_link(site):
    page = (site / "index.html").read_text()
    assert f'contact: "{CONTACT}"' in page
    assert page.index("HERON_STATIC") < page.index("/static/app.js")
    assert "github.com/" in page and "live ·" not in page


def test_page_links_icons_that_exist(site):
    page = (site / "index.html").read_text()
    for name in ("favicon.ico", "favicon.svg", "apple-touch-icon.png"):
        assert f'href="/static/{name}"' in page and (site / "static" / name).stat().st_size > 0
    assert (site / "favicon.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"  # a real .ico, not the page's HTML


def test_only_the_contact_address_is_public(site):
    found = set()
    for path in site.rglob("*"):
        if path.is_file():
            found |= set(ADDRESS_RE.findall(path.read_text(errors="ignore")))
    assert found <= {CONTACT}


def test_email_report_keeps_the_result_but_not_the_messages(site):
    report = json.loads((site / "results" / "email.report.json").read_text())
    assert report["verdict"]["sentence"] and report["comparison"]
    assert report["errors"]["confusions"] and all(c["examples"] == [] for c in report["errors"]["confusions"])
    raw = (site / "reports" / "email.html").read_text()
    assert "Recipients:" not in raw and "Subject:" not in raw


def test_other_reports_keep_their_examples(site):
    report = json.loads((site / "results" / "banking77.report.json").read_text())
    assert any(c["examples"] for c in report["errors"]["confusions"])


@pytest.mark.parametrize("leak", ["someone@yahoo.com", "maildir/someone-x", "/Users/me/runs"])
def test_build_stops_on_private_data(tmp_path, leak):
    (tmp_path / "report.json").write_text(json.dumps({"example": leak}))
    with pytest.raises(ValueError, match="public output"):
        check_public(tmp_path)
