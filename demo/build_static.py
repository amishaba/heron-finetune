"""Build web/: the Heron page as static files for Vercel. Finished runs only, so no server, keys or spend.

The page and scripts are copied from demo/web, so this stays the real UI. A small inline script answers the page's
/api calls from the files in web/results (not "data" or "runs": .gitignore skips folders with those names), and
window.HERON_STATIC swaps the upload for a contact note. Featured runs
come from demo/featured (see build_featured.py), which already leaves out quoted emails; the build stops if any
email address other than the contact one reaches web/.

    .venv/bin/python demo/build_static.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from demo.build_featured import check_public

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).parent / "web"
FEATURED = Path(__file__).parent / "featured"
OUT = ROOT / "web"
CONTACT = "amishabansal21@gmail.com"

# (text in demo/web/index.html, what the static page says instead)
PAGE_EDITS = [
    ("live · automated LLM fine-tuning", "automated LLM fine-tuning"),
    (">Try it</a>", ">Run yours</a>"),
    ('content="Upload a labeled dataset. An agent', 'content="Give it a labeled dataset. An agent'),
    ('content="Upload a dataset; an agent', 'content="Give it a dataset; an agent'),
]

SHIM = """<script>
window.HERON_STATIC = { contact: %(contact)s };
(function () {
  const slugs = %(slugs)s, realFetch = window.fetch.bind(window);
  const missing = (error) => Promise.resolve(new Response(JSON.stringify({ error }), {
    status: 404, headers: { "content-type": "application/json" },
  }));
  window.fetch = (url, opts) => {
    const path = String(url);
    if (!path.startsWith("/api/")) return realFetch(url, opts);
    if (path === "/api/featured") return realFetch("/results/featured.json", opts);
    const report = path.match(/^\\/api\\/featured\\/([a-z0-9-]+)\\/report$/);
    if (report && slugs.includes(report[1])) return realFetch(`/results/${report[1]}.report.json`, opts);
    return missing("unknown report");
  };
})();
</script>
"""


def build_page(slugs: list[str]) -> str:
    page = (SOURCE / "index.html").read_text()
    for old, new in PAGE_EDITS:
        if page.count(old) != 1:
            raise ValueError(f"index.html changed: expected exactly one {old!r}")
        page = page.replace(old, new)
    shim = SHIM % {"contact": json.dumps(CONTACT), "slugs": json.dumps(slugs)}
    scripts = '  <script src="/static/art.js"></script>'
    if page.count(scripts) != 1:
        raise ValueError("index.html changed: can't find where to add the static script")
    return page.replace(scripts, shim + scripts)


def build(out: Path = OUT) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True)
    (out / "results").mkdir()
    (out / "reports").mkdir()
    for name in ("style.css", "art.js", "app.js", "favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        shutil.copy(SOURCE / name, out / "static" / name)
    shutil.copy(SOURCE / "favicon.ico", out / "favicon.ico")  # for the raw reports, which don't link an icon

    runs = json.loads((FEATURED / "featured.json").read_text())
    for run in runs:
        slug = run["slug"]
        shutil.copy(FEATURED / f"{slug}.report.json", out / "results" / f"{slug}.report.json")
        shutil.copy(FEATURED / f"{slug}.html", out / "reports" / f"{slug}.html")
    (out / "results" / "featured.json").write_text(json.dumps(runs, indent=2))
    (out / "index.html").write_text(build_page([run["slug"] for run in runs]))
    check_public(out, frozenset({CONTACT}))


if __name__ == "__main__":
    build()
    print(f"built {OUT}")
