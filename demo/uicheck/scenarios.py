"""Behavioral checks of the demo UI in a real (headless) Chrome: clicks every link and button, walks the
sample → plan → approve → result flow, Back/Forward, bad uploads, unknown URLs, phone width, leaving a page
mid-poll. Prints PASS/FAIL per check and exits non-zero on any failure.

    FT_DEMO_DRY_RUN=1 FT_DEMO_RUNS_PER_VISITOR_PER_DAY=0 FT_RUNS_DIR=/tmp/ftui/runs FT_DEMO_DIR=/tmp/ftui/up \
        .venv/bin/uvicorn demo.server:app --port 8000 &
    .venv/bin/python demo/uicheck/scenarios.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from harness import Browser  # noqa: E402

BASE = os.environ.get("FT_UI_BASE", "http://127.0.0.1:8000")
HERE = tempfile.mkdtemp()  # scratch files for the upload checks
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


# A #section link should land the section right under the sticky header: its top minus the header's height.
# (+ 0 turns -0 into 0: CDP can't return -0 by value and would hand back None.)
TOP = """(id) => Math.round(document.getElementById(id)?.getBoundingClientRect().top
  - document.querySelector('header.top').getBoundingClientRect().height) + 0"""
APP_MARK = "document.querySelector('#app > *')?.dataset.mark"


async def home_and_anchors(b):
    await b.goto(BASE + "/", 3)
    check("home renders plates", await b.js("document.querySelectorAll('.run-plate').length") == 4)
    check("home renders specimens", await b.js("document.querySelectorAll('.specimen').length") == 4)
    check("trust row filled", "92.4" in (await b.js("document.getElementById('trust').textContent") or ""))
    # in-page nav: must scroll to the section WITHOUT re-rendering the whole view
    await b.js("document.querySelector('#app > *').dataset.mark = 'same-render'")
    for label, sel, target in [
        ("nav How it works", 'nav a[href="/#method"]', "method"),
        ("nav Runs", 'nav a[href="/#runs"]', "runs"),
        ("nav Results", 'nav a[href="/#results"]', "results"),
        ("nav Try it", "nav a.cta", "try"),
        ("hero Upload a dataset", '.ctas a[href="#try"]', "try"),
        ("hero See real runs", '.ctas a[href="#runs"]', "runs"),
    ]:
        await b.js("window.scrollTo(0,0)")
        await asyncio.sleep(0.2)
        await b.js(f"document.querySelector({sel!r}).click()")
        await asyncio.sleep(1.6)  # smooth scroll
        top = await b.js(f"({TOP})('{target}')")
        same = await b.js(APP_MARK) == "same-render"
        check(
            f"{label} lands on #{target} without re-render",
            top is not None and abs(top) <= 4 and same,
            f"top={top} same_view={same}",
        )


async def deep_links(b):
    for h in ["results", "try", "method", "runs"]:
        await b.goto(f"{BASE}/#{h}", 3)
        top = await b.js(f"({TOP})('{h}')")
        check(f"deep link /#{h}", top is not None and abs(top) <= 4, f"top={top}")


async def sample_flow(b):
    await b.goto(BASE + "/", 2.5)
    await b.click('[data-sample="banking77"]', 1.5)
    path = await b.js("location.pathname")
    check("sample button navigates to run page", path.startswith("/run/demo-"), path)
    check("run page scrolled to top", await b.js("Math.round(scrollY)") == 0)
    planned = await b.until("document.getElementById('approve') ? 1 : 0", 90)
    check("plan + approve button appear", planned)
    check("estimate rendered", "Total" in (await b.js("document.getElementById('main').textContent") or ""))
    # the main column must not be rebuilt while nothing changed (keeps scroll/focus)
    await b.js("document.getElementById('main').firstElementChild.dataset.mark = 'kept'")
    await asyncio.sleep(9)  # > one idle poll
    check(
        "main column not rebuilt between identical polls",
        await b.js("document.getElementById('main').firstElementChild?.dataset.mark") == "kept",
    )
    await b.click("#approve", 0.5)
    done = await b.until("document.querySelector('.big-verdict') ? 1 : 0", 180)
    check("approve → run completes → verdict shown", done)
    check("status pill says Done", "Done" in (await b.js("document.querySelector('.status-pill')?.textContent") or ""))
    check("orbit hidden when finished", await b.js("document.getElementById('vortex').hidden") is True)
    await b.click(".result-actions a", 2.5)
    path = await b.js("location.pathname")
    check(
        "full report opens as an app page",
        path.endswith("/report") and await b.js("document.querySelectorAll('.report-sec').length") >= 7,
        path,
    )
    await b.js("history.back()")
    await asyncio.sleep(2)
    run_url = await b.js("location.href")
    return run_url


async def leave_mid_poll(b):
    # in-app navigation home -> run (pushState), then Back while the run page is polling every 3-8 s
    await b.goto(BASE + "/", 2.5)
    await b.click('[data-sample="banking77"]', 1.0)
    polling = await b.until("__fetches.filter(f => f[1].includes('/api/runs/demo-')).length >= 2 ? 1 : 0", 30)
    await b.js("history.back()")
    await asyncio.sleep(1.5)
    on_home = await b.js("location.pathname === '/' && document.querySelectorAll('.run-plate').length === 4")
    n0 = await b.js("__fetches.filter(f => f[1].includes('/api/runs/demo-')).length")
    await asyncio.sleep(10)
    n1 = await b.js("__fetches.filter(f => f[1].includes('/api/runs/demo-')).length")
    check(
        "Back mid-poll: home shown, polling stopped",
        polling and on_home and n0 == n1,
        f"polling={polling} home={on_home} fetches {n0}->{n1}",
    )


async def history(b, run_url):
    await b.goto(BASE + "/", 2.5)
    await b.goto(run_url, 2.5)
    await b.js("history.back()")
    await asyncio.sleep(2.5)
    check("back → home renders", await b.js("document.querySelectorAll('.run-plate').length") == 4)
    await b.js("history.forward()")
    await asyncio.sleep(2.5)
    check("forward → run renders", await b.js("!!document.querySelector('.big-verdict')"))
    # a nav link from the run page goes home and lands on the section
    await b.js("document.querySelector('nav a[href=\"/#method\"]').click()")
    await asyncio.sleep(3)
    top = await b.js(f"({TOP})('method')")
    check("nav link from run page lands on #method", top is not None and abs(top) <= 4, f"top={top}")


async def decline_and_errors(b):
    await b.goto(BASE + "/", 2.5)
    await b.click('[data-sample="too-small"]', 1)
    declined = await b.until("document.querySelector('[data-link=\"/\"]') ? 1 : 0", 60)
    check("decline shows reason + back link", declined)
    await b.click('[data-link="/"]', 2)
    check(
        "'Try another dataset' goes home",
        await b.js("location.pathname") == "/" and await b.js("document.querySelectorAll('.run-plate').length") == 4,
    )
    # bad uploads show a message, page stays
    bad = os.path.join(HERE, "bad.txt")
    Path(bad).write_text("a,b\n1,2\n")
    await b.set_file("#file", bad)
    await asyncio.sleep(1.5)
    check(
        "bad file type shows error", "csv or .jsonl" in (await b.js("document.getElementById('err').textContent") or "")
    )
    one = os.path.join(HERE, "onecol.csv")
    Path(one).write_text("a\n1\n2\n")
    await b.set_file("#file", one)
    await asyncio.sleep(1.5)
    check(
        "one-column CSV shows error", "two columns" in (await b.js("document.getElementById('err').textContent") or "")
    )
    # unknown run id and unknown path
    await b.goto(BASE + "/run/demo-00000000", 2.5)
    msg = await b.js("document.getElementById('main')?.textContent || document.body.textContent")
    check("unknown run shows a message", "unknown run" in (msg or ""), (msg or "")[:60])
    await b.goto(BASE + "/no/such/page", 2)
    path = await b.js("location.pathname")
    plates = await b.js("document.querySelectorAll('.run-plate').length")
    api = await b.js("fetch('/api/nope').then(r => r.status)")
    check(
        "unknown path → home page (URL corrected); unknown API path still 404",
        path == "/" and plates == 4 and api == 404,
        f"path={path} plates={plates} api={api}",
    )


async def featured_reports(b):
    await b.goto(BASE + "/", 2.5)
    slugs = await b.js("[...document.querySelectorAll('.run-plate a.btn')].map(a => a.getAttribute('href'))")
    check("featured plates link to report pages", len(slugs or []) == 4, f"{slugs}")
    # clicking a plate's "Full report" stays in the app (no reload) and renders the report with the site nav
    await b.js("document.querySelector('#app > *').dataset.mark = 'spa'")
    await b.click(".run-plate a.btn", 2.5)
    path = await b.js("location.pathname")
    check(
        "Full report opens in-app",
        path == slugs[0] and await b.js("document.querySelectorAll('.report-sec').length") >= 7,
        path,
    )
    for href in slugs:
        await b.goto(BASE + href, 2.5)
        sections = await b.js("document.querySelectorAll('.report-sec').length")
        raw = await b.js(
            "fetch(document.querySelector('.report-actions a[target]').getAttribute('href')).then(r => r.status)"
        )
        check(
            f"{href}: report renders, raw report loads", sections >= 7 and raw == 200, f"sections={sections} raw={raw}"
        )


NAV = """JSON.stringify([...document.querySelectorAll('header.top nav a')]
  .map((a) => a.textContent.trim() + ' ' + a.getAttribute('href')))"""
NEW_SAMPLE_RUN = """(() => { const f = new FormData(); f.append('sample', 'too-small');
  return fetch('/api/runs', { method: 'POST', body: f }).then((r) => r.json()).then((j) => j.id); })()"""


async def same_nav_everywhere(b):
    await b.goto(BASE + "/", 2.5)
    home_nav = await b.js(NAV)
    run_id = await b.js(NEW_SAMPLE_RUN)
    for path in ["/featured/banking77", f"/run/{run_id}", "/no/such/page"]:
        await b.goto(BASE + path, 2.5)
        check(f"same top nav on {path}", await b.js(NAV) == home_nav)


async def resize_and_replay(b):
    await b.goto(BASE + "/", 2.5)
    await b.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
    await asyncio.sleep(1.2)
    over = await b.js("document.documentElement.scrollWidth - innerWidth")
    check("390px phone: no horizontal overflow", over == 0, f"overflow={over}px")
    await b.call("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
    await asyncio.sleep(1)
    lines = await b.until("document.querySelectorAll('#console .ln').length >= 9 ? 1 : 0", 20)
    chip = await b.until("document.getElementById('chip').classList.contains('on') ? 1 : 0", 20)
    check("console replay runs and shows result chip", lines and chip)
    await asyncio.sleep(7)
    restarted = await b.js("document.querySelectorAll('#console .ln').length < 9")
    check("console replay restarts", restarted)


async def main():
    for name, fn in [
        ("home", home_and_anchors),
        ("deep links", deep_links),
        ("featured", featured_reports),
        ("resize", resize_and_replay),
        ("decline/errors", decline_and_errors),
    ]:
        async with Browser() as b:
            print(f"--- {name}")
            await fn(b)
            errs = await b.page_errors()
            check(f"{name}: no JS errors", not errs, "; ".join(map(str, errs))[:300])
    async with Browser() as b:
        print("--- sample flow + history")
        run_url = await sample_flow(b)
        await history(b, run_url)
        await leave_mid_poll(b)
        errs = await b.page_errors()
        check("flow: no JS errors", not errs, "; ".join(map(str, errs))[:300])
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


asyncio.run(main())
