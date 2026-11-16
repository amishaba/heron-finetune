"""Minimal Chrome DevTools Protocol driver for behavioral UI checks (headless Chrome; needs only `websockets`)."""

import asyncio
import base64
import itertools
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
_ports = itertools.count(9400)

# Runs before any page script: counts fetches and collects uncaught errors and rejections.
INSTRUMENT = """
window.__fetches = [];
const realFetch = window.fetch;
window.fetch = (u, o) => { __fetches.push([Math.round(performance.now()), String(u)]); return realFetch(u, o); };
window.__errors = [];
addEventListener('error', (e) => __errors.push(String(e.message)));
addEventListener('unhandledrejection', (e) => __errors.push('unhandled: ' + (e.reason?.message || e.reason)));
"""

# Scrolls an element into view and clicks it, like a person would. The scroll is instant: a person clicks a
# button that is already on screen, and a smooth scroll still running would race the page's own scrolling.
CLICK = """(sel) => {
  const el = document.querySelector(sel);
  if (!el) return false;
  el.scrollIntoView({ block: 'center', behavior: 'instant' });
  el.click();
  return true;
}"""


class Browser:
    def __init__(self, width=1440, height=900):
        self.port = next(_ports)
        self.width, self.height = width, height
        self.errors: list[str] = []
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}

    async def __aenter__(self):
        self.proc = subprocess.Popen(
            [CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={self.port}",
             f"--user-data-dir={tempfile.mkdtemp()}", f"--window-size={self.width},{self.height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        tabs = []
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                break
            except OSError:
                time.sleep(0.1)
        url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        self.ws = await websockets.connect(url, max_size=50_000_000)
        self._reader = asyncio.create_task(self._read())
        await self.call("Page.enable")
        await self.call("Runtime.enable")
        await self.call("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT)
        return self

    async def __aexit__(self, *exc):
        self._reader.cancel()
        await self.ws.close()
        self.proc.kill()

    async def _read(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg and msg["id"] in self._pending:
                self._pending.pop(msg["id"]).set_result(msg)
            elif msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                self.errors.append((d.get("exception") or {}).get("description") or d.get("text"))

    async def call(self, method, **params):
        self._id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._id] = fut
        await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        msg = await asyncio.wait_for(fut, 30)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    async def goto(self, url, wait=2.5):
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            return f"JS-ERROR: {d.get('exception', {}).get('description', d.get('text'))}"
        return r.get("result", {}).get("value")

    async def click(self, selector, wait=1.0):
        ok = await self.js(f"({CLICK})({json.dumps(selector)})")
        await asyncio.sleep(wait)
        return ok

    async def until(self, expr, timeout=60, every=0.5):
        end = time.time() + timeout
        while time.time() < end:
            v = await self.js(expr)
            if v and not (isinstance(v, str) and v.startswith("JS-ERROR")):
                return v
            await asyncio.sleep(every)
        return None

    async def set_file(self, selector, path):
        doc = await self.call("DOM.getDocument")
        node = await self.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector=selector)
        await self.call("DOM.setFileInputFiles", files=[path], nodeId=node["nodeId"])

    async def shot(self, path):
        r = await self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["data"]))

    async def page_errors(self):
        return self.errors + (await self.js("window.__errors || []") or [])
