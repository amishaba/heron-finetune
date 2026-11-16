// Heron demo: a tiny client-side router with three views, all under the same top nav.
//   home    /                    hero, results, upload, method, real runs (featured runs come from /api/featured)
//   run     /run/<id>            one live run: the agent's plan, approval, progress, results (polls /api/runs/<id>)
//   report  /featured/<slug>     a finished run's full report (report data from /api/featured/<slug>/report
//           /run/<id>/report     or /api/runs/<id>/report.json; the original HTML stays one link away)
// Views render into #app. Anything a view starts (timers, animation loops) is registered with onLeave()
// and stopped on the next navigation; async work checks isCurrent() before touching the page.
// demo/build_static.py sets window.HERON_STATIC for the hosted, server-less page: finished runs only, no live
// runs, and a contact note where the upload would be.

const STATIC = window.HERON_STATIC || null;
const app = document.getElementById("app");
const $ = (sel, root = document) => root.querySelector(sel);

// ------------------------------------------------------------------ formatting

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);
const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const fmt = (x, digits = 1) => (isNum(x) ? x.toFixed(digits) : "—");
const usd = (x) => (isNum(x) ? `$${x.toFixed(x < 0.1 ? 3 : 2)}` : "—");
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");
const METRIC_NAMES = { accuracy: "accuracy", macro_f1: "macro-F1", agreement: "agreement", field_accuracy: "field accuracy" };
const metricName = (m) => METRIC_NAMES[m] || m || "";

function modelName(row) {
  const model = row.model || "";
  if (row.kind === "tuned") {
    const size = model.match(/(\d+(?:\.\d+)?B)\b/);
    return `Fine-tuned ${size ? size[1] : "model"}`;
  }
  return model.replace("gpt-6-luna", "GPT").replace("(pseudo-baseline)", "").trim();
}

const tunedRow = (run) => (run.comparison || []).find((r) => r.kind === "tuned");
const bestBaseline = (run) =>
  (run.comparison || []).filter((r) => r.kind === "baseline").reduce((a, c) => (!a || c.primary > a.primary ? c : a), null);

async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || body.detail || `request failed (${res.status})`);
  return body;
}

// ------------------------------------------------------------------ routing and view lifecycle

let viewId = 0;
let viewPath = null;
let cleanups = [];
const onLeave = (fn) => cleanups.push(fn);
const isCurrent = (id) => id === viewId;

function route() {
  cleanups.forEach((fn) => fn());
  cleanups = [];
  const id = ++viewId;
  const path = location.pathname;
  const run = STATIC ? null : path.match(/^\/run\/(demo-[0-9a-f]{8})(\/report)?$/);
  const featured = path.match(/^\/featured\/([a-z0-9-]+)$/);
  if (!run && !featured && path !== "/") history.replaceState({}, "", "/" + location.hash);  // unknown path → home
  viewPath = location.pathname;
  if (featured) reportView({ data: `/api/featured/${featured[1]}/report`, raw: `/featured/${featured[1]}/raw`, slug: featured[1] }, id);
  else if (run && run[2]) reportView({ data: `/api/runs/${run[1]}/report.json`, raw: `/api/runs/${run[1]}/report`, back: `/run/${run[1]}` }, id);
  else if (run) runView(run[1], id);
  else homeView(id);
}

// A new view starts at the top at once (the page's smooth scrolling is for in-page #section links only).
const toTop = () => window.scrollTo({ top: 0, behavior: "instant" });

function go(path) {
  history.pushState({}, "", path);
  route();
  toTop();
}

// popstate also fires for in-page #section links; those only scroll (the browser does that), so only a
// change of path re-renders the view.
window.addEventListener("popstate", () => { if (location.pathname !== viewPath) route(); });
document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-link]");
  if (link) { e.preventDefault(); go(link.dataset.link); }
});

// ------------------------------------------------------------------ shared pieces

// 10×10 dot matrix, one dot per percentage point; dots fill in (staggered) when their card is revealed.
function dots(value, tuned = false) {
  const on = Math.round(Math.max(0, Math.min(100, value || 0)));
  const cells = Array.from({ length: 100 }, (_, i) => `<i class="${i < on ? "on" : ""}" style="--i:${i}"></i>`).join("");
  return `<div class="dots${tuned ? " tuned" : ""}">${cells}</div>`;
}

const comparisonGrid = (rows) => `<div class="cmp">${rows.map((r) => `
  <div class="${r.kind === "tuned" ? "tuned" : ""}">${dots(r.primary, r.kind === "tuned")}
    <div class="who">${esc(modelName(r))}</div><div class="val">${fmt(r.primary)}</div></div>`).join("")}</div>`;

const errorBox = (message) => `<div class="err">${esc(message)}</div>`;

function observeReveals() {
  const reveal = (el) => {
    el.classList.add("in");
    el.querySelectorAll("[data-to]").forEach(countUp);
  };
  const els = document.querySelectorAll(".reveal:not(.in)");
  if (!("IntersectionObserver" in window)) { els.forEach(reveal); return; }
  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); }
  }), { threshold: 0.18 });
  els.forEach((el) => io.observe(el));
  onLeave(() => io.disconnect());
}

function countUp(el) {
  const to = Number(el.dataset.to);
  const digits = Number(el.dataset.digits || 1);
  if (!Number.isFinite(to)) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { el.textContent = to.toFixed(digits); return; }
  const t0 = performance.now(), duration = 1100;
  const step = (now) => {
    const p = Math.min(1, (now - t0) / duration);
    el.textContent = (to * (1 - Math.pow(1 - p, 3))).toFixed(digits);
    if (p < 1 && el.isConnected) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ------------------------------------------------------------------ home view

const STEPS = [
  ["Reads the data", "Profiles the columns, samples rows, finds the input and the label, and checks there's enough to learn from and grade."],
  ["Plans the run", "Writes the config and a cost-and-time estimate. Rules in code fix the test size, baselines and budget. You approve."],
  ["Asks GPT first", "Zero-shot and few-shot GPT answer the same held-out questions, so there's a real bar to beat."],
  ["Fine-tunes", "LoRA on a 4B model on real GPUs. If it misses the bar, a fixer agent reads the diagnosis and tries again."],
  ["Benchmarks", "Paired bootstrap on test: accuracy, macro-F1, latency, cost per 1,000 requests. Then the GPU is torn down."],
];

// The hero figure's three labels; their x positions in style.css match the columns drawn by Art.convergence.
const FLOW = [
  ["data", "Your data", "csv · jsonl · logs"],
  ["agent", "The agent", "profile · plan · rules in code"],
  ["model", "A small model", "qwen3.5-4b · lora"],
];

const SAMPLES = [
  ["banking77", "Banking support intents · 3,000 rows · 77 labels"],
  ["too-small", "A 60-row file · watch the agent decline it"],
];

async function homeView(id) {
  app.innerHTML = `
    <section class="hero">
      <div class="frame">
        <div class="hero-figure">
          <canvas id="flow" aria-hidden="true"></canvas>
          <div class="corner tl lbl">Fig. 01 <span class="muted">· messy data in, one measured model out</span></div>
          <div class="corner tr lbl muted">series.077 nr.001</div>
          ${FLOW.map(([key, label, caption]) => `<span class="flow-label ${key}">${label}</span><span class="flow-caption ${key} lbl muted">${caption}</span>`).join("")}
        </div>
        <div class="hero-top">
          <div class="reveal">
            <span class="chip"><span class="pulse"></span>Automated LLM fine-tuning</span>
            <h1>Heron<i>.</i></h1>
          </div>
          <div class="reveal delayed">
            <p class="deck">Hand it a labeled dataset. An agent plans the fine-tune, trains a small open model on real GPUs, repairs its own failed runs, and proves, with confidence intervals, whether it beats GPT.</p>
            <div class="ctas"><a class="btn dark" href="#try">${STATIC ? "Run it on your data" : "Upload a dataset"} <span class="arr">→</span></a><a class="btn" href="#runs">See real runs</a></div>
            <div class="trust" id="trust"></div>
          </div>
        </div>
        <div class="hero-foot">
          <div class="hero-replay">
            <span class="lbl muted">A real agent trace, replayed · run proof-banking77</span>
            <div class="result-chip" id="chip"><b>+6.4 pts</b><span>BANKING77 · fine-tuned 4B vs GPT</span></div>
          </div>
          <div class="console"><div class="bar"><i></i><i></i><i></i><span>replay · run proof-banking77</span></div><div class="body" id="console"></div></div>
        </div>
      </div>
    </section>
    <section class="strip"><div class="frame">
      <span class="lbl muted">Run on</span>
      ${["BANKING77 · 77 intents", "LEDGAR · 100 clause types", "Enron email · time split", "CORD receipts · declined", "Qwen3.5-4B vs gpt-6-luna"]
        .map((p) => `<span class="pill">${p}</span>`).join("")}
    </div></section>

    <section class="plate" id="results">
      <div class="frame">
        <div class="plate-head reveal"><span class="lbl">Plate II · measured, not claimed</span>
          <h2>What it <i>did</i>.</h2><p>Each dot is one percentage point on a held-out test set. Blue: the fine-tuned model. Black: GPT.</p></div>
        <figure class="figure reveal">
          <canvas id="net" aria-hidden="true"></canvas>
          <figcaption class="corner tl lbl">Fig. 02 <span class="muted">· a dataset, before and after fine-tuning</span></figcaption>
          <div class="corner tr lbl muted">one color per learned class</div>
        </figure>
        <div class="specimens" id="specimens"></div>
      </div>
    </section>

    ${STATIC ? contactPlate() : uploadPlate()}

    <section class="plate method" id="method">
      <div class="frame">
        <div class="plate-head reveal"><span class="lbl">Plate IV · the trajectory of a run</span>
          <h2>Agent plans, <i>code executes</i>.</h2></div>
        <div class="card method-card reveal">
          ${trajectory()}
          <div class="steps">${STEPS.map(([t, d], i) => `<div><span class="lbl muted">0${i + 1}</span><h4>${t}</h4><p>${d}</p></div>`).join("")}</div>
        </div>
      </div>
    </section>

    <section class="plate" id="runs">
      <div class="frame">
        <div class="plate-head reveal"><span class="lbl">Plate V · finished runs, zero human edits</span>
          <h2>Real <i>runs</i>.</h2><p>Run end to end by this pipeline. Numbers on held-out test sets; click through for each full report.</p></div>
        <div class="plates" id="featured"><div class="card run-plate muted">Loading…</div></div>
      </div>
    </section>`;

  onLeave(Art.convergence($("#flow")));
  onLeave(Art.network($("#net"), { cx: 0.5, cy: 0.54, r: 0.4, stretch: 1.9 }));
  onLeave(replayConsole($("#console"), $("#chip")));
  if (!STATIC) wireUpload();
  observeReveals();

  if (!STATIC) api("/api/status").then((s) => { if (isCurrent(id)) $("#limits").textContent = limitsText(s); }).catch(() => {});

  const runs = await api("/api/featured").catch(() => []);
  if (!isCurrent(id)) return;
  $("#trust").innerHTML = trustRow(runs);
  $("#specimens").innerHTML = specimens(runs);
  $("#featured").innerHTML = runs.length ? runs.map(runPlate).join("") : `<div class="card run-plate muted">No featured runs yet.</div>`;
  observeReveals();
  scrollToHash();  // only now is the page at its final height, so a #section link can land
}

const uploadPlate = () => `
    <section class="plate" id="try">
      <div class="frame try"><div class="card try-card reveal">
        <div>
          <span class="lbl">Plate III <span class="star">*</span> try it</span>
          <h3>Bring a <i>dataset</i>.<br>Leave with a model.</h3>
          <p>Any CSV or JSONL with a text column and a label column. The agent works out which is which, or tells you why fine-tuning isn't a good fit.</p>
          <div class="samples">${SAMPLES.map(([key, label]) => `<button class="btn" data-sample="${key}"><span class="arr">→</span> ${label}</button>`).join("")}</div>
          <p class="fine" id="limits"></p>
          <div id="err"></div>
        </div>
        <div class="drop" id="drop" role="button" tabindex="0" aria-label="Upload a CSV or JSONL file">
          ${dropDiagram()}
          <div class="core"><b>Drop a dataset here</b><span class="lbl muted">.csv · .jsonl · or click</span></div>
          <input type="file" id="file" accept=".csv,.jsonl" hidden>
        </div>
      </div></div>
    </section>`;

// The hosted page offers runs by email.
function contactPlate() {
  const mail = `mailto:${esc(STATIC.contact)}?subject=${encodeURIComponent("Run Heron on my data")}`;
  return `
    <section class="plate" id="try">
      <div class="frame try"><div class="card try-card reveal">
        <div>
          <span class="lbl">Plate III <span class="star">*</span> run it</span>
          <h3>Got a <i>dataset</i>?</h3>
          <div class="samples"><a class="btn dark address" href="${mail}">${esc(STATIC.contact)}<span class="arr">→</span></a></div>
        </div>
        <a class="drop" href="${mail}" aria-label="Email about running Heron on your data">
          ${dropDiagram()}
          <div class="core"><b>Email me</b><span class="lbl muted">a dataset · or just a question</span></div>
        </a>
      </div></div>
    </section>`;
}

// The browser tries to scroll to #section before this script has drawn it, so land there once the view is
// complete, and again whenever the page height changes during the next few seconds (late web fonts reflow
// the text above the target). Stops as soon as the visitor scrolls themselves.
const SETTLE_MS = 3000;
function scrollToHash() {
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!target) return;
  const land = () => target.isConnected && target.scrollIntoView({ behavior: "instant", block: "start" });
  land();
  const ro = new ResizeObserver(land);
  const done = () => {
    ro.disconnect();
    clearTimeout(timer);
    USER_SCROLL_EVENTS.forEach((ev) => window.removeEventListener(ev, done));
  };
  const timer = setTimeout(done, SETTLE_MS);
  ro.observe(document.body);
  USER_SCROLL_EVENTS.forEach((ev) => window.addEventListener(ev, done, { once: true, passive: true }));
  onLeave(done);
}
const USER_SCROLL_EVENTS = ["wheel", "touchstart", "keydown", "mousedown"];

function trustRow(runs) {
  const b77 = runs.find((r) => r.slug === "banking77"), ledgar = runs.find((r) => r.slug === "ledgar");
  const tuned = b77 && tunedRow(b77), base = b77 && bestBaseline(b77);
  if (!tuned || !base) return "";
  const items = [
    [`${fmt(tuned.accuracy)}%`, `vs GPT ${fmt(base.accuracy)}% · BANKING77`],
    [`${fmt(base.p50_ms / tuned.p50_ms)}×`, "faster responses"],
  ];
  if (ledgar) items.push([`+${fmt(ledgar.diff)} pts`, "on LEDGAR, 95% CI excludes 0"]);
  return items.map(([n, c]) => `<div><b>${n}</b>${c}</div>`).join("");
}

function specimens(runs) {
  const by = Object.fromEntries(runs.map((r) => [r.slug, r]));
  const card = (label, big, caption, art) =>
    `<div class="card lift specimen reveal"><span class="lbl">${label}</span><div class="big">${big}</div><div class="cap">${caption}</div><div class="art">${art}</div></div>`;
  const counter = (value, digits = 1) => `<span data-to="${value}" data-digits="${digits}">0</span>`;
  const pair = (left, leftLabel, right, rightLabel) =>
    `<div class="pair"><div>${left}<div class="lbl">${leftLabel}</div></div><div>${right}<div class="lbl">${rightLabel}</div></div></div>`;
  const out = [];

  const b77 = by.banking77, t = b77 && tunedRow(b77), b = b77 && bestBaseline(b77);
  if (t && b) {
    out.push(card("No. 1 · BANKING77", `${counter(t.accuracy)}<small>% vs ${fmt(b.accuracy)}%</small>`,
      "Test accuracy on 77 banking intents: the fine-tuned 4B model against few-shot GPT.",
      pair(dots(t.primary, true), "Fine-tuned", dots(b.primary), "GPT")));
    const speed = b.p50_ms / t.p50_ms, n = Math.round(speed);
    if (isNum(speed)) {
      out.push(card("No. 2 · latency", `${counter(speed)}<small>× faster</small>`,
        `Median response ${fmt(t.p50_ms, 0)} ms against GPT's ${fmt(b.p50_ms, 0)} ms, at about half the cost per request.`,
        `<div class="dots sq static" style="--cols:${n + 1}"><i class="on tuned-dot"></i>${"<i class='on'></i>".repeat(n)}</div>
         <div class="lbl dots-caption">1 blue : ${n} black = time per answer</div>`));
    }
  }
  const led = by.ledgar, lt = led && tunedRow(led), lb = led && bestBaseline(led);
  if (lt && lb && led.diff_ci95) {
    out.push(card("No. 3 · LEDGAR", `+${counter(led.diff)}<small> pts</small>`,
      `100 contract-clause types, a dataset it had never seen. 95% CI [+${fmt(led.diff_ci95[0])}, +${fmt(led.diff_ci95[1])}].`,
      pair(dots(lt.primary, true), "Fine-tuned", dots(lb.primary), "GPT")));
  }
  const heal = by["self-healing"];
  if (heal && heal.attempts?.length > 1) {
    const first = heal.attempts[0], last = heal.attempts.at(-1);
    out.push(card("No. 4 · self-healing", `${fmt(first.val)}<small> → </small>${fmt(last.val)}`,
      "A deliberately broken config. The fixer agent read the diagnosis and repaired it in one retry.",
      pair(dots(first.val), "Attempt 1", dots(last.val, true), "Attempt 2")));
  }
  return out.join("");
}

function verdictTag(word) {
  if (word === "beat") return `<span class="tag accent">beat GPT</span>`;
  if (word === "fell short") return `<span class="tag">fell short</span>`;
  return `<span class="tag">${esc(word || "")} · not significant</span>`;
}

const runPlate = (r, i) => `
  <article class="card lift run-plate reveal">
    <div class="meta"><span class="lbl">Run 0${i + 1} <span class="star">*</span></span>
      <span class="lbl muted">${esc(metricName(r.primary_metric))} · ${(r.n_test || 0).toLocaleString()} test</span></div>
    <h3>${esc(r.title)}</h3>
    <div class="story">${esc(r.story)}</div>
    ${comparisonGrid(r.comparison || [])}
    <div class="verdict">${esc(r.sentence)}</div>
    <div class="foot"><span>${verdictTag(r.word)}<span class="tag">${esc(r.interventions)} human edits</span><span class="tag">${usd(r.spend_usd)}</span></span>
      <a class="btn" href="${esc(r.report)}" data-link="${esc(r.report)}">Full report <span class="arr">→</span></a></div>
  </article>`;

// A replay of the real agent trace from run proof-banking77 (agent/transcript.jsonl), then its result.
const TRACE = [
  ["› get_profile()", "data/banking77 · 2 tables · 13,083 rows"],
  ["› sample_rows(train.csv, n=6)", "text → category"],
  ["› value_counts(category)", "77 classes, top class 1.9%"],
  ["› sample_rows(test.csv)", "✗ refused: test files are held out"],
  ["› list_base_models(text)", "Qwen3.5-4B · Qwen3.5-9B"],
  ["› estimate_cost()", "✗ invalid YAML · agent fixes it"],
  ["› validate_task()", "✓ valid · rule: test_max → 3,080", "ok"],
  ["› submit_task()", "✓ plan approved", "ok"],
  ["  baselines", "GPT zero-shot + 1-shot on 4,080 Qs"],
  ["  train", "LoRA · Qwen3.5-4B · 3 epochs · 14 min"],
  ["  evaluate", "3,080 held-out messages"],
  ["✓ 92.4% vs 86.0%", "beat GPT by +6.4", "win"],
];
const CONSOLE_LINES = 9, LINE_MS = 750, SLOW_LINE_MS = 1100, HOLD_MS = 5200;

function replayConsole(el, chip) {
  let timer = 0, i = 0;
  const line = ([cmd, out, kind]) =>
    `<div class="ln ${kind || ""}"><span class="cmd">${esc(cmd)}</span> <span class="out">${esc(out)}</span></div>`;
  const restart = () => { i = 0; el.innerHTML = ""; chip.classList.remove("on"); next(); };
  const next = () => {
    if (i >= TRACE.length) { chip.classList.add("on"); timer = setTimeout(restart, HOLD_MS); return; }
    el.insertAdjacentHTML("beforeend", line(TRACE[i++]));
    while (el.children.length > CONSOLE_LINES) el.firstElementChild.remove();
    timer = setTimeout(next, i < CONSOLE_LINES ? LINE_MS : SLOW_LINE_MS);
  };
  restart();
  return () => clearTimeout(timer);
}

// The drop zone: a dotted circle on crosshairs, two tilted orbits carrying points (the glow is CSS).
function dropDiagram() {
  const cx = 65, cy = 50;
  const orbits = [[60, 17, -16, [0.4, 2.6, 4.4]], [46, 30, 28, [1.2, 3.9]]];
  const point = (rx, ry, deg, t) => {
    const a = (deg * Math.PI) / 180, x = rx * Math.cos(t), y = ry * Math.sin(t);
    return [cx + x * Math.cos(a) - y * Math.sin(a), cy + x * Math.sin(a) + y * Math.cos(a)];
  };
  const drawn = orbits.map(([rx, ry, deg, at]) =>
    `<ellipse class="orbit" cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" transform="rotate(${deg} ${cx} ${cy})"/>` +
    at.map((t) => { const [x, y] = point(rx, ry, deg, t); return `<circle class="orbit-dot" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="1.4"/>`; }).join(""),
  ).join("");
  return `<svg viewBox="0 0 130 100" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    <line class="axis" x1="0" y1="${cy}" x2="130" y2="${cy}"/><line class="axis" x1="${cx}" y1="0" x2="${cx}" y2="100"/>
    <circle class="ring" cx="${cx}" cy="${cy}" r="34"/>
    <g class="orbits">${drawn}</g>
  </svg>`;
}

// The five steps as a transit map: one colored line per leg, ring stations, and the retry loop as a branch
// from Fine-tunes back to Asks GPT first's bar. Station x positions line up with the five step columns below.
const STATIONS = [[101, 120, "profile"], [330, 120, "plan"], [560, 64, "baselines"], [790, 64, "train"], [1019, 120, "benchmark"]];
function trajectory() {
  const legs = [
    "M 101 120 H 330",
    "M 330 120 H 418 L 474 64 H 560",
    "M 560 64 H 790",
    "M 790 64 H 876 L 932 120 H 1019",
  ];
  const lines = legs.map((d, i) => `<path class="line l${i + 1} draw" d="${d}"/>`).join("");
  const retry = `<path class="line ghost draw" d="M 766 64 L 730 24 H 620 L 584 64"/>`;
  const stations = STATIONS.map(([x, y, name], i) =>
    `<circle class="station" cx="${x}" cy="${y}" r="11"/><text class="station-label" x="${x}" y="${y + 38}">0${i + 1} ${name}</text>`).join("");
  return `<svg viewBox="0 0 1120 170" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
    ${lines}${retry}${stations}
    <text class="station-label muted" x="672" y="14">retry, if it misses the bar</text>
  </svg>`;
}

function wireUpload() {
  const drop = $("#drop"), input = $("#file");
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const file = e.dataTransfer.files[0];
    if (file) startRun({ file });
  });
  input.addEventListener("change", () => input.files[0] && startRun({ file: input.files[0] }));
  document.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => startRun({ sample: b.dataset.sample })));
}

async function startRun({ file, sample }) {
  const err = $("#err");
  err.innerHTML = "";
  const body = new FormData();
  if (file) body.append("file", file);
  else body.append("sample", sample);
  try {
    const { id } = await api("/api/runs", { method: "POST", body });
    go(`/run/${id}`);
  } catch (e) {
    err.innerHTML = errorBox(e.message);
  }
}

// ------------------------------------------------------------------ run view

const STAGE_LABELS = { load: "Load", split: "Split", baseline: "Ask GPT", prepare: "Prepare", train: "Fine-tune", predict: "Deploy + predict", evaluate: "Score", report: "Report" };
const STATUS_LABELS = { planning: "The agent is reading", planned: "Waiting for approval", declined: "Declined", queued: "In the queue", running: "Running", done: "Done", failed: "Stopped" };
const ORBIT_CAPTIONS = { planning: "reading the data", planned: "idle until you approve", queued: "waiting for the GPU", running: "training · evaluating" };
const BUSY = new Set(["planning", "queued", "running"]);
const FINISHED = new Set(["done", "failed", "declined"]);
const POLL_MS = 3000, POLL_IDLE_MS = 8000;

function runView(runId, id) {
  app.innerHTML = `
    <div class="frame">
      <div class="run-head" id="head"><span class="lbl muted">Loading…</span></div>
      <div class="run-grid">
        <div class="card run-side">
          <span class="lbl">Trajectory</span>
          <ul class="track" id="track"></ul>
          <canvas class="vortex" id="vortex" aria-hidden="true"></canvas>
          <div class="lbl muted orbit-caption" id="orbit-caption"></div>
          <div id="attempts"></div>
        </div>
        <div class="card" id="main"></div>
      </div>
      <div class="log-wrap"><span class="lbl">Live log</span><pre class="log" id="log"></pre></div>
    </div>`;
  const orbit = $("#vortex");
  onLeave(Art.vortex(orbit));

  let timer = 0, mainKey = "";
  onLeave(() => clearTimeout(timer));

  $("#main").addEventListener("click", async (e) => {
    const btn = e.target.closest("#approve");
    if (!btn) return;
    btn.disabled = true;
    try {
      await api(`/api/runs/${runId}/approve`, { method: "POST" });
      mainKey = "";
      poll();
    } catch (err) {
      btn.disabled = false;
      $("#approve-error").innerHTML = errorBox(err.message);
    }
  });

  async function poll() {
    clearTimeout(timer);
    let run;
    try {
      run = await api(`/api/runs/${runId}`);
    } catch (err) {
      if (isCurrent(id)) $("#main").innerHTML = errorBox(err.message);
      return;
    }
    if (!isCurrent(id)) return;
    renderHead(run);
    renderTrack(run);
    renderSide(run, orbit);
    // the main column only re-renders when what it shows changes, so scrolling and focus survive polling
    const key = `${run.status}|${Boolean(run.results)}|${Boolean(run.plan)}`;
    if (key !== mainKey) { $("#main").innerHTML = mainColumn(run); mainKey = key; }
    renderLog(run.log || []);
    if (!FINISHED.has(run.status)) timer = setTimeout(poll, run.status === "planned" ? POLL_IDLE_MS : POLL_MS);
  }
  poll();
}

function renderHead(run) {
  const tone = BUSY.has(run.status) ? "busy" : run.status === "done" ? "ok" : "";
  $("#head").innerHTML = `
    <div><span class="lbl">Run · ${esc(run.id)}${run.dry_run ? ` <span class="muted">· preview, simulated models</span>` : ""}</span>
      <h1>${esc(run.filename)}<i>.</i></h1></div>
    <div class="run-status"><span class="status-pill ${tone}"><span class="dot"></span>${esc(STATUS_LABELS[run.status] || run.status)}</span>
      <div class="lbl muted">${(run.rows || 0).toLocaleString()} rows</div></div>`;
}

function renderTrack(run) {
  const stages = run.progress?.length ? run.progress : Object.keys(STAGE_LABELS).map((stage) => ({ stage, status: "pending" }));
  const firstPending = run.status === "running" ? stages.findIndex((p) => p.status !== "done") : -1;
  const item = (cls, label, note) => `<li class="${cls}"><span class="node"></span><span>${label}</span><span>${note}</span></li>`;
  const rows = [item(run.status === "planning" ? "running" : "done", "Agent plans", run.agent_turns ? `${run.agent_turns} turns` : "")];
  stages.forEach((p, i) => {
    let cls = p.status === "done" || p.status === "failed" ? p.status : i === firstPending ? "running" : "";
    if (run.status === "declined") cls = "";
    rows.push(item(cls, esc(STAGE_LABELS[p.stage] || p.stage), p.attempt ? `a${p.attempt}` : ""));
  });
  $("#track").innerHTML = rows.join("");
}

function renderSide(run, orbit) {
  orbit.hidden = FINISHED.has(run.status);
  orbit.dataset.speed = BUSY.has(run.status) ? "1.6" : "0.35";  // fast while working, slow while waiting
  $("#orbit-caption").textContent = ORBIT_CAPTIONS[run.status] || "";
  const attempts = (run.attempts || []).map((a) => `<li><span class="k">Attempt ${a.attempt}</span><span>${fmt(a.val)}
    <span class="muted">/ bar ${fmt(a.threshold)}</span> <span class="${a.passed ? "accent" : "muted"}">${a.passed ? "pass" : "miss"}</span></span></li>`);
  const fixes = (run.fixes || []).map((f) => `<li class="rule"><span class="k">Fixer</span><span>${esc(f.replace(/^fixer:\s*/, ""))}</span></li>`);
  $("#attempts").innerHTML = attempts.length || fixes.length ? `<ul class="why side">${attempts.join("")}${fixes.join("")}</ul>` : "";
}

function renderLog(lines) {
  const log = $("#log");
  log.innerHTML = lines.map((l) => `<span>${esc(l) || " "}</span>`).join("");
  log.scrollTop = log.scrollHeight;
}

function mainColumn(run) {
  const parts = [];
  if (run.results) parts.push(resultsBlock(run));
  if (run.status === "planning") {
    parts.push(`<div class="sec"><h3>The agent is reading your data</h3>
      <p class="muted">It profiles the columns, looks at sample rows, checks the labels and drafts a plan. Usually under a minute.</p></div>`);
  } else if (run.status === "declined") {
    parts.push(`<div class="sec"><h3>${esc(cap(run.fit_label || "declined"))}.</h3><p class="big-verdict reason">${esc(cap(run.reason))}</p>
      <ul class="why"><li><span class="k">Suggestion</span><span>${esc(cap(run.suggestion))}</span></li></ul>
      <button class="btn" data-link="/"><span class="arr">←</span> Try another dataset</button></div>`);
  } else if (run.status === "failed") {
    parts.push(`<div class="sec"><h3>The run stopped.</h3>${errorBox(run.error)}<p class="fine">Any GPU endpoint was torn down; nothing is left running.</p></div>`);
  }
  if (run.plan && !run.results) parts.push(planBlock(run));
  return parts.join("");
}

function planBlock(run) {
  const p = run.plan, est = p.estimate || {};
  const lines = (est.lines || []).map((l) => `<tr><td>${esc(l.item)}</td><td class="n">${usd(l.usd)}</td><td class="n">${l.minutes ?? "—"} min</td></tr>`).join("");
  const why = Object.entries(run.rationale || {}).map(([k, v]) => `<li><span class="k">${esc(k.replace("_", " "))}</span><span>${esc(cap(v))}</span></li>`).join("");
  const rules = (run.corrections || []).map((c) => `<li class="rule"><span class="k">Rule applied</span><span>${esc(c)}</span></li>`).join("");
  const examples = (p.examples || []).slice(0, 6).map((e) => `<tr><td class="text">${esc(e.input)}</td><td>${esc(e.expected)}</td></tr>`).join("");
  const approve = run.status === "planned" ? `
    <div class="approve"><span class="cost">${usd(est.total_usd)}<small>~${est.total_minutes ?? "?"} min</small></span>
      <button class="btn dark" id="approve">Approve and fine-tune <span class="arr">→</span></button>
      <span class="fine">Starts a real training job; you'll watch it here.</span></div><div id="approve-error"></div>` : "";
  return `
    <div class="sec"><h3>The agent's plan <small>checkpoint</small></h3>
      <dl class="spec">
        <dt>Task</dt><dd>${esc(p.task)}</dd>
        <dt>Labels</dt><dd>${esc(p.labels ?? "—")} classes, from ${esc(p.label_source)}</dd>
        <dt>Splits</dt><dd>${(p.splits?.train ?? 0).toLocaleString()} train · ${esc(p.splits?.val ?? 0)} val · ${esc(p.splits?.test ?? 0)} test <span class="muted">(${esc(p.split_method)})</span></dd>
        <dt>Contest</dt><dd>${esc(p.base_model)} against ${(p.baselines || []).map(esc).join(", ")}</dd>
        <dt>Metric</dt><dd>${esc(metricName(p.metric))}</dd>
      </dl></div>
    ${approve}
    <div class="sec"><h3>Estimate</h3><div class="scroll"><table><tr><th>Step</th><th class="n">Cost</th><th class="n">Time</th></tr>${lines}
      <tr class="total"><td>Total</td><td class="n">${usd(est.total_usd)}</td><td class="n">${est.total_minutes ?? "—"} min</td></tr></table></div></div>
    <div class="sec"><h3>Why</h3><ul class="why">${why}${rules}</ul></div>
    ${examples ? `<div class="sec"><h3>From the test set <small>held out</small></h3><div class="scroll"><table><tr><th>Input</th><th>Expected label</th></tr>${examples}</table></div></div>` : ""}`;
}

function resultsBlock(run) {
  const rows = run.results.comparison || [];
  const sentence = esc(run.results.verdict?.sentence || "").replace(/(beat [^,(]+? by [\d.]+ points)/, "<b>$1</b>");
  const table = rows.map((r) => `<tr class="${r.kind === "tuned" ? "tuned" : ""}"><td>${esc(modelName(r))}</td><td class="n">${fmt(r.accuracy)}</td>
    <td class="n">${fmt(r.macro_f1)}</td><td class="n">${fmt(r.p50_ms, 0)} ms</td><td class="n">${usd(r.cost_per_1k)}</td></tr>`).join("");
  return `
    <div class="sec"><span class="lbl">Result <span class="star">***</span></span><p class="big-verdict">${sentence}</p></div>
    <div class="sec">${comparisonGrid(rows)}</div>
    <div class="sec"><div class="scroll"><table><tr><th>Model</th><th class="n">Accuracy</th><th class="n">Macro-F1</th><th class="n">Median latency</th><th class="n">$ / 1k req</th></tr>${table}</table></div>
      <div class="result-actions"><a class="btn" href="/run/${esc(run.id)}/report" data-link="/run/${esc(run.id)}/report">Full report <span class="arr">→</span></a>
      <span class="fine">Spent on this run: ${usd(run.spent_usd)}</span></div></div>`;
}

// ------------------------------------------------------------------ report view

async function reportView(src, id) {
  app.innerHTML = `<div class="frame report"><div class="run-head"><span class="lbl muted">Loading report…</span></div></div>`;
  let report, meta;
  try {
    [report, meta] = await Promise.all([
      api(src.data),
      src.slug ? api("/api/featured").then((all) => all.find((r) => r.slug === src.slug)).catch(() => null) : null,
    ]);
  } catch (err) {
    if (isCurrent(id)) app.innerHTML = `<div class="frame report"><div class="run-head">${errorBox(err.message)}</div></div>`;
    return;
  }
  if (!isCurrent(id)) return;
  const r = report;
  app.innerHTML = `
    <div class="frame report">
      ${reportHead(r, meta, src)}
      ${reportVerdict(r)}
      ${reportSection("Models", "comparison on the held-out test set", reportComparison(r))}
      ${r.attempts?.length ? reportSection("Attempts", "scored on validation; the best one ships", reportAttempts(r)) : ""}
      ${reportErrors(r)}
      ${reportSection("Setup", "data, training and the agent's reasoning", reportSetup(r))}
      <div class="report-pair">
        ${reportSection("Spend", "everything this run cost", reportSpend(r))}
        ${reportSection("Human edits", "after approval", reportInterventions(r))}
      </div>
      ${r.privacy ? reportSection("Privacy", "what left this machine", reportPrivacy(r.privacy)) : ""}
      ${reportSection("Run metadata", "for the record", reportMetadata(r))}
    </div>`;
  observeReveals();
  toTop();
}

const reportSection = (title, sub, body) => body ? `
  <section class="report-sec reveal"><div class="report-sec-head"><h2>${title}</h2><span class="lbl muted">${sub}</span></div>${body}</section>` : "";

function reportHead(r, meta, src) {
  const tuned = (r.comparison || []).find((c) => c.kind === "tuned");
  const title = meta?.title || cap(r.task?.description || r.task?.name || "Run report");
  const back = src.back ? `<a class="btn" href="${src.back}" data-link="${src.back}"><span class="arr">←</span> Live run page</a>`
    : `<a class="btn" href="/#runs"><span class="arr">←</span> All runs</a>`;
  return `
    <div class="run-head report-head">
      <div><span class="lbl">Report · ${esc(r.metadata?.run_id || "")} · ${esc((r.generated_at || "").slice(0, 10))}</span>
        <h1>${esc(title)}</h1>
        <p class="report-story">${esc(meta?.story || r.task?.description || "")}</p>
        <div class="report-tags">${verdictTag(r.verdict?.word)}<span class="tag">${esc(r.interventions?.count ?? 0)} human edits</span>
          <span class="tag">${usd(r.spend?.total_usd)} total</span>${tuned ? `<span class="tag">${esc(modelName(tuned))}</span>` : ""}
          ${r.dry_run ? `<span class="tag">simulated</span>` : ""}</div></div>
      <div class="report-actions">${back}<a class="btn" href="${src.raw}" target="_blank" rel="noopener">Raw report <span class="arr">↗</span></a></div>
    </div>`;
}

function reportVerdict(r) {
  const v = r.verdict;
  if (!v) return "";
  const tuned = (r.comparison || []).find((c) => c.kind === "tuned");
  // the baseline the verdict sentence compares against (verdict.comparisons maps its name to its row label)
  const refLabel = (v.comparisons || []).find((c) => c.baseline === v.reference_model)?.label;
  const ref = (r.comparison || []).find((c) => c.kind === "baseline" && c.model === refLabel) || bestBaseline(r);
  const sentence = esc(v.sentence || "").replace(/((?:beat|fell short of|was ahead of|was behind) [^,(]+? by [\d.]+ points)/, "<b>$1</b>");
  const stats = [];
  if (tuned) stats.push([`${fmt(tuned.primary)}`, `fine-tuned ${metricName(r.primary_metric)}`, "tuned"]);
  if (ref) stats.push([`${fmt(ref.primary)}`, `${modelName(ref)} ${metricName(r.primary_metric)}`, ""]);
  if (isNum(v.diff) && v.diff_ci95) stats.push([`${v.diff > 0 ? "+" : ""}${fmt(v.diff)}`, `points, 95% CI [${fmt(v.diff_ci95[0])}, ${fmt(v.diff_ci95[1])}]`, ""]);
  if (tuned && ref && isNum(tuned.p50_ms) && isNum(ref.p50_ms)) stats.push([`${fmt(ref.p50_ms / tuned.p50_ms)}×`, `faster (${fmt(tuned.p50_ms, 0)} vs ${fmt(ref.p50_ms, 0)} ms)`, ""]);
  return `
    <section class="report-sec reveal">
      <span class="lbl">Verdict <span class="star">***</span></span>
      <p class="big-verdict">${sentence}</p>
      <p class="fine">${esc(v.headline)}</p>
      ${r.retrain?.attempt ? `<p class="fine">Shipped model retrained on train + val (attempt ${esc(r.retrain.attempt)}): test ${fmt(r.retrain.test)} vs ${fmt(r.retrain.base_test)} before retraining. Pass/fail stays with attempt ${esc(r.retrain.base_attempt)}'s validation score.</p>` : ""}
      <div class="report-stats">${stats.map(([n, c, cls]) => `<div class="card stat-card ${cls}"><b>${n}</b><span>${esc(c)}</span></div>`).join("")}</div>
    </section>`;
}

function reportComparison(r) {
  const rows = r.comparison || [];
  if (!rows.length) return "";
  const truth = r.truth_mode;
  // accuracy and macro-F1 columns, minus whichever one is already the primary metric
  const extra = [["accuracy", "Accuracy"], ["macro_f1", "Macro-F1"]].filter(([k]) => k !== r.primary_metric && !(k === "accuracy" && r.primary_metric === "agreement"));
  const table = rows.map((c) => `<tr class="${c.kind === "tuned" ? "tuned" : ""}">
      <td>${esc(modelName(c))}</td><td class="n">${fmt(c.primary)}</td><td class="n muted">${c.ci95 ? `${fmt(c.ci95[0])}–${fmt(c.ci95[1])}` : "—"}</td>
      ${truth ? `<td class="n">${fmt(c.truth_accuracy)}</td>` : ""}${extra.map(([k]) => `<td class="n">${fmt(c[k])}</td>`).join("")}
      <td class="n">${fmt(c.invalid_rate)}%</td><td class="n">${fmt(c.p50_ms, 0)}</td><td class="n">${fmt(c.p95_ms, 0)}</td>
      <td class="n">${usd(c.cost_per_1k)}</td><td class="n">${isNum(c.break_even_per_day) ? Math.round(c.break_even_per_day).toLocaleString() : "—"}</td></tr>`).join("");
  return `<div class="card report-card">${comparisonGrid(rows)}</div>
    <div class="scroll"><table class="report-table"><tr><th>Model</th><th class="n">${esc(metricName(r.primary_metric))}</th><th class="n">95% CI</th>
      ${truth ? `<th class="n">Truth acc.</th>` : ""}${extra.map(([, label]) => `<th class="n">${label}</th>`).join("")}<th class="n">Invalid</th>
      <th class="n">p50 ms</th><th class="n">p95 ms</th><th class="n">$ / 1k</th><th class="n">Break-even / day</th></tr>${table}</table></div>
    <p class="fine">Baseline cost is actual spend per request; fine-tuned cost assumes a fully used endpoint. Break-even is the requests per day at which a 24/7 endpoint costs the same as calling that baseline.</p>`;
}

function reportAttempts(r) {
  const items = r.attempts.map((a) => {
    const who = a.by_fixer ? `by the fixer, from attempt ${esc(a.built_on)}` : a.by_hand ? "by hand" : "starting config";
    const changes = a.changes?.length ? a.changes.map(esc).join(" · ") : a.n === 1 ? "as approved" : esc(a.patch);
    const reasons = (a.reasoning || "").split(/(?=fixer: )/).map((t) => t.replace(/^fixer:\s*/, "").trim()).filter(Boolean);
    return `<div class="card attempt ${a.final ? "final" : ""}">
      <div class="attempt-top"><span class="lbl">Attempt ${esc(a.n)}${a.final ? " · shipped" : ""}</span><span class="tag ${a.passed ? "accent" : ""}">${a.passed ? "pass" : "miss"}</span></div>
      <div class="attempt-nums"><div><b>${fmt(a.val)}</b><span>val (bar ${fmt(a.threshold)})</span></div><div><b>${fmt(a.test)}</b><span>test</span></div></div>
      <p class="muted">${who}: ${changes}</p>
      ${reasons.length ? `<ul class="why side">${reasons.map((t) => `<li class="rule"><span class="k">Why</span><span>${esc(t)}</span></li>`).join("")}</ul>` : ""}
    </div>`;
  }).join("");
  const stopped = r.retry_stopped ? `<p class="fine">The retry loop stopped: ${esc(r.retry_stopped)}.</p>` : "";
  return `<div class="attempts-grid">${items}</div>${stopped}`;
}

function reportErrors(r) {
  const e = r.errors || {};
  const parts = [];
  if (e.per_class?.length) {
    const worst = e.per_class.slice(0, 10);
    parts.push(`<div class="card report-card"><h3>Weakest classes <small>F1 on test</small></h3>
      <div class="barlist">${worst.map((c) => `<div class="barrow"><span class="name">${esc(c.label)}</span>
        <div class="bar"><i style="width:${Math.max(0, Math.min(100, c.f1))}%"></i></div><span class="v">${fmt(c.f1)}</span><span class="muted">${esc(c.support)}</span></div>`).join("")}</div></div>`);
  }
  if (e.confusions?.length) {
    parts.push(`<div class="card report-card"><h3>Most common mistakes <small>true → predicted</small></h3>
      ${e.confusions.slice(0, 6).map((c) => `<details class="confusion"><summary><span>${esc(c.true)} <span class="muted">→</span> ${esc(c.pred)}</span><b>${esc(c.count)}</b></summary>
        ${(c.examples || []).slice(0, 2).map((x) => `<p class="example">${esc(String(x).slice(0, 280))}${String(x).length > 280 ? "…" : ""}</p>`).join("")}</details>`).join("")}</div>`);
  }
  if (e.per_field?.length) {
    parts.push(`<div class="card report-card"><h3>Fields <small>accuracy per field</small></h3>
      <div class="barlist">${e.per_field.map((f) => `<div class="barrow"><span class="name">${esc(f.field)}</span><div class="bar"><i style="width:${Math.max(0, Math.min(100, f.accuracy ?? f.score ?? 0))}%"></i></div>
        <span class="v">${fmt(f.accuracy ?? f.score)}</span><span></span></div>`).join("")}</div></div>`);
  }
  return parts.length ? reportSection("Where it goes wrong", "fine-tuned model, test set", `<div class="report-pair">${parts.join("")}</div>`) : "";
}

function lossCurve(points) {
  if (!points?.length) return "";
  const w = 520, h = 120, max = Math.max(...points), min = Math.min(...points);
  const xy = points.map((y, i) => `${(i / Math.max(points.length - 1, 1)) * w},${h - ((y - min) / (max - min || 1)) * (h - 10) - 5}`).join(" ");
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="training loss curve"><polyline points="${xy}"/></svg>
    <div class="spark-caption"><span>loss ${fmt(points[0], 2)}</span><span>${points.length} steps</span><span>${fmt(points.at(-1), 3)}</span></div>`;
}

function reportSetup(r) {
  const s = r.setup || {}, t = r.task || {};
  const splits = s.splits ? `${(s.splits.train ?? 0).toLocaleString()} train · ${(s.splits.val ?? 0).toLocaleString()} val · ${(s.splits.test ?? 0).toLocaleString()} test` : "—";
  const dups = s.duplicates_dropped?.total ? ` · ${s.duplicates_dropped.total.toLocaleString()} duplicates dropped` : "";
  const baselines = (s.baselines || []).map((b) => `${b.model} (${b.reasoning_effort || "default"}${b.few_shot ? `, ${b.few_shot}-shot` : ", zero-shot"})`).join(", ");
  const rationale = (s.agent_rationale || []).map((x) => {
    const [k, ...rest] = String(x).split(": ");
    return rest.length ? `<li><span class="k">${esc(k)}</span><span>${esc(cap(rest.join(": ")))}</span></li>` : `<li><span class="k">note</span><span>${esc(x)}</span></li>`;
  }).join("");
  const warnings = (s.warnings || []).map((w) => `<li class="rule"><span class="k">Warning</span><span>${esc(w)}</span></li>`).join("");
  return `<div class="report-pair">
    <div class="card report-card"><h3>Data and training</h3><dl class="spec">
      <dt>Task</dt><dd>${esc(t.type)} · ${esc(t.input_kind)} → ${esc(t.n_labels ?? "")} labels</dd>
      <dt>Instruction</dt><dd>${esc(t.instruction)}</dd>
      <dt>Labels from</dt><dd>${esc(s.label_source)}</dd>
      <dt>Splits</dt><dd>${splits} <span class="muted">(${esc(s.split_method)})${dups}</span></dd>
      <dt>Model</dt><dd>${esc(s.base_model)} · LoRA · ${esc(s.epochs)} epoch${s.epochs === 1 ? "" : "s"} · lr ${esc(s.learning_rate)}${s.max_examples ? ` · ${s.max_examples.toLocaleString()} examples` : ""}</dd>
      <dt>Baselines</dt><dd>${esc(baselines)}</dd>
      <dt>Pass bar</dt><dd>${esc(r.verdict?.val_rule || "")}</dd>
    </dl></div>
    <div class="card report-card"><h3>Training loss</h3>${lossCurve(s.loss_curve) || `<p class="muted">No loss curve recorded.</p>`}
      ${rationale || warnings ? `<h3 class="sub">The agent's reasoning</h3><ul class="why">${rationale}${warnings}</ul>` : ""}</div>
  </div>`;
}

function reportSpend(r) {
  const s = r.spend || {};
  const stages = Object.entries(s.by_stage || {});
  const max = Math.max(...stages.map(([, v]) => v), 0.0001);
  return `<div class="card report-card"><div class="spend-total"><b>${usd(s.total_usd)}</b><span>of a ${usd(s.budget_usd)} budget${s.simulated ? " · simulated" : ""}</span></div>
    <div class="barlist">${stages.map(([k, v]) => `<div class="barrow"><span class="name">${esc(k)}</span><div class="bar"><i style="width:${(v / max) * 100}%"></i></div><span class="v">${usd(v)}</span><span></span></div>`).join("")}</div>
    <p class="fine">${Object.entries(s.by_provider || {}).map(([k, v]) => `${esc(k)} ${usd(v)}`).join(" · ")}</p></div>`;
}

function reportInterventions(r) {
  const iv = r.interventions || { count: 0, items: [] };
  const items = (iv.items || []).map((x) => `<li><span class="k">Edit</span><span>${esc(typeof x === "string" ? x : x.what || JSON.stringify(x))}</span></li>`).join("");
  return `<div class="card report-card"><div class="spend-total"><b>${esc(iv.count)}</b><span>human interventions after approval</span></div>
    ${items ? `<ul class="why">${items}</ul>` : `<p class="muted">The run went from approval to report with nobody touching it.</p>`}</div>`;
}

function reportPrivacy(p) {
  const redaction = p.redaction || {};
  const rows = [
    ["Messages", redaction.messages?.toLocaleString?.() ?? redaction.messages],
    ["Redaction", redaction.redacted ? "on" : "off"],
    ["Fields used", (redaction.fields_rendered || []).join(", ")],
    ["Stays local", redaction.stays_local],
  ].filter(([, v]) => v !== undefined && v !== "");
  return `<div class="card report-card"><dl class="spec">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
    ${(p.sent || []).length ? `<ul class="why">${p.sent.map((x) => `<li><span class="k">Sent</span><span>${esc(x)}</span></li>`).join("")}</ul>` : ""}</div>`;
}

function reportMetadata(r) {
  const m = r.metadata || {};
  return `<details class="card report-card meta"><summary>Run id, dates and provider ids</summary><dl class="spec">
    <dt>Run</dt><dd>${esc(m.run_id)}</dd><dt>Created</dt><dd>${esc(m.created_at)}</dd><dt>Report</dt><dd>${esc(r.generated_at)}</dd>
    <dt>Code</dt><dd>${esc(m.git_commit)}</dd><dt>Shipped attempt</dt><dd>${esc(m.final_attempt)}</dd>
    ${Object.entries(m.external_ids || {}).map(([k, v]) => `<dt>Attempt ${esc(k)}</dt><dd class="mono-sm">${esc(Object.values(v || {}).join(" · "))}</dd>`).join("")}
  </dl></details>`;
}

route();
