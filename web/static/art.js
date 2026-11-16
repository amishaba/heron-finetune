// Generative art for the demo: data converging into a model (hero), a dataset that organizes itself
// (results figure) and an orbit scribble (training).
// Colors come from the CSS custom properties in style.css, so the palette lives in one place.
// Each scene returns a stop() function and also stops itself once its canvas leaves the page.
const Art = (() => {
  const TAU = Math.PI * 2;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  function palette() {
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    return {
      ink: v("--ink"),
      accent: v("--accent"),
      // one hue per learned class (the transit-map palette)
      classes: ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6"].map(v),
    };
  }

  // seeded random, so the composition is the same on every load
  function rng(seed) {
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  }

  // Sizes the canvas for the device pixel ratio, calls onResize(size) after every size change, and runs
  // draw(ctx, size, now) every frame (or once per resize with reduced motion) until stop() or the canvas
  // is removed. ctx and size are passed in, so draw can run during setup. Returns stop().
  function scene(canvas, onResize, draw) {
    const ctx = canvas.getContext("2d");
    const size = { w: 0, h: 0 };
    let raf = 0;
    let running = true;
    const fit = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const r = canvas.getBoundingClientRect();
      size.w = r.width;
      size.h = r.height;
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      onResize(size);
      if (reduced && visible()) draw(ctx, size, performance.now());  // a resize clears the canvas; a static frame must be redrawn
    };
    const visible = () => size.w > 0 && size.h > 0;  // hidden canvases have no size; drawing then would be invalid
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);
    const stop = () => { running = false; cancelAnimationFrame(raf); ro.disconnect(); };
    const loop = (now) => {
      if (!running) return;
      if (!canvas.isConnected) { stop(); return; }
      if (visible()) draw(ctx, size, now);
      raf = requestAnimationFrame(loop);
    };
    fit();
    if (!reduced) raf = requestAnimationFrame(loop);
    return stop;
  }

  // ---------------------------------------------------------------- network (results figure)
  // ~150 points start as one tangled cloud, sort themselves into six colored clusters, then dissolve.
  function network(canvas, opts = {}) {
    const N = opts.n || 150, K = 6, CYCLE = 15000, REPEL = 120;
    const colors = palette();
    const rand = rng(7);
    const nodes = Array.from({ length: N }, (_, i) => ({
      cls: i % K, angle: rand() * TAU, radius: Math.sqrt(rand()), jitter: rand() * TAU, speed: 0.4 + rand(), size: 2 + rand() * 3.2,
      x: 0, y: 0, cx: 0, cy: 0, tx: 0, ty: 0,
    }));
    // chaos edges: two random partners each; order edges: the two nearest points of the same class
    const chaos = nodes.map((_, i) => [Math.floor(rand() * N), Math.floor(rand() * N)].filter((j) => j !== i));
    let order = [];
    let mouse = { x: -1e4, y: -1e4 };

    function layout({ w, h }) {
      const cx = w * (opts.cx ?? (w > 900 ? 0.63 : 0.5)), cy = h * (opts.cy ?? 0.44), R = Math.min(w, h) * (opts.r ?? 0.34);
      const sx = clamp((w / Math.min(w, h)) * 0.6, 1, opts.stretch ?? 1);  // spread out in wide frames, never squeeze
      const spread = Math.min(w, h) * 0.075;
      const centers = Array.from({ length: K }, (_, k) => {
        const a = -Math.PI / 2 + (k / K) * TAU + 0.35;
        return { x: cx + Math.cos(a) * R * 0.78 * sx, y: cy + Math.sin(a) * R * 0.62 };
      });
      for (const n of nodes) {
        n.cx = cx + Math.cos(n.angle) * n.radius * R * 1.25 * sx;  // chaos position: one tangled cloud
        n.cy = cy + Math.sin(n.angle) * n.radius * R * 0.95;
        n.tx = centers[n.cls].x + Math.cos(n.jitter * 3.1) * spread * n.radius;  // ordered position: its cluster
        n.ty = centers[n.cls].y + Math.sin(n.jitter * 2.3) * spread * n.radius;
      }
      order = nodes.map((n, i) => nodes
        .map((m, j) => [j, (m.tx - n.tx) ** 2 + (m.ty - n.ty) ** 2])
        .filter(([j]) => j !== i && nodes[j].cls === n.cls)
        .sort((a, b) => a[1] - b[1])
        .slice(0, 2)
        .map(([j]) => j));
    }

    // 0..1, how organized: chaos (hold) → organize → ordered (hold) → dissolve
    function organized(t) {
      const p = (t % CYCLE) / CYCLE;
      if (p < 0.18) return 0;
      if (p < 0.45) return ease((p - 0.18) / 0.27);
      if (p < 0.82) return 1;
      return 1 - ease((p - 0.82) / 0.18);
    }

    const t0 = performance.now();
    const stop = scene(canvas, layout, (ctx, size, now) => {
      const t = reduced ? CYCLE * 0.6 : now - t0;
      const o = organized(t);
      const wobble = reduced ? 0 : 6 * (1 - o * 0.6);
      ctx.clearRect(0, 0, size.w, size.h);
      for (const n of nodes) {
        let x = n.cx + (n.tx - n.cx) * o + Math.cos(t * 0.0006 * n.speed + n.jitter) * wobble;
        let y = n.cy + (n.ty - n.cy) * o + Math.sin(t * 0.0005 * n.speed + n.jitter * 1.7) * wobble;
        const dx = x - mouse.x, dy = y - mouse.y, d = Math.hypot(dx, dy);
        if (d < REPEL) { const push = (1 - d / REPEL) * 26 / (d || 1); x += dx * push; y += dy * push; }
        n.x = x;
        n.y = y;
      }
      ctx.lineWidth = 0.6;
      ctx.strokeStyle = colors.ink;
      ctx.globalAlpha = 0.22 * (1 - o);  // chaos lines fade out...
      ctx.beginPath();
      nodes.forEach((n, i) => chaos[i].forEach((j) => { ctx.moveTo(n.x, n.y); ctx.lineTo(nodes[j].x, nodes[j].y); }));
      ctx.stroke();
      if (o > 0.01) {  // ...cluster lines fade in
        ctx.globalAlpha = 0.55 * o;
        nodes.forEach((n, i) => {
          ctx.strokeStyle = colors.classes[n.cls];
          ctx.beginPath();
          order[i].forEach((j) => { ctx.moveTo(n.x, n.y); ctx.lineTo(nodes[j].x, nodes[j].y); });
          ctx.stroke();
        });
      }
      const sorted = o > 0.5;
      ctx.globalAlpha = sorted ? 0.35 + 0.65 * o : 1;
      for (const n of nodes) {
        ctx.fillStyle = sorted ? colors.classes[n.cls] : colors.ink;
        ctx.beginPath();
        if (sorted && n.cls % 3 === 2) ctx.rect(n.x - n.size, n.y - n.size, n.size * 2, n.size * 2);  // squares, like the pixel prints
        else ctx.arc(n.x, n.y, n.size, 0, TAU);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });

    canvas.addEventListener("pointermove", (e) => { const r = canvas.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
    canvas.addEventListener("pointerleave", () => (mouse = { x: -1e4, y: -1e4 }));
    return stop;
  }

  // ---------------------------------------------------------------- convergence (hero)
  // Tangled curves on the left (raw data) wander down the page, each ends on one of the agent's points, and
  // from there a straight line runs into a ruled box (the model). Drawn progressively, held, then redrawn.
  // The hero's text labels (.flow-label / .flow-caption in style.css) sit at these x fractions: data 0.17,
  // agent AGENT_X (44%), model BOX_X + BOX_W / 2 (84%). Change them together.
  const AGENT_X = 0.44, BOX_X = 0.74, BOX_W = 0.2;
  function convergence(canvas) {
    const colors = palette();
    const rand = rng(11);
    const N = 13, CYCLE = 11000;
    const strands = Array.from({ length: N }, (_, i) => ({
      // control points for a wandering path from off-canvas top/bottom-left to its agent point
      pts: Array.from({ length: 5 }, () => [rand() * 0.36, rand()]),
      start: [rand() * 0.3, rand() < 0.5 ? -0.08 : 1.08],
      slot: (i + 0.5) / N,
    }));
    let paths = [];
    function layout({ w, h }) {
      const top = h * 0.22, span = h * 0.56, apex = { x: w * BOX_X, y: h * 0.5 };
      paths = strands.map((st) => {
        const node = { x: w * AGENT_X, y: top + st.slot * span };
        const pts = [[st.start[0] * w, st.start[1] * h], ...st.pts.map(([x, y]) => [x * w, y * h]), [node.x - w * 0.1, node.y], [node.x, node.y]];
        return { pts, node, apex };
      });
    }
    // length-parameterized drawing of a smooth path through pts (Catmull-Rom → Bézier), up to fraction f
    function curve(ctx, pts, f) {
      const segs = pts.length - 1, upto = f * segs;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 0; i < segs && i < upto; i++) {
        const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(segs, i + 2)];
        const t = Math.min(1, upto - i);
        const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
        const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
        if (t >= 1) ctx.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1]);
        else {  // the partial last segment: sample its cubic up to t
          const at = (tt, a, b, c, d) => (1 - tt) ** 3 * a + 3 * (1 - tt) ** 2 * tt * b + 3 * (1 - tt) * tt * tt * c + tt ** 3 * d;
          for (let k = 1; k <= 8; k++) {
            const tt = (k / 8) * t;
            ctx.lineTo(at(tt, p1[0], c1[0], c2[0], p2[0]), at(tt, p1[1], c1[1], c2[1], p2[1]));
          }
        }
      }
      ctx.stroke();
    }
    const t0 = performance.now();
    return scene(canvas, layout, (ctx, size, now) => {
      const p = reduced ? 1 : ((now - t0) % CYCLE) / CYCLE;
      // timeline: tangles 0-.38, convergence .38-.62, box rules .62-.74, hold, fade out .92-1
      const tangle = reduced ? 1 : ease(clamp(p / 0.38, 0, 1));
      const conv = reduced ? 1 : ease(clamp((p - 0.38) / 0.24, 0, 1));
      const rules = reduced ? 1 : clamp((p - 0.62) / 0.12, 0, 1);
      const fade = reduced ? 1 : 1 - clamp((p - 0.92) / 0.08, 0, 1);
      const { w, h } = size;
      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = fade;
      ctx.strokeStyle = colors.ink;
      ctx.lineWidth = 1;
      for (const path of paths) curve(ctx, path.pts, tangle);
      ctx.fillStyle = colors.ink;
      for (const path of paths) {
        if (tangle < 0.98) continue;
        ctx.beginPath();
        ctx.arc(path.node.x, path.node.y, 2.4, 0, TAU);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(path.node.x, path.node.y);
        ctx.lineTo(path.node.x + (path.apex.x - path.node.x) * conv, path.node.y + (path.apex.y - path.node.y) * conv);
        ctx.stroke();
      }
      // the model: a box of fine rules, filling in line by line, with the apex point in the accent color
      const bx = w * BOX_X, bw = w * BOX_W, by = h * 0.4, bh = h * 0.2, lines = 14;
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 2]);
      for (let i = 0; i < Math.round(lines * rules); i++) {
        const y = by + (i / (lines - 1)) * bh;
        ctx.beginPath();
        ctx.moveTo(bx, y);
        ctx.lineTo(bx + bw, y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (conv > 0.99) {
        ctx.fillStyle = colors.accent;
        ctx.beginPath();
        ctx.arc(bx, h * 0.5, 5, 0, TAU);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });
  }

  // ---------------------------------------------------------------- orbit scribble
  // Accent-colored trails orbiting a dark disc. Speed follows canvas.dataset.speed (set by the run view).
  function vortex(canvas, opts = {}) {
    const P = 22, TRAIL = 70;
    const colors = palette();
    const rand = rng(opts.seed || 3);
    const particles = Array.from({ length: P }, () => ({
      a: rand() * TAU, r: 0.32 + rand() * 0.55, dir: (0.6 + rand()) * (rand() < 0.5 ? 1 : -1), phase: rand() * TAU, trail: [],
    }));
    return scene(canvas, () => {}, (ctx, size, now) => {
      const t = now / 1000, speed = Number(canvas.dataset.speed || 1);
      const cx = size.w / 2, cy = size.h / 2, R = Math.min(size.w, size.h) / 2 - 8;
      ctx.clearRect(0, 0, size.w, size.h);
      ctx.lineWidth = 1.4;
      for (const p of particles) {
        p.a += 0.012 * p.dir * speed;
        const r = R * clamp(p.r + Math.sin(t * 0.9 + p.phase) * 0.08, 0.2, 0.98);
        const x = cx + Math.cos(p.a) * r * (1 + 0.12 * Math.sin(p.a * 3 + p.phase));
        const y = cy + Math.sin(p.a) * r * (1 + 0.12 * Math.cos(p.a * 2 + p.phase));
        p.trail.push([x, y]);
        if (p.trail.length > TRAIL) p.trail.shift();
        ctx.strokeStyle = colors.accent;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        p.trail.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = colors.ink;
        ctx.beginPath();
        ctx.arc(x, y, 2.6, 0, TAU);
        ctx.fill();
      }
      ctx.fillStyle = colors.ink;
      ctx.beginPath();
      ctx.arc(cx, cy, R * 0.2, 0, TAU);
      ctx.fill();
    });
  }

  return { convergence, network, vortex };
})();
