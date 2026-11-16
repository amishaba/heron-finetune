# The demo (Heron)

There are two versions of the same page:

- **Static site (the public one):** the finished runs on Vercel, with no server, API keys or spend. Live runs are by
  email. See [Static site on Vercel](#static-site-on-vercel).
- **Live demo:** the page plus the real pipeline in one container, so visitors can start paid runs. It's described
  below.

## Static site on Vercel

`.venv/bin/python demo/build_static.py` writes `web/` from `demo/web` and `demo/featured`. Commit `web/`: Vercel serves
it as is, with no build step (`vercel.json`), and `.vercelignore` limits uploads to the built site.

The email run's report quotes real messages from the Enron corpus, so the snapshot in `demo/featured` leaves them out.
The build stops if any email address other than the contact one, or a local path, reaches `web/`.

## Live demo

One container serves the page, a small JSON API and the real pipeline. A visitor uploads a CSV/JSONL (or picks
a sample), the agent plans it (a fraction of a cent), and on approval the run trains and benchmarks for real
(~25 min, ~$5: Together's $4 training minimum + endpoint time + GPT baselines).

## Guards (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `FT_TOTAL_BUDGET_USD` | 100 | Hard stop for all spend on this server. Its `runs/` starts empty, so this is the demo's own cap. |
| `FT_DEMO_RUNS_PER_VISITOR_PER_DAY` | 1 | Fine-tunes per visitor (IP) per day; `0` = no limit. |
| `FT_DEMO_MAX_ROWS` / `FT_DEMO_MAX_BYTES` | 5000 / 5 MB | Upload limits. |
| `FT_DEMO_TEST_MAX` / `FT_DEMO_VAL_MAX` | 300 / 200 | Smaller held-out sets so a run finishes in ~25 min. |
| `FT_DEMO_DRY_RUN` | off | `1` = fake models, no spend (for trying the site). |

Paid runs go through one queue, one at a time. A server restart re-queues approved runs that didn't finish (the
pipeline resumes from its last completed stage), and every endpoint is torn down even when a run fails.

## Deploy on Railway

1. railway.com → New Project → Deploy from GitHub repo → this repo. Railway builds the `Dockerfile`.
2. Variables: `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `FT_TOTAL_BUDGET_USD` (the demo's own cap, e.g. `25`),
   `FT_DEMO_RUNS_PER_VISITOR_PER_DAY` (e.g. `1`).
3. Add a volume mounted at `/data` (runs and uploads live there; `FT_RUNS_DIR` / `FT_DEMO_DIR` point into it).
4. Settings → Networking → Generate Domain.

Keep it to one instance: the run queue is in memory. Any host that runs a Dockerfile with a persistent disk at
`/data` works the same way (Fly.io, Render).

## Featured runs

`python demo/build_featured.py` snapshots finished runs from `runs/` into `demo/featured/` (numbers copied from
the reports, account names scrubbed, quoted emails left out). They're shown to every visitor at no cost.

## UI checks

`demo/uicheck/scenarios.py` drives a headless Chrome through the real page (every link and button, the sample →
plan → approve → result flow, Back/Forward, bad uploads, unknown URLs, phone width, leaving mid-poll) and checks the
outcomes: 47 checks. Start a dry-run server first (command in the file's docstring), then run it.
