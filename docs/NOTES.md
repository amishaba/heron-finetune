# NOTES

The build log: differences from `MVP_SPEC.md`, decisions, run results and incidents, in the order they happened. Where this file and the spec disagree, this file wins.

The project was built with an AI coding agent working from the spec, one milestone at a time; "the user" below is the author, who made the decisions and approved every paid run.

## Decisions (Oct 16, 2026)

### Frontier provider: OpenAI, not Anthropic

The spec uses Claude for baselines, frontier labeling, the agent and simulated logs. We use **OpenAI only** for all four. Reason: cost. No Anthropic provider is built for now; `FrontierProvider` stays provider-agnostic so one can be added later.

| Use | Spec says | We use |
| --- | --- | --- |
| Baselines | `claude-haiku-5-5` + `claude-sonnet-5-5` | GPT-5.6 **Luna** only (a stronger second baseline, e.g. Sol, comes later) |
| Agent (M3, M4) | `claude-opus-5-5` | Luna (default; revisit if config quality is poor) |
| Frontier labeling | `targets.frontier_model` (Claude) | Luna |
| Simulated logs (M6) | `claude-haiku-5-5` | Luna |

To verify against OpenAI docs before coding (exact model IDs, current prices):

- Luna's model ID (expected `gpt-5.6-luna`; unconfirmed).
- Price. Press reports from Jul–Aug 2026, unconfirmed on OpenAI's page: Luna $0.20 in / $0.02 cached / $1.20 out per 1M tokens.

Spec sections that change because of this:

- **7 (task.yaml), `baselines[*]`:** `effort` + `thinking` are replaced by OpenAI's reasoning effort setting. Exact field names follow the OpenAI docs.
- **9.6 (baseline):** structured outputs use OpenAI's JSON schema mode (strict). Prompt caching is automatic on OpenAI (prompts ≥ 1,024 tokens); no `cache_control` code. Short classify prompts may not cache; report actual cached tokens.
- **10.1:** replaced by an OpenAI provider (Chat Completions or Responses, structured outputs, Batch API, images as data URLs). Key from `OPENAI_API_KEY`.
- **7, 14.3 (`baselines`):** one entry (Luna) instead of two. `pass_bar.compare_to: best_baseline` still works with a single baseline. Adding a second model later is a `task.yaml` change only.
- **11.1 (prices.yaml):** `anthropic:` section becomes `openai:` with Luna, plus the Batch discount.
- **14.3 (email baselines):** Luna with `context_file`.
- **15 (logs):** simulated logs come from Luna. Logs normalization still accepts both OpenAI and Anthropic shapes.

### Budget

Total real spend across all runs: ideally under **$100** for now. Firm caps decided at the paid milestones. Track cumulative spend across runs (shown in `ft status`) and ask before any run whose estimate would push the total near $100.

Why Luna only: Sol (~$5 / $30 per 1M tokens, ~25× Luna) would cost roughly $25–30 for BANKING77 val + test alone, too much of the budget for v0.

### Email pass bar (section 14)

v0 keeps `max_gap_points: 0`: a tie passes. Done-when updated in the spec to "match or beat". Report verdict wording follows the rule added to spec section 9.11 (paired bootstrap of the difference: "beat" only when the 95% CI is entirely above 0, "matched" when it includes 0).

### Training on frontier outputs (logs mode, M6)

OpenAI's terms restrict using outputs to develop competing models. Accepted for this personal research demo; keep the reminder the spec prints when `targets.from` is `logs` or `frontier`.

### Git

Report metadata leaves the git commit blank when there is no repo.

## M0 build notes (Oct 16, 2026)

### Environment

- Machine had no `uv` and only Python 3.15. `brew install uv` started compiling LLVM from source, so it was stopped. `uv` is installed from PyPI into `~/.local/share/uv-venv`, linked at `~/.local/bin/uv` (not on PATH). Python 3.12 is uv-managed; the project venv is `.venv/`.
- `together[cli]` and `datasets` are optional extras (`uv sync --extra together --extra hf`), installed when M1/M2 need them.

### Differences from the spec

- **`Server` interface (10):** `create / is_ready / teardown / list_ft_endpoints` instead of a `deploy()` context manager. The context manager lives in `ft/stages/serve.py`, so the teardown guarantees (IDs saved on create, `finally`, SIGTERM → SystemExit, atexit, idempotent) are written once for every provider.
- **Stage list:** `serve` runs inside `predict` (the endpoint only lives while predicting). Status shows `predict`; endpoint records are in `state.json` and `attempts/<n>/endpoint.json`.
- **Baseline config:** `baselines[*]` has `reasoning_effort` instead of `effort` + `thinking`. Allowed values per model are checked by the OpenAI provider at M1.
- **Near-duplicates (9.3):** inputs that match after case/whitespace normalization are dropped before splitting (keep first, so train wins over test), and the count is logged and shown in the report. BANKING77 has 12. The leakage check still runs after splitting and fails loudly. Without the drop, the proof dataset would always fail the check.
- **`splits.test_fraction`** (new, default 0.2): test size when there is no predefined test split. The spec only had `test_max`.
- **Metrics:** computed with numpy (fast enough to bootstrap 1,000 times), so scikit-learn isn't a dependency.
- **`--from STAGE`** keeps training job IDs, so `--from train` polls the existing job instead of paying for a new one. To retrain, start a new run (or, from M4, a new attempt).
- **Endpoint names:** `ft-<run_id>-a<n>`, lowercase letters, digits and hyphens, at most 60 characters. Together's real limit is checked at M1.

### Spend guard additions

- Each provider section of `prices.yaml` has `verified:`. Real runs refuse a paid stage until its provider's prices are verified.
- Total real spend across all runs is capped by `FT_TOTAL_BUDGET_USD` (default 100, from the user's budget). `ft status` shows it.
- Dry runs price everything (marked `simulated`) so the report's cost sections are exercised. Missing Together endpoint rates use a $2.50/h placeholder in dry runs only. Simulated spend never counts toward real budgets.
- Fake providers don't sleep. Throughput in dry runs comes from simulated latency ÷ concurrency.

### Report

- Styled in the user's design language (black field, hairline rules, serif headings + mono). The report must have no external assets, so it uses system serif/mono fallbacks instead of Instrument Serif / JetBrains Mono.
- Verdict wording follows spec 9.11 ("beat" only when the paired-bootstrap CI of the test difference is entirely above 0).

## Pre-M1 changes and verified facts (Oct 16, 2026)

### Changes requested after M0

- **Duplicates:** one rule, applied in the split stage to exact and case/whitespace duplicates. When copies span splits, the most valuable split keeps its copy (test > val > train). Copies inside one predefined val/test split are all kept. Copies in train or unsplit data keep only the first. Every drop goes to `data/duplicates_dropped.jsonl` (with label conflicts flagged) and into the report. BANKING77: test keeps all 3,080; 11 train copies dropped (7 duplicated a test message, 4 repeated within train, no label conflicts). `load` no longer drops anything.
- **Report verdict** leads with pass/fail against the bar (val), then the test comparison.

### OpenAI (developers.openai.com, checked 2026-10-16)

- `gpt-5.6-luna`: $0.20 in / $0.02 cached / $0.25 cache write / $1.20 out per 1M; Batch is 50% off. Reasoning tokens bill as output.
- **`gpt-6-luna` exists** (released 2026-09-22): $0.10 / $0.01 / $0.50, half of GPT-5.6 Luna. Baseline choice pending with the user.
- Effort values: `none, low, medium (default), high, xhigh, max`. There is no `minimal`. `temperature` isn't documented for GPT-5.6, so we don't send it.
- Use the **Responses API** ("recommended for all new projects"). Structured output via `text.format` json_schema strict. Limits: ≤1000 enum values in total (77 is fine), `additionalProperties: false`, every field required. A refusal item may not follow the schema, so it's marked invalid.
- Prompt caching is automatic at ≥1,024 tokens. The BANKING77 baseline prompt is ~440 tokens, so it won't cache. `max_output_tokens` includes reasoning tokens, and `status: "incomplete"` means the budget was too small, which counts as invalid.
- Batch API: ≤50,000 requests / 200 MB per file, one model per file, `completion_window: "24h"`.
- Services Agreement (v.010126) restricts using Output to build competing models, **except** "models primarily intended to categorize, classify, or organize data … if these models are not distributed or made commercially available to third parties". Our classifiers fit that exception while they stay personal.

### Together (docs.together.ai + together.ai/pricing, checked 2026-10-16)

- `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` support LoRA SFT. Both deploy on `1xnvidia-h100-80gb` at **$5.49/h**, billed per minute while ready. Serverless LoRA is discontinued.
- LoRA SFT costs **$0.34 per 1M billed tokens** (dataset tokens × epochs, plus eval tokens), with a **$4.00 minimum per job**. A cancelled job pays for completed steps.
- **`Qwen/Qwen3-VL-8B-Instruct` is no longer fine-tunable** (changelog 2026-07-29). Qwen3.5-4B/9B are listed as vision-tunable. CORD's base model needs the user's OK before substituting (spec 18).
- Endpoints do **not** auto-stop by default. At M1 we'll pass `--inactive-timeout 30` as a backstop, on top of our own teardown.
- SDK/CLI: the upload/create/retrieve flow matches the spec. The model id is `model_object_id` (REST) / `api_model_object_id` (SDK). Top-level `train_on_inputs` is deprecated in favour of `training_method.train_on_inputs`. API suffix ≤64 characters (CLI ≤40). The CLI has a global `--json`. Endpoint name rules aren't documented. Deploy via the CLI (the SDK needs 3 calls).
- Unverified until we have an API key: `tg beta models configs` (deployability) and `tg fine-tuning model-limits` (learning-rate bounds, LoRA rank; docs disagree between max rank 64 and 128).

### CORD v2 (HF datasets-server, checked 2026-10-16)

- Splits 800 / 100 / 100, CC-BY-4.0, columns `image` + `ground_truth` (JSON string). Our target_map paths exist.
- `menu` is a single dict in ~1/3 of rows; `as_list` already handles that. `sub_total` is absent in ~1/4 of rows, so missing values become "".
- Prices mix "75,000", "22.000", "40,000.", "Rp 0.000". The `money` normalizer (digits only) has to handle these. `cnt` formats vary ("1 x", "x1", "1X").
- Images range from 432×648 to 2304×4096. Downscale to ≤1,280px as the spec says.

## Decisions (Oct 16, 2026, continued)

- **Baseline: `gpt-6-luna` at reasoning effort `low`** (the user's choice). This replaces GPT-5.6 Luna in the examples. Confirm on the first live call that gpt-6-luna accepts `low` and the strict json_schema format.
- **Together prices confirmed by the user** ($0.34/1M LoRA tokens, $4.00 job minimum, $5.49/h H100). `prices.yaml` together section is now `verified: true`.
- **CORD (M2) uses `Qwen/Qwen3.5-4B`** instead of Qwen3-VL-8B (approved by the user).

## M1 build notes (Oct 16, 2026)

### Verified live (free or < $0.01)

- `tg beta models configs`: Qwen3.5-4B has one config (`cr_CfHSNAes5gNmGHLdTQSVv`, 1x H100). Qwen3.5-9B has two (1x H100 `cr_Cg2pEYrLqgASTNAtUnV6u`, 4x H100), so deploy pins the 1x config (to confirm at M4 that a base-model config applies to a fine-tune).
- `tg fine-tuning model-limits`: lr 1e-8..1.0, LoRA rank default 64 / max 128, `merge_output_lora: true`, `supports_reasoning: true`.
- gpt-6-luna with effort `low` + a strict 77-value enum works. Measured ~860 input tokens/call (label lists tokenize at ~2 chars/token, so the estimator now uses chars/2 for system prompts), ~43 output tokens, ~1.8 s p50.
- Together accepted the BANKING77 training file (COMPLETED). `estimate_price` says **$4.00** (job minimum) for 667K billed tokens (3 epochs). The probe file was deleted.
- The real `ft run` gets through load → label and stops at the baseline spend prompt as designed.

### Provider design

- **OpenAI:** Responses API, one AsyncOpenAI client per `predict()` call (each `asyncio.run` has its own loop). SDK retries handle 429/5xx with retry-after (instead of tenacity). Reasoning budget is added on top of answer tokens (`low`: +1,024); we pay only for what's used.
- **Together training:** upload (free) → `estimate_price` (free) → spend guard → `create` (with `random_seed`). The job's `total_price` (nanodollars) is recorded as actual spend when present. The loss curve comes from `list_metrics`.
- **Together serving:** the CLI with `--json --non-interactive --project <id>`. `deploy` passes `--min/max-replicas 1`, `--inactive-timeout 30` (backstop) and `--traffic-weight 1` (without it the new deployment gets no traffic). If the model is an adapter, it retries with `--merge` (the CLI checks this before creating anything). Readiness = a deployment in `DEPLOYMENT_STATE_READY` with ≥1 ready replica. Teardown = `rm <ep_id> --force`, and "not found" counts as done.
- **Leak protection added:** the endpoint name goes into state.json *before* the deploy call. If deploy dies after the endpoint exists, we look it up by name and tear it down. `ft teardown` finds endpoints by name when the id was never saved. `ft teardown --all` lists every `ft-` endpoint on the account and records spend through the owning run. `ft status` on real runs also lists `ft-` endpoints on the account.
- **Tuned inference:** OpenAI SDK at `https://api-inference.together.ai/v1`, `temperature=0`, `chat_template_kwargs.enable_thinking=false` (dropped if the server rejects it), and any `<think>…</think>` is stripped before parsing.
- **Outages vs. model errors:** API failures are retried once; if more than 2% still fail, the stage stops instead of scoring an outage as wrong answers. Refusals and "incomplete" (output budget) are model outcomes, so they count as wrong and aren't retried.

## Generality audit (Oct 16, 2026)

### BANKING77-specific code found in `ft/`

A search for dataset names, column names, label names and 77/3080/10003 found only:

- `splits.test_max` defaulted to **3080** (BANKING77's test size). It's now 2000; `examples/banking77/task.yaml` sets 3080 itself.
- A comment in the baseline cost estimator cited BANKING77 for the ~2 chars/token measurement. It's reworded; the heuristic is generic (and errs high).

Nothing else: column names, labels, paths and sizes all come from `task.yaml`. BANKING77-only things (`fetch.sh`, the CSV paths, `test_max: 3080`, the instruction text) live in `examples/banking77/`.

### Assumptions that hold for BANKING77 but not in general

The new fixture `examples/fixtures/support_routing/` (JSONL, different columns, no predefined split, 60%/3% classes, multi-sentence inputs, long label names, messy rows) broke three of these silently before the fixes. Its run "passed" while label variants became separate classes and a blank message went into training.

| Assumption (true for BANKING77) | When it doesn't hold | How the code handles it |
|---|---|---|
| CSV, UTF-8, comma-separated, header row | JSONL; CSV with a BOM (Excel); TSV/Excel/Parquet | `csv` and `jsonl` sources; BOM tolerated (`utf-8-sig`). Other formats aren't supported: a clear "no .csv/.jsonl files" error. HF datasets come in M2. |
| Columns `text` / `category` | Any other names; nested JSON fields | Names come from `task.input.field` / `task.output.field`. A missing column errors with the list of available columns. Nested input fields aren't supported (`target_map` covers nested targets only). |
| Predefined train/test files | One file, no split | Files named train/val/validation/dev/test become split hints; anything else is carved: test = `test_fraction` (default 0.2) capped at `test_max`, val = `val_fraction` of the rest capped at `val_max`, both stratified. The report says "carved" vs "predefined+carved". |
| Balanced classes (~130 each) | Lopsided (e.g. 60% / 3%) | Stratified carving that **never takes a class's last training example** (`keep_one`). Warnings (in `splits.json` and the report) when the metric is `accuracy` and the top class is >40% of train, and for classes with <5 training examples. `macro_f1` is available; choosing it automatically is the agent's job (M3). |
| Labels are clean, consistent strings | Case/whitespace variants ("Billing", " billing "); int/bool labels in JSONL | Variants are **merged into the most common spelling** and logged ("merged … into …" in the report). Non-string labels become strings. Synonyms ("refund" vs "refunds") are not merged; that's left to the agent (M3). |
| Exactly one label per row | Multi-label (a list) or structured targets | Classify rejects lists/dicts with a clear "exactly one label per row" error. Multi-label isn't supported. |
| No blank inputs, every row labeled | Blank/whitespace inputs; missing labels | Dropped at load and counted (`load_stats.json`, report "data cleaning"). |
| Short label names | Long names (`legal_and_compliance_escalation`) | Output budgets scale with the longest label (tuned: max(16, len/2+8) tokens; baseline: max(64, len/2+32)), so long labels aren't truncated. |
| Short inputs (~1 sentence) | Long documents | Not truncated. Cost estimates scale with length, and the profile shows length p95/max. **Gap:** inputs beyond the model's sequence limit (Together 131K tokens for 4B, 65K for 9B) aren't handled; Together's file validation would reject them. Revisit with CUAD. |
| ≤77 classes | Hundreds or thousands of classes | Fine up to OpenAI's structured-output limit (≤1,000 enum values). **Gap:** above that, baselines would fail; not handled. |
| No duplicates across splits | Duplicates, including with conflicting labels | One rule (test > val > train; train/unsplit keeps the first), logged with label conflicts flagged. |
| Labels present in every split | A class only in val/test | Warning; those rows count as wrong (nothing can predict an unseen label). |
| ~10K rows | Tiny or huge datasets | Tiny: training still costs Together's $4 minimum, and wide CIs show in the report. Huge: `val_max`/`test_max` cap eval cost and `train.max_examples` caps training. |
| English text | Other languages | Nothing assumes English (normalization is casefold + whitespace). |

### Known limitation found along the way

Stage input hashes cover data and config, not code. After a code change, an existing run resumes with stale outputs, so start a new run (that's what we did for `m1-banking77`).

## M1 run incident (Oct 16, 2026): deploy 409

- **What happened:** training finished (12.5 min, billed $4.00). `tg beta endpoints deploy … --traffic-weight 1` created the endpoint and its deployment, then failed with **409 Conflict** when setting the traffic split. On a new endpoint, the CLI sends the etag it read *before* creating the deployment, and the deployment creation bumps it. Our cleanup found the endpoint by name and called `rm --force`. That scaled the deployment to 0 replicas, but the delete itself also got 409 (the deployment was mid-transition).
- **Cost:** none for the endpoint. It never reached READY, and it was `STOPPED` with 0 replicas when inspected. It was deleted by hand, and the account has no endpoints. Logged as 2 `human_intervention` events (manual delete, code fix mid-run), so the report counts them.
- **Fixes:**
  - deploy no longer passes `--traffic-weight`. Routing is set afterwards with `tg beta endpoints update <dep> --traffic-weight 1`, which re-reads the etag, and only if the deployment isn't already routed.
  - 409s on routing and on teardown are retried with backoff (5 s → 60 s, ~4 min total).
  - A re-deploy under the same name archives the earlier record, so teardown can't mistake the new endpoint for an already torn-down one.
- **Lesson:** the leak protections worked (name saved before deploy → found and scaled down), but teardown needs retries. The "ft- endpoints on the Together account" line in `ft status` is how this was checked.

## Named baselines and few-shot (Oct 16, 2026)

- `baselines[*].name` (default: the model name if it's used once, else `<model>-<effort>[-<N>shot]`) keys folders, report rows and `pass_bar.compare_to`, so one model can appear with several settings.
- `baselines[*].few_shot: N` puts N train examples per label (seeded, never val/test) in the system prompt, after the label list. The examples are saved to `baselines/<name>/few_shot.json` and reused. Because they sit in the stable prefix and requests share `prompt_cache_key`, OpenAI's automatic caching covers them (cached input ~10x cheaper). Classify only.
- Each baseline is cached on its own (`done.json`, keyed by settings + exact prompt + val/test data), so adding baselines to a finished run pays only for the new ones. `ft resume <run> --update-task <file>` swaps in a new task.yaml (same task name), logs a `task_updated` event the report lists, and re-runs only stages whose inputs changed.
- The M1 run's original `baselines/gpt-6-luna` was renamed to `gpt-6-luna-low` with a cache marker (same settings and prompt, 64-token output budget unchanged), so it wasn't paid for twice.
- The report adds a "tuned vs each baseline" table, with the beat/matched/fell-short rule applied per baseline.
- Bug found by the new tests: the report's "always rerun" hash was a timestamp to the second, so two runs within one second skipped the report. It now uses a random value.

## M5: email cleaner (Oct 16, 2026)

Moved ahead of CORD at the user's request. Daily scheduling is left out: `ft email apply` is on demand only.

- **`ft email pull`** (scope `gmail.readonly`; token `~/.config/ft/gmail_token.json`) exports the last 12 months, skipping the last 7 days, chats, drafts and sent mail, to `data/email/messages.jsonl` (chmod 600, gitignored), plus a `.meta.json`. Reply detection lists your sent mail in the window (threadId + date only). Messages are fetched in batches of 50 with backoff on rate limits. Spam/trash are excluded (Gmail's default).
- **Runs never call Gmail.** The load stage renders, redacts and labels the local export, so a run is reproducible from the export file. Deviation from spec 9.4: behavior labels are computed at load, because split needs targets. The label stage validates them.
- **Behavior rules** use a tiny parser (AND/OR/NOT/parentheses, never `eval`) over `opened, starred, replied_after, archived, has_unsubscribe`. Unknown features or labels fail when task.yaml is loaded.
- **Redaction** (`ft/redact.py`): Presidio with spaCy `en_core_web_sm` (12 MB, instead of the 560 MB default; swap if recall disappoints) plus pattern recognizers. It replaces PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, CREDIT_CARD, IBAN_CODE, US_SSN/bank/passport/licence, UK_NHS, IP_ADDRESS, CRYPTO, URL with `<TYPE>`. Senders become `sender_<salted sha256[:6]>` + domain (salt in `~/.config/ft/salt`, never uploaded). Redaction counts go to `data/redaction.json` and the report's "what left this machine" section, which lists every upload (OpenAI: val/test + context file + few-shot examples; Together: training file and endpoint traffic).
- **`ft email apply --run <id>`** (scope `gmail.modify`, separate token `gmail_modify_token.json`) fetches mail since the checkpoint (`~/.config/ft/email_state.json`; first time: last 24 h), renders and redacts it the same way, deploys the run's best model, classifies, tears down, and **adds** `ft/Probably skip` / `ft/Important`. The code has no path that removes labels, archives, deletes or marks read; `add_labels` asserts the request only adds. The log (`email_apply_log.jsonl`) holds ids and counts only. Each apply pays for endpoint start-up plus a minute or two (~$0.50–1.00).
- **Tests** use a synthetic, fabricated export: parsing, `replied_after`, rules (incl. injection-shaped input), redaction, the spec's render format, a full by_time dry run checking that no names, addresses or phone digits reach the training file, and apply against a fake Gmail service (only addLabelIds, checkpoint saved, endpoint torn down, no content in the log).

### BANKING77 extra baselines: results (Oct 16, 2026)

| Baseline | Test acc | $ / 1k req | p50 | Notes |
|---|---|---|---|---|
| gpt-6-luna low | 82.0 | $0.106 | 1.1 s | reused from M1 |
| gpt-6-luna medium | 83.0 | $0.119 | 1.3 s | +1 point for 67 vs 39 output tokens/call |
| gpt-6-luna low, 3-shot | **87.0** | **$0.075** | 2.0 s | 231 examples (~6.1K tokens) in the prompt; **99.6% of input tokens served from cache**, so cheaper than zero-shot |
| tuned Qwen3.5-4B | **92.3** | $0.042 (full utilization) | 0.41 s | beats all three (paired CI vs 3-shot: +4.2 to +6.5) |

- Cost of the two new baselines: $0.80 (run total $5.69 of the $12 cap).
- **Rate limits:** 3-shot at concurrency 16 sends ~6K tokens per call, over the Build tier's 2M tokens/min. 33/999 val and 277/3080 test calls failed even after the SDK's own retries; our single retry pass recovered all of them. Known limit: concurrency isn't adapted to prompt size. If it bites, add a per-baseline concurrency or a tokens-per-minute budget.
- **Fixed:** when a stage stopped on >2% failures, the calls that had succeeded weren't added to spend. `TooManyFailures` now carries the predictions, and the baseline stage records their cost before stopping.

## Email demo on Enron instead of Gmail (Oct 17, 2026)

The user decided not to connect their Gmail. The Gmail connector, `ft email pull` and `ft email apply` stay (with `examples/email/`), unused for now. The email demo uses the public Enron corpus instead.

- **Task change (deviation from spec 14):** spec 14 predicts *behavior* (important / read / skip) from Gmail signals. Enron has no read/starred flags, so the demo predicts **which of one employee's own folders they filed an email into** (`examples/enron/task.yaml`, `targets.from: dataset`). Same rendering, redaction, time split, macro-F1 and "match or beat" bar.
- **New generic sources** (`ft/connectors/mailfiles.py`), labeled by folder:
  - `maildir`: standard Maildir (cur/new, Maildir++ `.A.B` subfolders) and plain folder trees (Enron, MH).
  - `mbox`: a file or a directory of `.mbox` files (folder = file path), or `folder_from: header` for Google Takeout's `X-Gmail-Labels` (a message is kept only if exactly one non-excluded label remains).
  - Shared options: `exclude_folders` (case-insensitive globs on the folder path), `min_per_folder` (default 20), `redact` (default true).
  - Load drops, all counted in `load_stats.json`: excluded folders, copies of the same message (same sender, date, subject, body; Enron gives each copy its own Message-ID), messages filed in two kept folders (ambiguous), messages without a parseable date, empty messages, folders under the minimum.
  - Mail sources must split `by_time`, and maildir/mbox must use `targets.from: dataset` (checked when task.yaml loads).
- **Enron uses `redact: false`** (user decision, Oct 17). It's public data, and redaction was removing the words his folders are named after (London, Australia, Stanford, Rice, UT). Personal mail keeps the default `redact: true`.
- **OpenAI throttle:** a client-side token bucket keeps requests under 80% of `FT_OPENAI_TPM` (default 2M/min, the Build tier). OpenAI counts input (cached included) plus `max_output_tokens` against TPM, so long few-shot prompts at full concurrency were hitting 429s. This closes the "known limit" noted under the BANKING77 results.
- **Mailbox pick:** `examples/enron/pick_employee.py` ranks all 150 mailboxes by kept folders × filed emails, using the real loader. Another mailbox has the most folders (31) but only 1,502 emails, and a third of its newest mail sits in folders that start late. **One mailbox** (28 folders, 4,366 emails, 1999-11 to 2001-05; every test-period email is in a folder that already exists in train) wins on the product. Free local prep is in `runs/enron-first`: train 3,022 / val 336 / test 840, 168 exact rendered duplicates dropped (no label conflicts), 17 folders under 20 emails dropped (113 emails).
- **Raw header blocks stripped** (generic, in `mailfiles.strip_header_blocks`, also used by the Gmail connector). Forwards often paste a raw header block into the body (Return-Path, Received, Message-ID, X-…). A block starts at a transport header and runs to the next blank line, which also catches Enron's unindented wrapped lines. It needs at least two header-shaped lines, so prose such as "Received: your note" survives. Readable From/To/Cc/Subject/Date lines inside the block are kept. The chosen mailbox: 114 blocks removed, counted in `load_stats.json`.
- Primary metric: macro-F1, confirmed.

### Enron results (Oct 17, 2026)

Run `enron-first`: 28 folders, by_time split (train 3,022 / val 336 / test 840), redact false, macro-F1.

| Model | Val macro-F1 | Test macro-F1 (95% CI) | Test acc | $ / 1k req | p50 / p95 |
|---|---|---|---|---|---|
| gpt-6-luna low | 25.9 | 33.1 (29.8–36.6) | 43.6 | $0.133 | 2.1 s / 4.0 s |
| gpt-6-luna low, 3-shot (84 examples, ~23K-token prompt, 98% cached) | 38.5 | **52.9** (48.9–56.3) | 61.1 | $0.339 | 2.2 s / 18.4 s (throttled) |
| tuned Qwen3.5-4B | **49.1** | 47.6 (43.2–52.7) | 59.2 | $0.056 (full utilization) | 0.57 s / 0.69 s |

- Passed the val bar (49.1 vs 38.5). On test, the tuned model beat zero-shot Luna (+14.5, CI +9.0 to +20.1). Against 3-shot it is 5.3 points **behind**, CI −9.9 to +0.5, which the rule labels "matched" because the CI touches 0. Flagged to the user as borderline wording.
- **Val and test disagree** (tuned +10.5 on val, −5.3 on test). Val is small (336) and covers only five weeks, with several folders having 0–1 examples, so the val bar is noisy for this task.
- **Where the tuned model loses: folders with little training data that grow in the test period.** consultants (4 train / 32 test), sites (9/16), ees (14/7) and ei (31/19) all score F1 0 for the tuned model, against 50–83 for 3-shot. 3-shot sees the folder *names* plus 3 examples of every folder. The tuned model sees only the instruction (spec 9.5), so a rare folder's name never helps it. It wins on well-populated folders (management, personal, conferences, var, london, universities).
- Candidate fixes for attempt 2 (M4 menu): put the folder list in the tuned prompt, oversample rare folders, raise val_fraction.
- Spend: $4.96 (OpenAI $0.56, Together training $4.00 minimum + endpoint $0.40). Endpoint torn down; none left on the account. All runs: $10.64 of $100.

## Verdict wording, training options, hand-made attempts (Oct 17, 2026)

- **Wording rule changed** (user decision; spec 9.11 and the section 14 done-when updated at the user's request): "beat" / "fell short by X points" only when the 95% CI of the paired test difference excludes 0. Otherwise "ahead by X points" / "behind by X points, not statistically significant" (or "tied" when the point estimate is exactly 0). "Matched" is gone. Enron attempt 1 now reads "behind gpt-6-luna (low, 3-shot) by 5.3 points, not statistically significant".
- **`train.labels_in_prompt: true`** (classify): the tuned model's system prompt lists the labels, in training and at inference, so a rare label's name can help. Default off.
- **`train.class_weighting: none | sqrt | balanced`** + **`class_weight_cap`** (default 5), classify only. Together's SFT format has no per-example loss weight, so weighting is done by repeating rows: weight = (median class size / class size)^p (p = 0.5 sqrt, 1 balanced), clipped to [1, cap], so classes are never down-weighted. Fractional parts are sampled with a seed, and rows are shuffled so copies aren't adjacent. `prepare.json` lists each boosted class's weight and row counts.
- **Hashes stay stable:** the new options enter stage hashes only when set (`Train.hash_view()`), so finished runs aren't re-run or re-polled. Verified on `enron-first` attempt 1.
- **`ft attempt <run> --patch FILE --note TEXT [--after-test]`** adds a hand-designed attempt (same data, splits, baselines) and regenerates the report. It is logged as a human intervention. `--after-test` marks it in the attempts table ("by hand, after seeing test"), and if it becomes the final model the verdict carries a caveat that its test numbers aren't a clean holdout. The final model is still chosen by val. The attempts table now also shows each attempt's test score, labelled as for information only. `ft resume` finishes an interrupted hand-made attempt.

## M4 design notes (to apply when building the retry loop)

Added at the user's request, Oct 17, after the Enron run:

1. **Retrain the final model on train + val.** Once the loop has chosen a config (by val), retrain that config on train + val before the final test evaluation, so the shipped model sees the most recent data. This matters most under time splits, where val is the period just before test. Consequences to handle:
   - the retrained model has no held-out val of its own, so the pass/fail decision stays with the val-selected attempt;
   - the report shows the retrained model's test numbers and says it was retrained on train + val;
   - it costs one more training job (≥ $4 on Together) plus endpoint time, so it goes through the spend guard and estimate like any attempt;
   - class weighting and few-shot example pools are recomputed on the combined data.
2. **Warn about thin or late-arriving categories.** At split time, and in `diagnosis.json` for the fixer, warn when a category has very few training examples (the split stage already warns under 5; M4 should also flag low counts relative to the median), or when most of its examples fall after the train cutoff under a time split (e.g. ≥ 50% in val + test). Enron examples: rice (26 train / 126 later), consultants (4 / 32), sites (9 / 17), credit (23 / 33). These are the categories a tuned model can't learn and where few-shot baselines that see the label names win. The warning should appear in the report's setup section before any results, so it can't read as an excuse afterwards.

### Enron attempt 2 results (Oct 17, 2026)

Attempt 2 (by hand, after seeing test): `labels_in_prompt: true`, `class_weighting: sqrt`, cap 5 (3,022 → 3,196 rows).

| Attempt | Val macro-F1 | Test macro-F1 | Notes |
|---|---|---|---|
| 1 (as written) | **49.1** | **47.6** | final (best on val) |
| 2 (label list + sqrt weighting) | 47.2 | 44.7 | worse on both |

- Per folder, attempt 2 helped ei (F1 0 → 64), consultants (0 → 11), techmemos (40 → 54), var and management. It hurt credit (47 → 36), personal (64 → 59) and weather. sites and ees stayed at 0. The label list didn't let the 4B model use folder *names* the way few-shot Luna does: consultants reached 11 against Luna's 62.
- Attempt 1 stays final (chosen by val), so the report has no after-test caveat on the verdict. Attempt 2 is listed and marked "by hand, after seeing test". Interventions: 1.
- Spend: run $9.43 (OpenAI $0.56, training 2 × $4.00, endpoints $0.87). All runs $15.12 of $100. No endpoints left on the account.

## M5 closed (Oct 17, 2026)

Closed by the user's decision. What the spec's done-when asked for, and what we have:

- **Gmail path:** connector, behavior labels, redaction and on-demand `ft email apply` are built and tested on a synthetic inbox and a fake Gmail service. They have **never run against a real mailbox**, because the user chose not to connect Gmail. The daily schedule is left out at the user's request.
- **Email demo:** Enron instead (folder prediction for one employee's mailbox; see above). The tuned model passed the val bar and beat zero-shot Luna on test, but was behind 3-shot Luna by 5.3 points, not statistically significant. So the spec's "match or beat the best baseline on the time-split test" is **not** met on this demo.
- Generic `maildir` / `mbox` sources, header-block stripping, the new verdict wording, `labels_in_prompt`, `class_weighting` and `ft attempt` came out of this milestone.

## Future work

- **Agent warning on thin or late-arriving categories** (M3 agent / M4 fixer). When drafting or checking a task, the agent should warn when categories have very few training examples or appear mostly late in time (e.g. most examples after the train cutoff under a time split), because the small model can't learn those. Show it before any results. See the M4 design note above for thresholds and the Enron examples.
- **Hybrid routing (test after the MVP).** The tuned model handles inputs it's confident about. Low-confidence predictions and rare categories fall back to the frontier model (e.g. few-shot Luna).
  - **Report:** the combination's accuracy / macro-F1 and its cost per 1,000 requests, next to each model alone, plus the share of traffic routed to the frontier model.
  - **Confidence options:** token log-probs of the label from the tuned endpoint, or a margin between the top two labels. A per-class rule ("always route predictions of folders with fewer than N training examples") is the simplest version.
  - **Threshold choice:** pick it on val only, never test.
  - **Why it might work:** Enron suggests it pays off. The tuned model is stronger on well-populated folders, and 3-shot Luna is stronger on rare or late ones.

## M2: CORD, images and extraction (Oct 17, 2026)

### Verified (docs + live, Oct 17)

- **Together vision SFT** (docs.together.ai/docs/fine-tuning-vlm): Qwen3.5-4B is listed. Images must be base64 data URLs (PNG/JPEG/WEBP, ≤10 MB, ≤10 per example), user messages only. `train_vision` is off by default (only language weights train).
- **Live:** Together **rejects files that mix plain-string and multimodal messages** ("Cannot mix text-only and multimodal messages in the same dataset"), so in image tasks every message (system, user, assistant) is a list of parts. The pipeline's own fixture file then validated (COMPLETED); probe files deleted.
- **Live:** `estimate_price` is **unavailable for multimodal datasets** (`unavailable_reason: multimodal_dataset`), so the spend guard uses our local estimate: text chars/4 + one token per 32×32 image patch. The real cost is recorded from the job afterwards.
- The validator reports `dataset_has_sample_weights`, so Together may support per-example weights. Worth checking before relying on row repetition for `class_weighting`.
- **OpenAI images** (developers.openai.com images-vision): Responses API `input_image` with a base64 data URL; billing is patch-based (32×32), ×1.2 for the GPT-5.6 family. Not documented for gpt-6-luna. **Live (one call, $0.0006):** a 853×1280 receipt (1,080 patches) used ~1,370 image tokens (≈1.27×); strict extraction schema accepted; 768 output tokens (reasoning + JSON) for a 10-item receipt.
- Not yet verified: image input on a Together **dedicated endpoint** for the fine-tuned model (OpenAI-style `image_url` data URL). It is first exercised by the real run. If it fails, predict stops (>2% failures) and the endpoint is torn down.

### Build notes

- **`extract` task type** (`ft/tasks/extract.py`): schema of text/money/number/date/list_of fields, every field required ("" / []). Baselines get the schema in words plus a strict JSON schema; the tuned model gets the instruction only (spec 9.5). Targets and predictions are coerced to the schema (`conform`). Scoring follows spec 9.10: per-example mean over top-level fields, lists by item-level F1, and invalid output (not JSON, or missing a key) scores 0. The report adds a per-field table and field errors with examples.
- **Images:** downscaled once at load to `task.input.max_side` (default 1,280) and saved as JPEG under `data/images/<content hash>.jpg`, so every model sees the same pixels and identical images share a path (the duplicate check catches them).
- **Sources:**
  - `hf_dataset` (images) **streams by default** (`streaming: true`). HF caches point at a temp folder that is deleted afterwards, so nothing lands in `~/.cache/huggingface`, and each photo is shrunk as it arrives. CORD: 2.3 GB streamed, **143 MB** kept in `runs/cord`. (A first, non-streaming attempt was stopped at 468 MB and its cache deleted at the user's request.)
  - `image_folder` + labels file (CSV/JSONL with an optional `split` column).
- **Fixture:** `examples/fixtures/tiny_receipts` (30 synthetic receipts, CORD-shaped labels, same target_map) for offline dry runs.
- Split method now says `predefined` when val and test both come with the data (was "predefined+carved").
- Endpoint-time estimate is per task type (classify 0.4 s/call, extract 2 s/call).

### CORD data (run `cord`, free stages done)

- 1,000 images → train 798 / val 100 / test 100. 2 duplicate train images with conflicting labels dropped (first kept).
- After downscaling: median 853×1280, median 1,080 patches. Fields present: total 970, subtotal 661, tax 444, menu 998. Menu items per receipt: median 2, max 22. Median target JSON: 180 chars.

### Decisions and checks before the CORD run (Oct 17, 2026)

- **`count` is scored as a number** (user decision; deviation from spec 7.2, which has `count: text`). CORD writes counts as "1 x", "x1", "1X"; exact text matching marked a correct read wrong (seen live: label "1 x", Luna "1"). Names stay text (exact after casefold and whitespace), including CORD's own typos.
- **Load hash no longer includes extract field *types*** (they only change scoring), so the type change didn't re-stream 2.3 GB. The prepared `cord` run's stored load hash was recomputed after checking `records.jsonl` is byte-identical, and logged as a `state_migrated` event. The task change also shows in the report's task updates.
- **Qwen3.5-4B for images** (user asked to confirm before running):
  - (1) on Together's vision fine-tuning supported-models list (docs.together.ai/docs/fine-tuning-vlm, rechecked today);
  - (2) deployable: `tg beta models configs Qwen/Qwen3.5-4B` returns `cr_CfHSNAes5gNmGHLdTQSVv` (1× H100, certified 2026-09-22);
  - (3) the docs say fine-tuned "VLM endpoints accept the same vision request shapes as the base models", and base64 data URLs are accepted;
  - (4) **live:** serverless Qwen3.5-9B (same family and serving stack; 4B isn't serverless) took our exact request (OpenAI SDK, `image_url` base64 data URL) and read a CORD total correctly, for $0.0002 (1,114 input tokens vs a 1,080-patch estimate).
  - A fine-tuned 4B on a dedicated endpoint taking images is first exercised by the run itself.

### CORD run 1: tuned model outputs a blank template (Oct 17, 2026)

- **Luna (low):** val 90.2, **test 88.5** field accuracy (total 95, tax 95, subtotal 91, menu items F1 73), 100% valid JSON, $0.29 per 1k receipts.
- **Tuned Qwen3.5-4B:** val 22.0, FAIL. **All 200 val/test outputs are the same string:** `{"total_price":"","subtotal_price":"","tax_price":"","menu":[{"name":"","count":"","price":""}]}`. The 22 points are only fields that are empty in the target.
- **What's ruled out:**
  - training targets are right (798 rows, 719 distinct answers);
  - Together's validator saw a multimodal dataset (`dataset_is_multimodal: true`);
  - the job billed 785,789 tokens, matching our patch-based estimate, so images were tokenized as vision tokens in training;
  - at inference, image tokens reach the endpoint (~1,100 input tokens per call), and there were no errors.
- **Suspicious:**
  - the job ran with **`packing: true`** (several examples concatenated per sequence), a known source of image/text misalignment for VLMs;
  - training loss was 0.057 by step 3 and ~0.01 at the end, too low too early for reading receipt values, so the loss may be dominated by tokens that don't need the image;
  - epoch 1 took 8 min, epochs 2–3 about 25 s each.
  - Not ruled out: the merged fine-tuned model's vision path on the dedicated endpoint (the base 4B with images has never been tested on a dedicated endpoint; only serverless 9B was).
- Spend: $4.38 (OpenAI $0.055, training $4.00, endpoint 3.5 min $0.32). Endpoint torn down. All runs $19.50 of $100.

### CORD option 1: Together's tokenized training data (Oct 17, 2026; free)

Downloaded via `fine_tuning.retrieve_tokenized_dataset` (70.6 MB zstd, 265 MB unpacked; deleted after analysis).

- **Training data was correct.** Packed rows hold 4 receipts each, padded to 131,072 tokens. In all 400 segments in the archive:
  - each segment contains its own image (image-pad runs of 720–1,369 tokens, median 1,080, matching our patch estimate);
  - the answer labels sit after that image;
  - each segment's image matches our training row whose answer length correlates 0.995 with that segment's labelled-token count.
  - No image token carries a label. ~99 labelled tokens per example (the JSON answer).
- **The low loss is explained by dilution:** `token_weights` is 1.0 on every real token (images included, 494K) but labels exist only on ~40K, so the reported loss is roughly 1/12 of the per-answer-token loss: ~0.7 at step 3, ~0.12 at the end. That's normal learning, so **the model was reading receipts during training.**
- The archive holds 100 packed rows (400 examples). The job ran 25 steps/epoch at batch 8 = 200 rows, so the archive seems to be half the tokenized set. Doesn't change the conclusion.
- **Conclusion: option 1 does not point to packing.** It points to serving. The fine-tuned (merged) model on the dedicated endpoint ignores the image although image tokens arrive (~1,100 input tokens/call).
- **Serving alternatives checked (free):**
  - Together added `--attach-adapter` on 2026-10-13 (serve the LoRA on top of a base deployment instead of a merged copy), but it needs a config with `adapter_mode` dynamic/fixed.
  - Qwen3.5-4B's only config is `disabled`. Qwen3.5-9B has `dynamic` only on 4× H100 (~$22/h). So the adapter path isn't practical here.
- Per the user's rule (retrain with packing off only if option 1 pointed to packing), no retrain yet.

### Collapse check (Oct 17, 2026)

- `evaluate.collapse_check`: flags "collapsed: the model is ignoring its input" when ≥50% of outputs are one identical string **and** that share beats the most common target's share by ≥25 points (n ≥ 10), so a skewed but honest classifier isn't flagged.
  - Computed for every predictions file (baselines too) as `metrics.*.json["collapse"]`.
  - Shown at the top of the report's verdict.
  - Written to the new **`attempts/<n>/diagnosis.json`** (val only, never test) as a plain-language flag for the M4 fixer, together with an "invalid outputs > 20%" flag.
- The CORD report now flags attempt 1 (100% identical outputs vs a 2% top target). Re-evaluating was free.

### CORD paused: dedicated Qwen3.5-4B endpoints don't see images (Oct 17, 2026)

**Base-model check ($0.27, 3 min, torn down):**
- Base `Qwen/Qwen3.5-4B`, no fine-tune, deployed on a dedicated endpoint (the only config, `cr_CfHSNAes5gNmGHLdTQSVv`, 1× H100, sglang). Sent 5 CORD val receipts with the same request we use everywhere (OpenAI SDK, `image_url` base64 JPEG data URL, `enable_thinking: false`). Results in `runs/cord/base_probe.json`.
- **"What is the total?"**: `10.00` for all 5 receipts (true totals 45,500 / 23.000 / 20,000 / 28,000 / 41,000).
- **Extraction prompt:** the same invented receipt ("Coke", 10.00) or an empty one each time, and once: *"The image appears to be corrupted or contains only noise."*
- Image tokens do arrive (329–1,238 input tokens per call, varying with image size).

**Which part failed: Together's dedicated-endpoint serving of images for Qwen3.5-4B.** Not our data, training or request format:
- **Data and training worked:** Together's own tokenized data shows correct image/answer alignment, and the per-answer loss fell from ~0.7 to ~0.12.
- **Our request format works:** the identical request to serverless Qwen3.5-9B read a CORD total exactly.
- **On the dedicated 4B endpoint, both the base model and our fine-tune behave as if the image were noise.** The fine-tune's blank template is what a model trained to read receipts emits when it can't see one.
- The deploy config's certification lists only text benchmarks, so image serving was probably never exercised.
- Not tested: whether the fine-tune would read receipts on a working vision server. It probably would, given the training loss, but nothing on Together can show it today (no adapter-capable 4B config, and fine-tunes aren't served serverless).

**Status:** CORD is paused. CORD spend: $4.65 (Luna $0.06, training $4.00, endpoints $0.59). All runs: $19.77 of $100. The packing-off retrain was skipped by the user's decision (training was shown to work). Nothing left running on Together.

**Kept from M2** (works, tested, reusable): the `extract` task type and field scoring, image loading with streaming from Hugging Face, vision request formats for OpenAI and Together, Together's all-parts multimodal training format, the collapse check, and `diagnosis.json`. Luna (low) reached 88.5 test field accuracy on CORD at $0.29 per 1k receipts. That's the bar any later vision attempt has to meet.

## Future work (continued)

- **Retry vision on another provider after the MVP** (e.g. Fireworks, which supports Qwen-VL fine-tuning), behind the same `Trainer` / `Server` / `TunedModel` interfaces. Before paying for training, first check that the provider's *base* model reads a receipt on the exact serving path we'd use: the check that would have caught this on Together for $0.27.
- **Add that check to the pipeline:** for image tasks, a cheap base-model smoke test on the serving path (a few val images, look for collapse) before the paid training job.

## M3: the agent (Oct 17, 2026)

### What's built

- **`ft plan <dir>`** runs the agent on a raw folder and writes `<dir>/task.yaml`, or declines (exit 2). **`ft run <dir>`** on a folder with no task.yaml: plan → write task.yaml → create the run → free local stages (load, profile, split, label) → **human checkpoint** → pipeline. `--yes` skips the checkpoint; `--dry-run` uses a heuristic fake agent.
- **Checkpoint** (spec 12): input → output, label source, real split sizes and method, base model, metric, baselines, estimated cost per line, split warnings, and 10 random **test** examples with expected answers. `[a]pprove / [e]dit / [q]uit`. Edit opens `$EDITOR` on `<dir>/task.yaml`, revalidates, replaces the run's task and reruns the free stages. Quit leaves the run resumable before any paid stage.
- **Tools** as spec 12: `get_profile`, `sample_rows`, `value_counts`, `view_images` (≤6, shrunk to 1,024 px, sent as image parts of the tool result), `list_base_models`, `estimate_cost`, `validate_task`, `submit_task` (must validate; one-line rationale per decision: task, label source, base model, metric, split, baselines), `decline` (reason + suggestion).
  - `validate_task` checks the data, not just the schema: files, input/label/time columns, declared labels vs. data, ≥2 classes, deployable base model, that the model serves the input kind, baseline prices, estimated cost ≤ budget.
  - `estimate_cost` mirrors the split rules and uses prices.yaml and models.yaml (errs high).
- **Loop:** ≤30 turns; 2 nudges if the model stops calling tools; per-plan spend cap `FT_AGENT_MAX_USD` (default $1) plus the all-runs budget. Every tool call is an `agent_tool` event and each decision a `decision` event (the report already shows these as agent rationale). Transcript in `runs/<id>/agent/transcript.jsonl`.
- **Spend:** a plan writes `runs/<id>/plan.json`. The all-runs total counts it until the run starts, then the entries move into `state.json` (stage `agent`), so nothing is counted twice.
- **models.yaml `serves_images`** (default false): `list_base_models("image")` returns nothing today, so the agent declines image data with the reason "no deployable model serves images".

### Differences from spec 12

- **Agent model: `gpt-6-luna`, reasoning effort medium** (OpenAI-only decision; spec: claude-opus-5-5). Responses API, strict function tools, `store=false`. Each turn replays the model's output items, including encrypted reasoning, as OpenAI's reasoning guide asks for tool loops; nothing is stored at OpenAI.
- **The test set stays held out differently.** The spec splits after approval. Here the free local stages (including split) run *before* the checkpoint, so it shows real split sizes and real test examples. The agent itself never sees test rows: `sample_rows` / `value_counts` refuse test files, and it sees them only as column summaries in the profile (as the spec allows).
- **Baselines (user decision, Oct 17):** for classification, one zero-shot and one few-shot baseline, both Luna at low effort (the comparison customers make). Few-shot uses N train examples per label, N = 3, or fewer when there are many labels so the prompt holds about 100 examples. Extraction gets zero-shot only. Enforced in code (see below).
- **Frontier labeling** isn't available yet (label stage), so with no recorded answers the agent declines and suggests labeling a sample.
- **Done-when, CORD (agreed with the user, Oct 17):** CORD is paused (no deployable model serves images), so instead of a passing run the agent must **decline** a raw CORD folder as "good fit, but no available model on this provider can serve it", suggesting another provider.
- **Done-when, BANKING77 → LEDGAR (user decision, Oct 17):** the paid end-to-end raw-folder run uses a dataset we haven't configured before, LEDGAR (LexGLUE contract clause types), instead of re-running BANKING77. The agent still plans raw BANKING77 (no run), so its plan can be compared with the hand-written task.yaml.
- **Two kinds of decline** (user request): `decline(fit=...)` is either `not_a_fit` ("not a good fit for fine-tuning") or `no_available_model` ("good fit, but no available model on Together can serve it").
  - `no_available_model` is refused unless `list_base_models` returned nothing for some input kind in the same session.
  - Its suggestion always mentions trying another provider: if the agent's text doesn't, the tool appends one line.
  - The kind is stored in plan.json and the decision event, and printed by `ft plan`.
- **Cost rule added to the prompt:** keep the estimate under $15, using `train.max_examples` (stratified) when training dominates. Test sets over 5,000 rows get `test_max: 5000` (see the LEDGAR result below: the agent didn't follow this one).

### M3 live agent runs (Oct 17, 2026)

Agent: gpt-6-luna, medium effort. **Total agent spend $0.008 for five plans** (3–7 turns each, ~2–6K input tokens per turn). The pre-run guess of $0.10–0.20 was ~20× high.

| Folder | Outcome | Turns | Cost |
|---|---|---|---|
| `examples/fixtures/too_small` | declined, not a good fit: "only 60 labeled … below the 500-example minimum"; suggests few-shot prompting or more labels | 4 | $0.0005 |
| `examples/fixtures/open_ended` | declined, not a good fit: "unique, long-form answers … no consistent targets"; suggests a frontier model and a grading rubric | 3 | $0.0006 |
| `data/cord_raw` (998 receipts: hard-linked images and a labels.jsonl with JSON ground truth and a split column) | declined, good fit but no available model on Together: "otherwise a suitable fine-tuning task … no deployable model that serves image inputs"; suggests another provider, and a frontier vision model meanwhile | 3 | $0.0009 |
| `data/banking77` | submitted | 7 | $0.002 |
| `data/ledgar` | submitted, stopped at the checkpoint | 7 | $0.002 |

**BANKING77: the agent's task.yaml vs the hand-written `examples/banking77/task.yaml`.**
- **Same:** csv source, classify `text` → `category` with labels auto, dataset targets, predefined splits (val carved from train, max 1,000), Qwen3.5-4B with 9B as fallback, LoRA, 3 epochs, lr 1e-4, accuracy, pass bar within 2 points of the best baseline, concurrency 16, 3 attempts.
- **Different:**
  - name `banking77-routing`;
  - no `files:` mapping (the files are picked up by name; same effect);
  - instruction adds "Return exactly one category label.";
  - `test_max: 3083` instead of 3080 (both keep the whole 3,080-row test set);
  - one baseline (Luna low) instead of three (low, medium, low 3-shot);
  - budget $12 instead of $30 (twice its $5.81 estimate).
- The two baseline differences are what the prompt asks for. The hand-written file has extra baselines added in M1.

**LEDGAR** (`examples/ledgar/fetch.py` → `data/ledgar/{train,validation,test}.csv`: 60,000 / 10,000 / 10,000 provisions, 100 clause types, 28 MB streamed, no HF cache left behind):
- **Agent's plan:**
  - classify `text` → `label`, accuracy (top class 5.3%);
  - predefined splits mapped through `files:`;
  - `test_max: 10000`;
  - `max_examples: 40000` to stay under $15;
  - budget $27.
- **Free stages:** train 58,562 / val 1,000 / test 10,000; 1,505 duplicates dropped.
- **Checkpoint estimate $13.27:** baseline $1.94 (11,000 calls), training $10.00 (~9.8M tokens × 3 epochs), endpoint $1.33 (~15 min).
- **The agent ignored the new rule "test sets over 5,000 rows → test_max 5000"** and kept all 10,000, with the rationale "retain all 10,000 test examples". Prompt rules aren't enforced; `validate_task` doesn't check this one.
- **Alternatives (free estimates):**
  - test_max 5000: $12.20;
  - test_max 5000 and max_examples 20000: $7.20.
- Run `20261017-1702-ledgar` is waiting at the checkpoint (`ft resume 20261017-1702-ledgar`).

**Together bug report** drafted for the user to send: `docs/together-bug-report.md` (dedicated Qwen3.5-4B endpoints treat images as noise, even the base model).

### Agent rules enforced in code (Oct 17, 2026)

The LEDGAR plan broke a prompt rule (it kept 10,000 test rows), so at the user's request the mechanical rules moved into `ft/agent/rules.py`.
- `apply_rules` runs on every draft in `validate_task`, `estimate_cost` and `submit_task`.
- **Corrected** (each correction goes back to the agent, and on submit is logged as a `decision` event, "rule applied: …", which the report shows after the agent's own rationale):
  - base model and fallback: the smallest deployable model for the input kind, then the next one;
  - baselines: zero-shot plus few-shot for classify, zero-shot for extract;
  - `test_max`: a given test set is kept whole up to 5,000 rows and capped at 5,000 above that;
  - primary metric from the train label shares (accuracy ≤ 40% top class < macro_f1; field_accuracy for extract);
  - pass bar, concurrency and max_attempts;
  - `budget.max_spend_usd`: max(10, ⌈2 × estimate⌉).
- **Rejected:**
  - an estimate over `FT_AGENT_MAX_RUN_USD` (default $15); the error names the biggest cost line and points at `train.max_examples`;
  - an estimate over what's left of the all-runs budget;
  - image data when no model serves images (decline instead).
- **Left to the agent's judgment:** task type, input and label columns, time splits, instruction, `max_examples`, and declining.
- **The submitted task.yaml is the corrected one.** It's re-dumped as YAML, so the agent's formatting is lost when something was corrected.
- Human edits at the checkpoint aren't put through the rules: the human decides.

### Time estimates (Oct 17, 2026)

`estimate_run` now gives minutes per line and a total, shown at the checkpoint. Stages run one after another, so the total is their sum. Fitted to measured runs:
- **Frontier baseline:** the larger of (calls × 1.6 s / concurrency) and (calls × counted tokens / throttle TPM).
  - Counted tokens = input + output allowance, cached input included; the throttle runs at 80% of `FT_OPENAI_TPM`, default 2M.
  - Enron's 24k-token 3-shot prompt ran at ~1 call/s, TPM-bound, matching this.
- **Training:** 9 min + epoch-tokens / 350K per min (BANKING77 13.5 min, Enron 18.6 and 19.6 min).
- **Endpoint:** provisioning + calls, as before.

### Fixed: resume skipped a quit checkpoint; edit crashed (Oct 17, 2026)

- **Resume:** `ft resume` on a run quit at the checkpoint went straight to the paid stages. It reached the baseline spend prompt and aborted there on bad input, so $0 was spent. Now it shows the checkpoint again unless it was approved (`--yes` approves).
- **Edit:** the checkpoint's `[e]dit` called `typer.edit`, which doesn't exist; it's now `click.edit`.
- Both now have tests (147 tests pass).

### LEDGAR at the checkpoint, second version (Oct 17, 2026)

- **Checkpoint edit (user's choices):** `max_examples: 20000`; `test_max: 5000` (also what the rule gives); zero-shot plus 1-shot baselines (the new rule); budget $20 (the rule).
- The agent's original submission is kept in `runs/20261017-1702-ledgar/agent/task.agent-submitted.yaml`.
- **Split:** train 58,562 / val 1,000 / test 5,000.
- **Estimate $9.55, about 2 h 31 min:**

| Line | Cost | Time |
|---|---|---|
| Luna low zero-shot (6,000 calls) | $1.06 | ~10 min |
| Luna low 1-shot (6,000 calls, ~19.5k-token cached prompt of 100 examples) | $2.35 | ~78 min, TPM-bound |
| Training (~4.9M tokens × 3 epochs) | $5.00 | ~51 min |
| Endpoint | $1.14 | ~12 min |

- **The few-shot baseline sets the pace** at the default 2M TPM. A higher account limit (`FT_OPENAI_TPM`) shortens it proportionally.

### Spend guard for cached prompts (Oct 17, 2026)

- **What happened:** the first LEDGAR start ran the zero-shot baseline (6,000 calls, $0.64), then stopped before the 1-shot one.
  - The guard checks a stage's worst case, and for a cached few-shot prompt that's the no-cache cost: $26.57 against $3.25 cached. That's over the $20 cap.
  - The checkpoint estimate (and so the budget rule) assumed caching, so the two disagreed.
- **Fix** (generic, `baseline.py`): when caching is expected, the stage is checked against the cached estimate plus the no-cache cost of the first 400-call chunk.
  - After every chunk, `run_predictions(after_chunk=…)` projects the stage total from the billed usage so far.
  - If the projection would pass the run cap, it stops with `SpendStop` and records the partial spend.
  - So a caching failure costs at most about one chunk at full price, not the whole stage.
- **Test:** a dry-run test with `FakeOptions(frontier_cache=False)` stops partway; with caching on, it passes. 149 tests pass.
- **Rate limit checked live** (one call, ~$0.000004): the account allows 2,000,000 tokens and 5,000 requests per minute. That equals the `FT_OPENAI_TPM` default, so it's unchanged.

### LEDGAR: baselines done, training blocked by the Together balance (Oct 18, 2026, 01:55 UTC)

- **Baselines** (accuracy; macro-F1 in brackets):

| Baseline | Val | Test | Cost |
|---|---|---|---|
| Luna low, zero-shot | 74.5 | 73.64 (60.0) | $0.64 |
| Luna low, 1-shot | 74.1 | 73.44 (61.7) | $1.29 |

  - The 1-shot run read 99.2% of its input tokens from cache, and its test calls took 69 min (TPM-bound, as estimated).
  - The baseline stage took 83 min in all.
- **Training didn't start:** Together returned 402 `insufficient_balance` ("required combined balance and credit limit: 4.00 USD").
  - No job and no endpoint were created; only the training file was uploaded (`file-123037bc…`, reused on resume).
  - Together's own estimate for the job: $5.77 (ours: $5.00).
- **Fix:** a 402 when creating a job now raises a `SpendError` with a plain message ("add credits, then resume") instead of a traceback.
- Run spend so far $1.93; resume with `ft resume 20261017-1702-ledgar --yes` after adding credits.

### LEDGAR: first agent-planned paid run, passed (Oct 18, 2026)

- **Run `20261017-1702-ledgar`:** raw folder → agent plan (fixed by the rules and the user's checkpoint edit) → baselines → train → serve → report. No interventions after approval.
- **Restarts during the run:**
  - the cached-prompt spend guard stopped it before the few-shot baseline (fixed above);
  - Together's balance blocked training (the user added credits).
- **Results** (5,000 test provisions, 100 clause types):

| Model | Test accuracy | Macro-F1 | p50 latency | Cost / 1k |
|---|---|---|---|---|
| Luna low, zero-shot | 73.64 | 60.0 | 1.18 s | $0.106 |
| Luna low, 1-shot | 73.44 | 61.7 | 1.48 s | $0.215 |
| **Tuned Qwen3.5-4B** (20k examples, 3 epochs) | **84.06** | **73.6** | **0.48 s** | **$0.046** |

  - **Val:** 82.2 against a bar of 72.5, so PASS.
  - **Test:** beat Luna low by +10.4 points (95% CI [+9.3, +11.5]) and the 1-shot baseline by +10.6.
  - Invalid outputs 0.16%. No collapse.
- **Spend:** $6.85 (agent $0.002, baselines $1.93, training $4.00 as billed by Together against its own $5.77 estimate, endpoint 10 min $0.92).
  - The checkpoint estimate was $9.55: the few-shot baseline cost less with caching ($1.29 vs $2.35), and training was billed lower.
  - All runs: $26.63 of $100.
- **Endpoint torn down;** `ft status` shows none live on the account.
- **Time:** baselines 83 min (as estimated), training 21 min end to end (17 running) against an estimated 51, endpoint ~10 min.
  - The training time model was refit (12 min fixed + 1M epoch-tokens/min), so it now errs high by ~25% on LEDGAR rather than 2.4×.

## M3 approved (Oct 18, 2026)

Approved. `.env`, `data/` and `runs/` stay ignored.

## Hybrid routing, simulated on saved predictions (Oct 18, 2026; free, no API calls)

- **Command:** `ft hybrid <run>` (`ft/hybrid.py`), writing `runs/<run>/hybrid.json`.
- **Rule:**
  - The tuned model answers everything. If its answer is a label whose tuned **val** F1 is below a threshold, the few-shot Luna baseline's saved answer is used instead.
  - An invalid output, or a label never predicted on val, counts as F1 0.
  - The threshold is picked on val only: best val primary metric, with ties going to less routing.
- **Cost per 1k:** the tuned cost at full endpoint utilization (every input), plus Luna's billed cost for the routed inputs.

| Run (primary metric) | Picked on val | Routed (test) | Tuned test acc / macro-F1, $/1k | Few-shot Luna | Hybrid | Hybrid − tuned (primary, 95% CI) |
|---|---|---|---|---|---|---|
| BANKING77 (accuracy) | no routing | 0% | 92.27 / 92.33, $0.0419 | 86.95 / 87.01, $0.0752 (3-shot) | 92.27 / 92.33, $0.0419 | 0 |
| Enron (macro-F1) | no routing | 0% | 59.17 / 47.62, $0.0561 | 61.07 / 52.93, $0.3387 (3-shot) | 59.17 / 47.62, $0.0561 | 0 |
| LEDGAR (accuracy) | route F1-0 answers (4 labels + invalid) | 0.94% | 84.06 / 73.61, $0.0457 | 73.44 / 61.65, $0.2152 (1-shot) | 83.94 / 72.76, $0.0478 | −0.12 [−0.30, +0.06] |

- **On val, routing never helps by more than one example.** Each val curve falls as the routed share grows: BANKING77 91.2 → 85.8, Enron 49.1 → 38.5, LEDGAR 82.2 → 74.1. On a tuned model's weak labels, few-shot Luna isn't better on those same inputs.
- **Enron is the one case where routing would have helped on test** (reference only, not used to choose):
  - routing labels with val F1 < 75 sends 48% of inputs to Luna and gives a test macro-F1 of 50.1, against 47.6 for tuned;
  - val said the opposite (44.7 against 49.1).
  - That's the same time-drift problem as before (categories that arrive late), and val can't see it. It supports the M4 design note: warn on thin or late-arriving categories.
- **Conclusion:** routing on the predicted label's val F1 adds nothing here. A confidence signal per input (the tuned model's token logprobs, which we don't save yet) is the obvious next thing to try, after the MVP.

## After-MVP ideas (continued)

- **Route by per-answer confidence (token log-probabilities).** The label-F1 rule didn't help (above). The next signal to try is the tuned model's own confidence per input: the log-probability of its answer's tokens, or the margin to the runner-up. Below a threshold picked on val, route to few-shot Luna. Test with `ft hybrid` on saved predictions.
- **Saved from now on:** the predict stage saves `logprobs` on every tuned prediction when the provider returns them (`[{t, lp, top}]`, the first 64 output tokens, top 5 per token), so this can be tested later without rerunning.
  - Together's docs (checked 2026-10-18) take `logprobs` as an integer 0–20 and return `tokens` / `token_logprobs` / `top_logprobs`. OpenAI-shaped `content` lists are handled too.
  - Not yet seen live on a dedicated endpoint: if a request with `logprobs` is rejected, it's dropped for the rest of the run, so answers never suffer. The next paid run will show which happens.
  - Existing runs (BANKING77, Enron, LEDGAR) have no logprobs saved.

## M4: the retry loop (Oct 18, 2026)

### What's built

- **Loop** (`pipeline.run_pipeline`): after a failed attempt, if attempts remain:
  1. the fixer picks fixes;
  2. code builds the next attempt's patch;
  3. the next attempt's cost is estimated (training as the train guard will see it, the endpoint as `deployed` estimates it);
  4. if the estimate fits what's left of the run's cap, the attempt runs; otherwise the loop stops.
  - Every stop is a `retry_stopped` event, with the reason shown in the report.
  - Attempts already planned (resume, or added by hand) are run, not re-planned.
  - Final model = best attempt by val (unchanged).
- **Fixer** (`ft/fixer.py`): gpt-6-luna at medium effort, a separate prompt, two tools (`submit_fixes`, `stop_retrying`), at most 6 turns. Invalid choices go back to it with the reason. Spend is recorded under stage `fix`.
- **What the fixer reads:** `diagnosis.json` plus the history of every attempt (config, fixes, val result, which attempt it built on) and the limits. Val only, never test (a test checks this).
- **diagnosis.json** (spec 13): val metrics, worst 15 classes, top 8 confusions with 2 examples each, field errors (extract), invalid-output examples, the loss curve (≤30 points), training rows used / available, label-count spread, and thin / late-category warnings (train vs val).
- **Menu → patch:**
  - `more_epochs` (≤ 6 total), `change_learning_rate` (Together's model limits clamped to 1e-6 to 1e-3), `use_more_data` (raise or remove the cap), `bigger_model` (next fallback);
  - `labels_in_prompt` and `class_weighting` (existing generic options);
  - `stop`.
  - The patch can only touch `train.*`.
  - The next attempt builds on the best attempt so far by val, not necessarily the last one.
- **Report:** the attempts table marks "by the fixer, from attempt N", shows the changes in words, and gives the stop reason.
- **Thin / late-arriving categories** (M4 design note 2): the split stage warns about classes under 5 train examples, classes under 20% of the median class, and, under a time split, classes with ≥ 50% of their examples in val + test. These appear in the report's setup section before any results. On Enron it flags exactly consultants 4 / 32, rice 26 / 126, sites 9 / 17 and credit 23 / 33.
- **Dry-run done-when:** `tests/test_retry.py`. `FakeOptions(tuned_q_from_config=True)` makes fake accuracy follow learning rate, epochs and rows. The weak config fails attempt 1; the fake fixer changes the learning rate, epochs and data; attempt 2 passes (+30 points).
- **Fixes found while building:**
  - The next-attempt estimate had scaled Together's $4 minimum charge as if it were usage-based ($39 for a ~$5 attempt). Only a bill above the minimum is scaled now.
  - The next-attempt estimate had used the base attempt's billed endpoint minutes, below what the serve guard checks. It now uses the same estimate.
  - Fake tuned default quality is now 0.93 (tuned ahead of the frontier fake, as in real runs). Single-attempt tests use a perfect fake so they don't trip the loop.
- **Trainer** gained `limits(model)` (Together `fine_tuning.model_limits`, free). Qwen3.5-4B: lr 1e-8 to 1.0, 20 epochs.

### Differences from spec 13

- **`add_examples` and `drop_suspect_labels` are on the menu but marked unavailable.** Both need frontier labeling, which isn't built. Choosing one returns an error to the fixer.
- **`labels_in_prompt` and `class_weighting` are added to the menu.** They're generic training options built earlier (Enron).
- **Practical learning-rate range 1e-6 to 1e-3** on top of the provider's limits.
- **Not built yet: M4 design note 1** (retrain the chosen config on train + val before the final test).

### Done-when run (proposed, not run)

`examples/banking77-weak/task.yaml`: max_examples 300, 1 epoch, lr 1e-6. Baselines: Luna low zero-shot and 1-shot (the default comparison). Budget $20.
- **Estimate:** baselines $1.25 once; each attempt ~$5.07 ($4.00 training minimum + ~$1.07 endpoint); fixer a few cents.
- **Totals:** pass at attempt 2 ≈ $11.4, about 70 min. All 3 attempts ≈ $16.5, about 95 min.

### Retrain on train + val (M4 design note 1, built Oct 18, 2026)

- **Option `train.retrain_with_val`:** `null` (default) = on for `by_time` splits, off otherwise; `true` / `false` force it.
- **After the loop**, the best attempt by val is chosen as before. If it **passed**, and the estimate fits the run's remaining budget, a final attempt `by: retrain` repeats its config on train + val.
  - Prepare adds the val rows; class weights follow the combined rows.
  - Predict and evaluate run on **test only**: the retrain has seen val, so it has no val score.
  - `state.retrained_attempt` records it. A resumed run reuses it and never retrains twice for the same chosen attempt.
- **Pass/fail stays with the val-chosen attempt.** The report's tuned row, verdict sentence and loss curve use the retrain's test numbers, labelled "(attempt N, retrained on train + val as attempt M)".
  - A note under the attempts table gives both test scores and says why pass/fail isn't the retrain's.
  - A skipped retrain (the pick didn't pass, or it won't fit the budget) is noted with the reason.
- **Cost:** one more training job (≥ $4 on Together) plus test-only endpoint time.
- **Not part of the BANKING77 done-when run:** its split is predefined, not by time, so the retrain is off.
- **Tests:** a time-split dry run retrains and reports the retrain's test numbers with attempt 1's val decision; a random split doesn't retrain unless forced; a failing pick isn't retrained. 160 tests pass.

### M4 done-when run 1 stopped: a logprobs race emptied every tuned answer (Oct 18, 2026)

- **Run `m4-banking77-weak`:**
  - **Baselines:** Luna low val 80.38 / test 81.98; 1-shot val 84.68 / test 85.78. Bar 82.68.
  - **Attempt 1** (the weak config): val 0.0.
  - **The live fixer picked exactly the right three fixes:** lr 1e-6 → 2e-4, remove the 300-row cap, epochs 1 → 3.
  - **Attempt 2:** val 1.5.
  - The fixer then chose `labels_in_prompt` for attempt 3, and I stopped the run before its file was uploaded. No job, no endpoint. Run spend $9.63; all runs $36.26.
- **Cause (my bug, in this morning's logprobs change):**
  - In both attempts the endpoint rejected the requests that asked for `logprobs`. No logprobs were saved, and the rejection text wasn't recorded.
  - The fallback that drops a rejected option read **shared** flags. With 16 calls in flight, the first one dropped logprobs, and the rest found it already gone and dropped `enable_thinking: false` as well.
  - With thinking back on, every call spent its 32-token budget thinking and returned no answer text: 998 / 999 empty in attempt 1, 984 / 999 in attempt 2.
  - Training was fine; only the predictions were broken.
- **Fixes:**
  - **(1) Only drop what this request sent.** Each request drops only the option it actually sent, and records the endpoint's rejection message as a warning event. A test reproduces the 16-call burst.
  - **(2) Empty answers stop the loop.** When more than 50% of val calls return no answer text, diagnosis.json sets `serving_problem` and the loop stops with that reason instead of calling the fixer: no training fix can help. Tested.
- **Lesson:** an untested request option was shipped to a paid path. The logprobs fallback was meant to be safe and wasn't, because of concurrency. Note also that the fixer reads its diagnosis faithfully: with broken evidence it made a reasonable but useless choice. The serving check now stops the loop before that happens.

### Check: attempt 2 re-scored with the fix ($0.27, Oct 18, 2026)

- **Re-score:** attempt 2's model (`m4-banking77-weak`) was redeployed with the fixed client and scored on val only (`attempts/2/val.rescore.predictions.jsonl`; the original predictions were kept).
  - **91.89** val accuracy (M1: 91.19), macro-F1 92.09, invalid 0.1%, 0 empty answers, 4–6 output tokens.
  - So the earlier 1.5 was entirely the serving bug.
- **The rejection text** was "invalid type: integer `5`, expected a boolean". The dedicated endpoint takes the OpenAI form, not the integer in Together's docs. Requests now send `logprobs: true, top_logprobs: 5`, still dropped safely if rejected.
- **Fresh done-when run `m4-banking77-weak-2` started**, same task file.

### M4 done-when: passed (run `m4-banking77-weak-2`, Oct 18, 2026)

- **Start:** the weak config (300 rows, 1 epoch, lr 1e-6). Bar: within 2 points of Luna 1-shot's val 84.38, so 82.38.
- **Attempt 1:** val **0.8** (99% invalid: the barely-trained model answers in free text). It was a real failure this time: no empty answers, and logprobs were saved on all 999 val answers (the OpenAI form is accepted).
- **The fixer** ($0.0005) chose:
  - lr 1e-6 → 2e-4 ("far below the usual LoRA range");
  - label list in the prompt ("nearly all predictions are invalid free-form outputs");
  - remove the cap (300 → all 8,993 rows).
  - It kept 1 epoch.
- **Attempt 2:** val **88.69 → PASS.** Test 90.55 accuracy / 90.53 macro-F1, beating Luna low (82.44) by +8.1 [+7.0, +9.4] and Luna 1-shot (85.75) by +4.8 [+3.7, +5.9]. p50 0.39 s against ~1.1 s; $0.038 per 1k against $0.041–0.106.
- **Zero interventions after start.** Run spend **$9.50** (baselines $0.60, training 2 × $4.00, endpoints $0.90, fixer $0.0005). Endpoints torn down; none live on the account.
- **All runs: $46.02 of $100.** That includes the stopped first run ($9.63 + the $0.27 check).

## M4 approved; M6: logs mode (Oct 18, 2026)

M4 approved by the user and pushed (`45213d8`, `81cb812`, `c6723e8`). The train + val retrain (built during the M4 done-when run) is in `81cb812`.

### What's built (M6, free so far)

- **Log reader** (`ft/connectors/logs.py`): the native OpenAI-compatible log shape (spec 15.1), raw OpenAI chat-completions responses, and Anthropic Messages shapes (system as a string or blocks, content blocks, `cache_read_input_tokens`).
  - Normalized fields: `last_user_message`, `system_prompt`, `conversation`, `response_content`, plus model, usage, latency and timestamp.
  - Bad lines are skipped and counted.
  - A truth sidecar (`id` + `truth` / `label` / `target`) is joined by id.
- **`source: logs`** in load: the logged response is the target; truth rides along in `meta` for evaluation only (a test checks it's never the target). The label stage accepts `targets.from: logs`. The terms reminder prints, as before.
- **Evaluate:** in logs mode every metrics file gets `truth` (accuracy against the sidecar, with a CI). `logged_metrics` scores the logged model as a pseudo-baseline: agreement 100 by definition, its truth accuracy, its latency, and its cost per 1k from the logged usage at prices.yaml rates.
- **Report:** the logged model's row ("pseudo-baseline") and a truth-accuracy column. The verdict sentence gives agreement and both truth accuracies.
- **`ft simulate-logs <task> --out <dir> [--model gpt-6-luna] [--n 5000] [--effort low]`:**
  - an app-style prompt ("You power the automatic routing in a customer-facing app…") built from the task's instruction and labels;
  - one log per distinct input;
  - estimate, all-runs budget check and confirmation before calls;
  - spend recorded in `runs/sim-…/plan.json` so the all-runs total counts it;
  - writes `logs.jsonl` + `truth.jsonl`.
- **`examples/banking77-logs/task.yaml`** follows spec 15.3, with OpenAI in place of Claude: agreement ≥ 95 on val, no baselines.
- **Dry run at full size** (5,000 BANKING77 messages): 3,600 / 400 / 1,000 split, ~178k training tokens, all stages and report fine. 167 tests pass.

### Plan for the paid M6 run and the final proof runs (proposed, not run)

- **M6 done-when:** `ft simulate-logs examples/banking77 --out data/banking77-logs --n 5000` (~$0.94), then `ft run examples/banking77-logs` (training $4 minimum + endpoint ~$1, about $5). ≈ $6, up to $16 with two retries.
- **Final proof set** (MVP: every dataset from raw data to a report with zero interventions):
  - BANKING77 from the raw folder through the agent: ~$6.3.
  - CORD raw: agent declines: ~$0.001.
  - Simulated logs: the M6 run doubles as its proof run.
  - Email: Enron, fresh: baselines $0.56 + training $4 + endpoint ~$0.45, + train + val retrain (time split) ~$4.5. About $9.5.
- **Expected total ≈ $22,** taking all runs from $46.02 to ~$68.
- **Contingency:** $10 for retries.
- **Proposed hard stop:** `FT_TOTAL_BUDGET_USD=85`.

### M6 done-when run: not met (run `m6-banking77-logs`, Oct 18, 2026)

- **Logs:** `simulate-logs` wrote 5,000 BANKING77 calls to Luna low with the app-style prompt, for $0.37 (estimate $0.94). The logged answers match truth on 81.3%.
- **Run:** 3,600 / 400 / 1,000 split; bar ≥ 95 val agreement.
  - **Attempt 1** (4B): val agreement 90.25.
  - **Fixer:** labels_in_prompt + sqrt class weighting.
  - **Attempt 2:** **91.50** (the best).
  - **Fixer:** bigger_model.
  - **Attempt 3** (9B, the first fine-tuned 9B deploy, which worked on 1× H100): 90.75.
  - The loop stopped at max_attempts.
- **Test** (attempt 2): agreement **91.8**; truth accuracy 80.2 (tuned) vs 81.7 (logged Luna); $0.036 vs $0.075 per 1k; p50 0.37 s vs 1.28 s. The report shows agreement and truth accuracy (that half of the done-when is met).
- **Spend:** $13.22 (3 training jobs, 3 endpoints). The run was paused once on the Together balance; the user added credits.
- **Why it stalls around 91** (free analysis of the 82 test disagreements): 22 tuned right / Luna wrong, 37 Luna right / tuned wrong, 23 both wrong. The gap is mostly ambiguous messages where Luna's own answers are inconsistent (81.7% vs truth). A fixed 95% bar may be above what imitating this labeller allows.
- **Proposed (not run):** measure Luna's test-retest agreement on the 400 val messages (~$0.05) to see whether 95 is reachable at all, then either set the bar relative to it or add more logs.

### Luna self-consistency check ($0.03, Oct 18, 2026)

- **Test-retest:** Luna re-asked the 400 M6 val messages with the exact logged prompt and settings agrees with its own logged answers on **93.75%** (25 changed). Accuracy vs truth was 84.0 the first time and 83.0 the second. Saved in `runs/m6-banking77-logs/luna_retest.json`.
- **So a fixed 95% agreement bar is unreachable for these logs:** a second sample of Luna itself scores ~93.75.
  - A model that always gives Luna's most likely answer could exceed this (the mode beats a second sample), so a bar relative to self-agreement isn't lenient.
- **Proposed logs-mode bar:** val agreement within 2 points of the logged model's self-agreement (the same 2-point margin used against baselines), measured by one re-ask of val when the logged model is callable; otherwise a fixed `min_value`.
- **M6 re-judged:** bar 91.75; best attempt val 91.50 (attempt 2), so **a miss by 0.25 points** (one val message). Within the noise of the 400-message self-agreement estimate (±~2.4), but a formal miss. Test agreement 91.8.

### M6 closed as a recorded near miss; self-agreement bar built (Oct 18, 2026)

- **Decision (user):** accept M6 as is, 0.25 points short on val against the self-agreement bar. No rerun.
- **Built:** `pass_bar: {compare_to: logged_self_agreement, max_gap_points: N}` (logs mode only).
  - The load stage records the logged model, system prompt and reasoning effort (`data/logged_prompt.json`) when every log shares them.
  - The baseline stage re-asks that model the val inputs once, cached, with an estimate and the spend guard, in `baselines/logged-retest/`, and scores how often it answers the same.
  - The pass check uses that number minus the gap. Mixed prompts or models, or a model missing from prices.yaml, are a config error that points at `min_value`.
  - Two tests.
- `examples/banking77-logs/task.yaml` now uses it (spec 15.3 had `min_value: 95`). The M6 run was left as it ran.

## Final proof runs (Oct 18, 2026)

| Proof | Run | Outcome | Interventions | Spend |
|---|---|---|---|---|
| BANKING77, raw folder → agent | `proof-banking77` | **pass** (val 91.09 vs bar 82.58). Test acc 92.37 vs Luna 1-shot 86.01, **beat by +6.4** [+5.2, +7.5], and Luna zero-shot 82.01, beat by +10.4. $0.025 vs $0.046 / $0.108 per 1k; p50 0.24 s vs 1.2 s | 0 | $5.12 |
| CORD raw → agent | `20261018-0009-cord-raw` | **declined correctly:** "good fit, but no available model on Together can serve it"; suggests another provider | 0 | $0.001 |
| Simulated logs | `m6-banking77-logs` | **near miss** (user-accepted): val agreement 91.50 vs the self-agreement bar 91.75. Test agreement 91.8; truth acc 80.2 vs logged Luna 81.7; $0.036 vs $0.075 per 1k | 0 | $13.22 (+ logs $0.37, retest $0.03) |
| Email (Enron) | `proof-enron` | **pass on val** (macro-F1 43.23 vs bar 40.80, the 3-shot baseline's val; the task's gap is 0). Retrained on train + val (the first live retrain): test macro-F1 **48.67** (attempt 1 alone: 42.82, so +5.9 from the newer data). **Behind** Luna 3-shot 52.86 by 4.2, not statistically significant [−8.1, +1.7]; beat Luna zero-shot 31.06 by +17.6. $0.092 vs $0.337 per 1k; p50 0.70 s vs 2.2 s. Thin / late category warnings shown before results | 0 | $9.32 |

- **Pauses:** the logs and Enron runs each stopped once at a Together training step for an empty balance (402), and were resumed unchanged after the user added credits. Nothing in either run changed, so the reports count 0 interventions. Whether a provider-balance pause counts as a "manual retry" under spec 2 is a judgment call; recorded here.
- **Totals:** all runs **$74.08**, under the $85 hard stop and the $100 budget. No endpoints live on the Together account.
- **The Enron retrain mattered:** adding the newest month (val) lifted test macro-F1 by 5.9 points, which is what M4 design note 1 predicted for time splits with late-arriving folders.

## Fireworks as a second training provider (Nov 16, 2026)

`train.provider: fireworks` trains and serves on Fireworks through the same Trainer / Server / TunedModel interfaces as Together (`ft/providers/fireworks_provider.py`). Checked on Nov 16 against docs.fireworks.ai and the source of `fireworks-ai` 1.2.33 and `fireworks-training-cookbook` 2026.11.13.1; **not yet run against a live account** (no Fireworks key yet).

- **Training:** Fireworks' managed fine-tuning jobs no longer take new jobs. Training goes through their Training API, whose loop runs in the caller's process (the cookbook's `sft_loop` recipe), so there is no server-side job to poll.
  - `start` launches the loop as a detached worker (`ft/providers/fireworks_train.py`) and `poll` reads the status file it writes. The train stage and `ft resume` are unchanged.
  - There is no upload: the recipe reads the same chat JSONL from disk.
- **Serving:** an on-demand deployment over REST, one replica, with the fine-tuned model as `baseModel`. Deployments are named `ft-<run>-a<n>` like Together endpoints, so `ft status` and `ft teardown --all` find them.
- **Inference:** the same OpenAI-SDK code as Together at Fireworks' base URL; thinking is switched off with `reasoning_effort: none`.
- **Model:** `accounts/fireworks/models/qwen3p8-27b` (Qwen 3.8 27B), the smallest Qwen their serverless training lists. Fireworks has no 4B Qwen, so runs there use a larger model than the Together ones.
- **Guards until a first live run:** `models.yaml` marks the model `deployable: false` and `prices.yaml` marks Fireworks `verified: false` ($4.103 per 1M train tokens, $8.00/h per H100, from the pricing page). Real runs refuse to start until both are confirmed; dry runs work.
- **Confirmed in the package source:**
  - `sft_loop.Config` takes the fields the worker passes (`serverless`, `base_model`, `dataset`, `lora_rank`, `max_seq_len`, `epochs`, `learning_rate`, `log_path`, `output_model_id`); serverless needs `lora_rank > 0` and a concrete `max_seq_len`.
  - The recipe resolves the tokenizer itself (`Qwen/Qwen3.8-27B` for this model) and infers the renderer, so neither is set in `models.yaml`.
  - After the last step it promotes the final checkpoint under `output_model_id`.
  - The SDK's own deployment client uses the same calls: `POST /v1/accounts/<account>/deployments?deploymentId=`, camelCase body, `state == READY`, `DELETE ...?ignoreChecks=true`, account id from `GET /v1/accounts`. A 409 on create means the deployment exists, so `create` reuses it.
- **From the docs only, to confirm on the first live run:** that `qwen3p8-27b` is in the serverless training pool for the account, that `deploymentShape: default` serves a live-merged fine-tune, the `<model>#<deployment>` inference string, and the prices.
