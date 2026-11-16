# Auto Fine-Tuning Pipeline: MVP Spec

Written Oct 16, 2026, before any code, as the brief for the AI coding agent that built it (hence "ask the user"). Companion to the [idea brief](IDEA_BRIEF.md). The build then diverged in places (most visibly, OpenAI replaced Claude as the frontier provider); [NOTES.md](NOTES.md) records every difference and wins where the two disagree.

---

## 0. Build rules

- **Build milestone by milestone** (section 16). After each one, stop, summarize what works, show the report or test output, and wait for the go-ahead before starting the next.
- **Never spend money without asking.** Every paid action (Claude calls, Together training, Together endpoints) must go through the spend guard (section 11). Get the whole pipeline running in `--dry-run` mode with fake providers before any real API call.
- **Verify provider APIs before coding against them.** The API details below were checked against the docs on Oct 16, 2026, but these SDKs change often. Before writing each provider, open the linked docs (section 19) and confirm parameter names, field names and flows. If something here disagrees with the docs, follow the docs and note the difference in `NOTES.md`.
- **Endpoints cost money while idle.** Any code that creates a Together endpoint must guarantee teardown (section 9.8). Treat a leaked endpoint as a serious bug.
- **Ask the user** for API keys, OAuth credentials and any decision marked "ask" in this spec. Don't invent values.

---

## 1. What we're building

A tool that takes a folder of raw data and returns a small, fine-tuned open model that does one narrow task, deployed behind an OpenAI-compatible endpoint, plus a one-page report comparing it to frontier models (Claude) on accuracy, cost and speed.

**The product is the automation.** Every decision a machine-learning engineer would normally make is made by an agent: what the task is, where the right answers come from, which base model to use, whether the result is good enough, and how to fix it if not. Plain code executes those decisions. The agent never runs training code itself; it writes a config (`task.yaml`) and picks fixes from a fixed menu.

**Design principles**

1. **Agent decides, code executes.** The agent's only outputs are a validated `task.yaml`, a fit decision, and retry-loop fix choices. Everything else is deterministic code that reads `task.yaml`.
2. **Nothing in `ft/` knows about any specific dataset.** BANKING77, CORD and the user's inbox are test inputs in `examples/`, never special cases in code.
3. **Providers are swappable.** Stages call interfaces, not SDKs. Together is the v1 trainer; Fireworks or Tinker must be addable later without touching stages.
4. **Every run is inspectable and resumable.** One folder per run holds the config, data splits, predictions, metrics, spend, external IDs and an event log with the agent's reasoning.
5. **Honest reports.** Compare against the same test examples, never let the retry loop see the test set, and report cost break-even even when it's unflattering.

---

## 2. MVP definition of done

The MVP is done when one pipeline takes all four proof-set datasets from raw data to a report, with **zero human interventions** after the one approval checkpoint.

| Dataset | Task | Label source | What it proves |
| --- | --- | --- | --- |
| BANKING77 | Customer message → one of 77 intents | Dataset labels | Text classification end to end |
| CORD v2 | Receipt photo → structured fields | Dataset labels | Images and extraction, same pipeline |
| Simulated logs | BANKING77 messages run through Claude, saved as logs | Logs (Claude's answers) | Logs mode |
| User's Gmail | Email → important / read / skip | Recorded behavior | Behavior labels; tuned model beats a frontier model |

**Success metric:** human interventions per run, target zero. Every intervention (manual edit of `task.yaml` after approval, manual retry, code fix mid-run) is logged as a `human_intervention` event and counted in the report.

---

## 3. Not in the MVP

- A web UI. The MVP is a CLI.
- Multi-tenant hosting, auth, billing, customer accounts.
- Real-time logs capture through a live proxy (the log format is defined so a proxy can plug in later; MVP uses simulated logs).
- Database connectors (dlt) beyond files, Hugging Face datasets and Gmail.
- Reinforcement fine-tuning, DPO, full fine-tuning. LoRA SFT only.
- Fireworks and Tinker providers (interfaces must allow them; implementations come later).
- Local inference of downloaded weights (stretch goal for the email cleaner, see 14.7).

---

## 4. Architecture

```
            ┌──────────────┐
 raw data → │  load+profile │ → profile.json
            └──────┬───────┘
                   ▼
            ┌──────────────┐   (M3+) agent reads profile, decides fit,
            │    AGENT      │   writes task.yaml  ──► human approval
            └──────┬───────┘
                   ▼
   split → label → prepare → baseline (Claude) → train (Together) → serve → predict → evaluate
                                                                                   │
                                            ┌──────────── pass on val? ────────────┤
                                            │ no (M4+)                         yes │
                                            ▼                                      ▼
                                  AGENT picks a fix ──► new attempt          report.html
                                  (config patch)
```

**Stages** (section 9) run in order, each reading `task.yaml` and the run folder, and writing its outputs to the run folder. A stage is skipped if its outputs already exist and its inputs haven't changed (resumability).

**Attempts.** Stages from `prepare` through `evaluate` belong to an *attempt*. Attempt 1 uses `task.yaml` as written. The retry loop (M4) creates attempt 2, 3... by applying config patches. Stages before `prepare` run once per run.

---

## 5. Repo layout

```
ft/
  __init__.py
  cli.py                  # typer app: ft run / plan / status / resume / teardown / report / simulate-logs / email
  config.py               # pydantic models for task.yaml; load + validate
  context.py              # RunContext: paths, state, event log, providers, spend guard
  state.py                # state.json read/write; stage status; external IDs
  events.py               # append-only events.jsonl
  spend.py                # price table, estimates, guard, actuals
  stages/
    load.py               # read CSV / JSONL / image folder / HF dataset / gmail export / logs → records.jsonl
    profile.py            # profile.json for the agent and the report
    split.py              # train / val / test
    label.py              # attach targets: dataset | behavior | frontier | logs
    prepare.py            # training JSONL in the trainer's chat format (text or vision)
    baseline.py           # frontier predictions on val + test
    train.py              # upload, launch LoRA job, poll
    serve.py              # deploy endpoint, wait ready, guaranteed teardown
    predict.py            # tuned predictions on val + test
    evaluate.py           # metrics for any predictions file
    report.py             # report.html + report.json
  tasks/
    base.py               # TaskType interface: prompts, output schema, parse, normalize, score
    classify.py
    extract.py
  providers/
    base.py               # FrontierProvider, Trainer, Server, TunedModel interfaces
    anthropic_provider.py
    together_provider.py
    fake.py               # fakes for --dry-run and tests
  agent/
    agent.py              # tool-use loop (M3) and retry-loop fixer (M4)
    tools.py
    prompts.py
  connectors/
    gmail.py              # (M5)
    logs.py               # (M6) log format normalization
  redact.py               # (M5) Presidio-based redaction
  templates/
    report.html.j2
models.yaml               # registry of base models: id, kind (text/vision), size, deployable (verified), notes
prices.yaml               # price table (section 11)
examples/
  banking77/task.yaml
  banking77/fetch.sh      # downloads the CSVs into data/banking77/
  cord/task.yaml
  banking77-logs/task.yaml
  email/task.yaml
  email/about_me.md       # user writes this (M5)
  fixtures/               # tiny datasets for tests and agent decline cases
tests/
data/                     # gitignored
runs/                     # gitignored
NOTES.md                  # discrepancies found vs this spec, decisions made while building
pyproject.toml
.env.example
.gitignore
```

---

## 6. CLI

```
ft run <dir> [--yes] [--dry-run] [--run-id ID] [--from STAGE] [--max-spend USD]
    <dir> contains task.yaml → run it.
    <dir> has no task.yaml → (M3+) the agent profiles the data and drafts one, then asks for approval.
ft plan <dir>                  # (M3) agent writes task.yaml only, no paid stages except agent calls
ft status <run_id>             # stages, attempts, spend so far, live endpoints
ft resume <run_id>             # continue from the first incomplete stage
ft teardown <run_id> | --all   # delete endpoints created by ft (name prefix "ft-")
ft report <run_id>             # regenerate report from saved results
ft simulate-logs <task_dir> --model MODEL --n N     # (M6)
ft email pull | ft email apply --run <run_id>       # (M5)
```

`--yes` skips confirmation prompts (approval checkpoint and spend confirmations) but never skips the spend cap.

---

## 7. task.yaml

Validated by pydantic (`ft/config.py`). Unknown keys are errors. `source` is a discriminated union on `type`, so each source type has its own fields. All metrics are expressed in points (0–100). The agent writes this file in M3; before that, humans write it.

### 7.1 Full annotated example (BANKING77)

```yaml
name: banking77
description: Classify banking customer support messages by intent.

source:
  type: csv                   # csv | jsonl | image_folder | hf_dataset | gmail | logs
  path: data/banking77        # relative to repo root
  files:                      # optional; if a predefined split exists, name it
    train: train.csv
    test: test.csv

task:
  type: classify              # classify | extract
  input:
    kind: text                # text | image
    field: text               # column holding the input (or image path column)
  output:
    field: category           # column holding the target (classify) or the parsed JSON (extract)
    labels: auto              # classify only: "auto" = distinct values in train, or an explicit list
  instruction: >-             # one or two sentences; shared by the tuned model and baselines
    Classify the banking customer message into its intent label.

targets:                      # where the right answers come from (not to be confused with task.output.labels)
  from: dataset               # dataset | behavior | frontier | logs
  # frontier_model: claude-sonnet-5-5   # required when from: frontier; also enables the add_examples fix (M4)

splits:
  use_predefined: true        # use source.files.test as the test set if present
  val_fraction: 0.1           # carved from train, stratified for classify
  val_max: 1000
  test_max: 3080              # cap; sample stratified with seed if larger
  by_time: null               # e.g. "date" → oldest→train, then val, newest→test (used for email)
  seed: 42

baselines:                    # may be empty in logs mode (section 15)
  - model: claude-haiku-5-5
    effort: low
    thinking: disabled        # see 10.1 for per-model allowed values
    # context_file: path      # optional extra context added to this baseline's system prompt
  - model: claude-sonnet-5-5
    effort: low
    thinking: between_tools

train:
  provider: together
  base_model: Qwen/Qwen3.5-4B
  fallback_models: [Qwen/Qwen3.5-9B]   # used by the retry loop's bigger_model fix, in order
  lora: true
  epochs: 3
  learning_rate: 1.0e-4       # LoRA commonly uses 1e-4 to 2e-4; check `tg fine-tuning model-limits <model>` for bounds
  max_examples: null          # cap training rows (null = all)

eval:
  primary_metric: accuracy    # classify: accuracy | macro_f1 | agreement (logs) ; extract: field_accuracy
  pass_bar:                   # one of two shapes:
    compare_to: best_baseline #   relative: best_baseline | a model id from baselines, plus max_gap_points
    max_gap_points: 2.0       #   tuned may be at most 2 points below that baseline on val
    # min_value: 95.0         #   absolute: primary metric (in points, 0-100) must reach this on val
  concurrency: 16

budget:
  max_spend_usd: 30
  max_attempts: 3             # retry loop (M4)
```

### 7.2 CORD (images, extraction)

```yaml
name: cord
description: Extract totals and line items from receipt photos.
source:
  type: hf_dataset
  dataset: naver-clova-ix/cord-v2      # verify fields on first load; expected: image, ground_truth (JSON string)
  splits: {train: train, val: validation, test: test}
  target_map:                          # how load builds `target` from a JSON column (see 9.1)
    column: ground_truth
    root: gt_parse
    fields:
      total_price: total.total_price
      subtotal_price: sub_total.subtotal_price
      tax_price: sub_total.tax_price
      menu: {path: menu, as_list: true, item: {name: nm, count: cnt, price: price}}
task:
  type: extract
  input: {kind: image, field: image}
  output:
    field: target                       # built by load.py using source.target_map
    schema:
      total_price: money
      subtotal_price: money
      tax_price: money
      menu:                             # list of items
        list_of:
          name: text
          count: text
          price: money
  instruction: Extract the receipt's totals and line items as JSON.
targets: {from: dataset}
splits: {use_predefined: true, seed: 42}
baselines:
  - {model: claude-haiku-5-5, effort: low, thinking: disabled}
  - {model: claude-sonnet-5-5, effort: low, thinking: between_tools}
train:
  provider: together
  base_model: Qwen/Qwen3-VL-8B-Instruct   # verify it is vision-tunable AND deployable (section 18)
  fallback_models: []
  lora: true
  epochs: 3
  learning_rate: 1.0e-4
eval:
  primary_metric: field_accuracy
  pass_bar: {compare_to: best_baseline, max_gap_points: 3.0}
  concurrency: 8
budget: {max_spend_usd: 40, max_attempts: 3}
```

Field types for extraction: `text`, `money`, `number`, `date`, `list_of`. Types decide normalization in scoring (section 9.10). Keep schemas to at most 15 leaf fields and make every field required (missing values become empty strings, not null), to stay inside Claude's structured-output limits on optional and union-typed fields.

### 7.3 Email and logs examples

Defined in sections 14.3 and 15.3.

---

## 8. Run folder and state

```
runs/<run_id>/                 # run_id = <yyyymmdd-hhmm>-<task name>, e.g. 20261017-1430-banking77
  task.yaml                    # frozen copy used by this run
  state.json                   # stage status, attempt list, external IDs, spend actuals
  events.jsonl                 # every stage start/end, every agent decision + reasoning, every paid call summary
  data/
    records.jsonl              # normalized records: {id, input, target, meta}
    profile.json
    train.jsonl  val.jsonl  test.jsonl
  baselines/<model>/
    val.predictions.jsonl  test.predictions.jsonl  usage.json  metrics.val.json  metrics.test.json
  attempts/<n>/
    task.patch.yaml            # config changes vs task.yaml (empty for attempt 1)
    train.together.jsonl       # exact file uploaded
    training.json              # file id, job id, status history, loss curve, model_object_id
    endpoint.json              # endpoint id/name, deployment id, timestamps, minutes billed
    val.predictions.jsonl  test.predictions.jsonl  metrics.val.json  metrics.test.json
    diagnosis.json             # (M4) what the fixer saw
  report.html
  report.json
```

**Prediction file rows:** `{id, prediction_raw, prediction_parsed, valid, latency_ms, usage: {input_tokens, cache_read_tokens, cache_write_tokens, output_tokens}}`.

**state.json** must always contain every external resource ID the run created (file IDs, job IDs, endpoint IDs, deployment IDs) the moment it's created, so `ft teardown` and `ft resume` work after a crash.

**Resumability:** each stage writes its outputs to a temp name and renames on success, then marks itself done in `state.json` with a hash of its inputs. `ft resume` re-runs the first stage that is not done or whose input hash changed. Never re-launch a training job if `state.json` already has a job ID for this attempt; poll the existing one instead (Together warns that each create call starts a new billed job).

---

## 9. Stages

Stage signature: `def run(ctx: RunContext) -> StageResult`. Stages must not import provider SDKs directly.

### 9.1 load

Reads `source` and writes `records.jsonl` with one normalized record per example: `{id, input, target, meta}`.

- `csv` / `jsonl`: input from `task.input.field`, target from `task.output.field`. Keep other columns in `meta`. Record which file each row came from (`meta.split_hint`) when `source.files` names splits.
- `image_folder`: images plus a labels file (CSV/JSONL keyed by filename). Store image paths, not bytes.
- `hf_dataset`: uses `datasets.load_dataset`. Save images to `runs/<id>/data/images/` as JPEG and store paths.
- `target_map` (any source type): builds `target` from a JSON column. Parse `column`, descend into `root`, then fill each output field from its dotted path. `as_list: true` turns a single dict into a one-item list; `item` renames subfields. Missing keys become "". For CORD (section 7.2) this maps `gt_parse.total.total_price → total_price`, `gt_parse.menu[*].nm → menu[*].name`, and so on. This is generic code in `ft/`; the CORD-specific mapping lives only in `examples/cord/task.yaml`.
- `gmail` (M5) and `logs` (M6): see sections 14 and 15.

Drop exact-duplicate inputs (keep first); log how many were dropped.

### 9.2 profile

Writes `profile.json`: row count; per field dtype, null rate, length stats, cardinality, top 30 values with counts; for images, size stats and 6 sample paths; split hints; 20 random sample records (inputs truncated to 500 chars). Used by the agent (M3) and the report.

### 9.3 split

- If `use_predefined` and the source names a test split, use it; same for val. Otherwise carve.
- Val: `val_fraction` of train, capped at `val_max`, stratified by target for `classify`.
- Test: capped at `test_max`, stratified sample with `seed`.
- `by_time`: sort by that field; oldest → train, next → val, newest → test. Mandatory for email.
- Write `train.jsonl`, `val.jsonl`, `test.jsonl` and log sizes and label distributions.
- Leakage check: assert no input appears in more than one split (after whitespace/case normalization). Fail loudly if it does.

### 9.4 label

Attaches targets where they aren't already present.

- `dataset`: targets already in records; validate non-empty.
- `behavior`: computed by the connector's rules (section 14.4).
- `frontier`: label train records with `targets.frontier_model` using the same prompt as the baseline, via the Message Batches API (50% cheaper; most batches finish within an hour). Val and test must have independent ground truth; if they don't, the run must say clearly in the report that it measures agreement with the frontier model, not accuracy.
- `logs`: targets are the logged responses (section 15).

### 9.5 prepare

Writes the training file in the trainer's format. For Together text SFT (conversational format):

```json
{"messages": [
  {"role": "system", "content": "<task.instruction>"},
  {"role": "user", "content": "<input text>"},
  {"role": "assistant", "content": "<label, or compact JSON for extract>"}
]}
```

- The tuned model's system prompt is **only** `task.instruction`. It does not include the label list or the JSON schema; the model learns them from training. This keeps per-request tokens low and is part of the cost story in the report.
- Extraction targets are serialized as compact JSON with keys in schema order.
- Vision: follow Together's vision fine-tuning dataset format exactly (verify in docs; expected: user content as a list of `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` plus a text part). Downscale images so the longest side is at most 1,280 px before encoding; make this a setting.
- Respect `train.max_examples` (stratified sample for classify).
- Count approximate tokens (characters / 4 for text; a configurable per-image token estimate) for the spend estimate.

### 9.6 baseline

Runs every model in `baselines` on **val and test**. Uses the `FrontierProvider` (section 10.1).

- Prompt: system = `task.instruction` + task-specific detail (classify: "Answer with exactly one label from this list:" + one label per line; extract: the schema described in words). Put the stable part of the system prompt in a block with `cache_control` so it's cached (minimum cacheable length on the 5.5 models is 512 tokens; shorter prompts just aren't cached, no error).
- Output: use structured outputs (`output_config.format` with a JSON schema). Classify: `{"label": {"type": "string", "enum": [labels]}}`. Extract: schema built from `task.output.schema`. Compare enum values case-insensitively (Claude may change capitalization of enum values).
- Settings per model come from `baselines[*]`: `output_config.effort` and `thinking` (section 10.1 lists what each model accepts).
- `temperature` unset; `max_tokens` sized to the output (classify: 64; extract: 1,024) plus room for thinking if thinking isn't disabled.
- Record latency and usage for every call.
- Async with a semaphore (`eval.concurrency`), retries with exponential backoff on 429 and 5xx, honoring `retry-after`.

### 9.7 train

Uses the `Trainer` (section 10.2).

1. Preflight: confirm `base_model` is in `models.yaml` with `deployable: true` (verified via `tg beta models configs <model>`; section 18). Fail before uploading anything if not.
2. Upload the training file; poll the file's `processing_status` until `COMPLETED`; fail with the validation report on `INVALID_FORMAT` or `FAILED`.
3. Create the LoRA job with `epochs`, `learning_rate`, `lora=True`, `train_on_inputs="auto"`, `n_checkpoints=1`, `suffix=<run_id>-a<n>` (≤ 64 chars). Save the job ID to `state.json` immediately.
4. Poll every 60 s (`pending → queued → running → uploading → completed`; terminal: `completed`, `error`, `cancelled`). Queue waits can be up to an hour. Log status changes and fetch job events on error.
5. On completion, save the model registry ID (`model_object_id`, exposed by the Python SDK as `api_model_object_id` per the docs; verify) and the training loss curve if the monitoring API provides it.

### 9.8 serve

Uses the `Server` (section 10.2). Exposed as a context manager:

```python
with ctx.server.deploy(model_ref, name=f"ft-{run_id}-a{n}") as endpoint:
    predict(endpoint, val); predict(endpoint, test)
# teardown always runs here
```

- Deploy, then poll until `DEPLOYMENT_STATE_READY` (provisioning can take up to 10 minutes). Save endpoint and deployment IDs to `state.json` the moment they exist.
- Teardown in `finally`, plus an `atexit` hook and SIGINT/SIGTERM handlers. Teardown must be idempotent.
- Record start time, ready time and teardown time; minutes billed = teardown − deploy request.
- `ft teardown --all` finds every endpoint whose name starts with `ft-` and deletes it after confirmation.
- Endpoint names must be unique per attempt and valid for Together (lowercase, digits, hyphens; check limits).

### 9.9 predict

Runs the tuned endpoint on **val and test** while it's up. Same prompt as training (system = instruction, user = input). `max_tokens`: classify 16, extract 1,024. Deterministic decoding (`temperature=0`).

Call the endpoint through the **OpenAI Python SDK** pointed at the Together inference base URL. This proves the "change the base URL and model name" claim; fall back to the Together SDK only if needed.

Parse with the task type's parser; anything unparseable or outside the label set is `valid: false` and counts as wrong.

### 9.10 evaluate

One scorer for every predictions file (baselines and tuned), so comparisons are fair.

**Classify**
- Normalize: strip, collapse whitespace, case-insensitive match against labels.
- Metrics: accuracy, macro-F1, invalid rate, per-class precision/recall/F1, top 10 confusion pairs with 3 example inputs each.

**Extract**
- `json_valid_rate`: output parses and has every schema key.
- Per-field normalized exact match: `text` → casefold, collapse whitespace; `money`/`number` → keep digits only (CORD mixes "60.000" and "60,000" formats); `date` → ISO if parseable.
- `list_of` fields: item-level F1, where an item matches if all its normalized subfields match.
- `field_accuracy` (primary) = mean over top-level fields, where list fields contribute their F1.

**All tasks**
- Latency p50/p95, throughput at the configured concurrency.
- Cost: actual spend for this predictions file, and cost per 1,000 requests (section 11.3).
- Bootstrap 95% confidence interval for the primary metric (1,000 resamples), so the report can say whether a gap is real.

### 9.11 report

`report.html` (single file, Jinja2, inline CSS, no external assets, readable on a phone) and `report.json` with the same numbers.

Sections, in order:
1. **Verdict:** pass/fail against the bar, in one sentence with numbers: "Tuned Qwen3.5-4B scored 92.1% vs Claude Sonnet 5.5's 93.4% on 3,080 test messages." Wording against the compared baseline, decided by a paired bootstrap of the test-set difference (tuned − baseline, 1,000 resamples, same examples):
   - **"beat"** only when the tuned score is higher **and** the 95% CI of the difference is entirely above 0.
   - **"fell short by X points"** when the CI is entirely below 0.
   - **"ahead by X points"** / **"behind by X points"**, each followed by "not statistically significant", when the CI includes 0 (the sign of the point estimate picks the word).
   Never say "beat" or "fell short" when the CI includes 0.
2. **Comparison table:** each model × primary metric (with CI), secondary metrics, invalid rate, p50/p95 latency, cost per 1,000 requests, break-even volume.
3. **Attempts** (M4): config changes, val score, fixer's reasoning.
4. **Where it goes wrong:** per-class or per-field table, top confusion pairs or field errors with examples.
5. **How the task was set up:** task, label source, split sizes and method, base model, agent rationale (M3).
6. **Spend:** breakdown by stage and provider, against the budget.
7. **Human interventions:** count and list.
8. **Run metadata:** run ID, git commit, timestamps, external IDs.

All numbers are test-set numbers except the attempts table, which shows val.

---

## 10. Providers

```python
class FrontierProvider(Protocol):
    async def predict(self, items: list[Item], prompt: PromptSpec, settings: BaselineSettings) -> list[Prediction]: ...
    async def batch_predict(self, items, prompt, settings) -> list[Prediction]: ...   # Message Batches API

class Trainer(Protocol):
    def preflight(self, base_model: str) -> None: ...
    def upload(self, path: Path) -> str: ...                          # returns file id when processed
    def start(self, file_id: str, params: TrainParams) -> str: ...    # returns job id
    def poll(self, job_id: str) -> JobStatus: ...
    def result(self, job_id: str) -> ModelRef: ...

class Server(Protocol):
    def deploy(self, model: ModelRef, name: str) -> ContextManager[Endpoint]: ...
    def teardown(self, endpoint_id: str) -> None: ...
    def list_ft_endpoints(self) -> list[Endpoint]: ...

class TunedModel(Protocol):
    async def predict(self, endpoint: Endpoint, items, prompt) -> list[Prediction]: ...
```

### 10.1 Anthropic (baselines, frontier labeling, agent)

- Models (Oct 2026): `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`. The agent uses `claude-opus-5-5`.
- Effort: `output_config={"effort": "low" | "medium" | "high" | ...}`.
- Thinking settings that are accepted:
  - Haiku 5.5: `thinking={"type": "disabled"}` allowed at `high` effort or below.
  - Sonnet 5.5: no `"disabled"`; the lowest setting is `thinking={"type": "between_tools"}`, allowed at `low`/`medium`/`high`.
  - Opus 5.5: thinking is always on; `"disabled"` returns a 400.
- Structured outputs: `output_config={"format": {"type": "json_schema", "schema": {...}}, "effort": "low"}` (combine both keys in one `output_config`), or `client.messages.parse(..., output_format=PydanticModel)`. Enums support strings; no numeric/length constraints; `additionalProperties` must be `false`. A `refusal` or `max_tokens` stop reason can produce non-matching output → mark invalid.
- Prompt caching: `cache_control={"type": "ephemeral"}` on the stable system block. Check `usage.cache_read_input_tokens` to confirm hits.
- Message Batches: `client.messages.batches.create(requests=[{"custom_id", "params"}])`, poll with `.retrieve(id)`, read with `.results(id)`. 50% off; limit 100,000 requests or 256 MB per batch. `custom_id` must match `^[a-zA-Z0-9_-]{1,64}$`.
- Images: send as base64 image content blocks.
- Keys from `ANTHROPIC_API_KEY`.

### 10.2 Together (training, serving)

- Install: `together[cli]` (CLI ≥ 2.24.0 needed for the `tg beta endpoints` commands).
- Upload: `client.files.upload(file=..., purpose="fine-tune", check=True)`, then poll `client.files.retrieve(id).processing_status`.
- Train: `client.fine_tuning.create(training_file=..., model=..., n_epochs=..., learning_rate=..., lora=True, train_on_inputs="auto", n_checkpoints=1, suffix=...)`. Poll `client.fine_tuning.retrieve(id=...)`.
- Serve: fine-tuned models are served **only on dedicated endpoints**, billed per minute per running replica even when idle. The CLI does it in one step: `tg beta endpoints deploy <MODEL_OBJECT_ID> --endpoint <name>`; poll with `tg beta endpoints get <ENDPOINT_ID>`; tear down with `tg beta endpoints rm <ENDPOINT_ID> --force`. The SDK needs several calls (create endpoint, create deployment with a config ID, set traffic split) and a different teardown sequence (scale to zero, wait for stopped, drop traffic, delete deployment, delete endpoint). **Recommendation:** wrap the CLI with `subprocess` for deploy/get/rm (check for a JSON output flag; otherwise parse carefully) and keep the SDK for files and training. If a model has several deployment profiles, pass `--config <cr_...>` from `tg beta models configs <model>`.
- Inference: base URL `https://api-inference.together.ai/v1`, `model` = the endpoint string (`<project-slug>/<endpoint-name>`) printed by deploy, **not** the model object ID.
- Preflight: `tg beta models configs <BASE_MODEL>` must list at least one config, or the fine-tune can't be hosted.
- Keys from `TOGETHER_API_KEY`.

### 10.3 Fakes (`providers/fake.py`)

Used by `--dry-run` and tests. Deterministic given a seed.

- `FakeFrontier`: returns the correct target with probability `p` (default 0.9), otherwise a random label (classify) or a corrupted field (extract). Reports plausible token usage.
- `FakeTrainer`: completes after two polls; returns a fake `ModelRef`.
- `FakeServer`: context manager that records deploy/teardown calls (tests assert teardown always ran, including when predict raises).
- `FakeTuned`: correct with probability `q` (default 0.88; settable per attempt so retry-loop tests can show improvement).

`ft run examples/banking77 --dry-run` must produce a complete report with zero network calls except reading local data.

---

## 11. Cost and spend control

### 11.1 prices.yaml

```yaml
last_checked: 2026-10-16      # update whenever prices are re-verified
anthropic:                    # USD per million tokens; https://platform.claude.com/docs/en/about-claude/pricing
  claude-haiku-5-5:  {input: 0.10, cache_write_5m: 0.125, cache_read: 0.01, output: 0.50}   # prompts ≤ 100k tokens
  claude-sonnet-5-5: {input: 2.00, cache_write_5m: 2.50,  cache_read: 0.10, output: 10.00}
  claude-opus-5-5:   {input: 4.00, cache_write_5m: 5.00,  cache_read: 0.20, output: 20.00}
  batch_discount: 0.5
together:                     # verify at together.ai/pricing before the first real run (ask the user to confirm)
  training_per_mtok:
    lora_up_to_16b: 0.48      # from the idea brief; re-verify
  endpoint_per_hour: {}       # fill per hardware config shown by `tg beta models configs`; ask the user if unclear
  vision_tokens_per_image_estimate: 1000   # replace with measured value after the first CORD run
```

The spend guard refuses to run a paid stage whose price entry is missing.

### 11.2 Guard

- Before each paid stage: estimate cost, add to actual spend so far, compare with `budget.max_spend_usd` (or `--max-spend`). Over budget → stop with a clear message. Otherwise print the estimate and ask to confirm (skipped with `--yes`).
- After each paid call: add actual cost (from usage, job billing info if available, or endpoint minutes × rate) to `state.json`.
- Endpoint estimate = (10 min provisioning + predicted eval minutes) × hourly rate.

### 11.3 Cost per 1,000 requests and break-even

Fine-tuned models run on a dedicated GPU, so the comparison isn't per-token vs per-token. Report both of these, honestly:

- **Frontier cost per 1,000 requests** = mean actual cost per request on the test set × 1,000 (with caching as actually observed).
- **Tuned cost per 1,000 requests at full utilization** = hourly endpoint rate ÷ (measured throughput in requests/hour at `eval.concurrency`) × 1,000.
- **Break-even volume** = (hourly endpoint rate × 24) ÷ frontier cost per request = requests per day at which a 24/7 endpoint costs the same as calling the frontier model.

Note for the report and the user: Haiku 5.5 is very cheap ($0.10 / $0.50 per million input/output tokens), so for short classification tasks the break-even volume may be high. The report must show this rather than hide it; the stronger claims may be speed, data control and the "better" win (section 14).

---

## 12. The agent (M3)

**Job:** look at a data folder, decide whether fine-tuning fits, and write a `task.yaml` that runs without edits.

- Model: `claude-opus-5-5`, default effort, tool-use loop, max 30 turns, its own spend line in the budget.
- Input: `profile.json` plus the tool results below. It never sees the test split's targets beyond what the profile shows (the split happens after approval).
- Tools:
  - `get_profile()`
  - `sample_rows(file, n, where=None)` (inputs truncated to 1,000 chars)
  - `value_counts(file, column, top=50)`
  - `view_images(paths)` (max 6 per call, sent as image blocks)
  - `list_base_models(kind)` (from `models.yaml`, deployable only)
  - `estimate_cost(task_yaml)` (spend estimate for a full run)
  - `validate_task(task_yaml)` (pydantic errors as text)
  - `submit_task(task_yaml, rationale)` (must validate; ends the loop)
  - `decline(reason, suggestion)` (not a fit; ends the loop)
- System prompt must encode the fit rules from the brief:
  - Good fit: classify, route, extract, structured output, fixed format, fixed tool set; text or images; outcomes already recorded; outputs gradable automatically; enough examples (default minimum 500 labeled rows, or a frontier-labeling plan that gets there).
  - Poor fit: facts that change (suggest retrieval), open-ended reasoning, low volume, no way to judge a good output.
  - Prefer recorded outcomes or dataset labels over frontier labeling. Pick the smallest deployable base model that fits the input kind. Choose `accuracy` for balanced classes, `macro_f1` when the most common class is over 40%. Use `by_time` splits when there's a timestamp and behavior labels.
- Every decision is logged to `events.jsonl` with a one-line rationale; the report shows them.
- **Human checkpoint** (skipped with `--yes`): print task summary (input → output, label source, split sizes, base model, estimated cost), plus 10 random test examples with their expected answers. Prompt `[a]pprove / [e]dit / [q]uit`. Edit opens `$EDITOR`, then revalidates; an edit at this point is not an intervention, but any later manual change is.
- Fixtures to test declines: `examples/fixtures/too_small/` (60 rows), `examples/fixtures/open_ended/` (long free-form answers with no consistent target).

**Done when:** `ft run data/banking77` and `ft run <cord raw folder>` (no task.yaml) produce passing runs with zero edits after approval, and both decline fixtures are declined with sensible reasons.

---

## 13. Retry loop (M4)

- After each attempt, check the tuned model's **val** primary metric against the pass bar: within `max_gap_points` of the chosen baseline's **val** score, or at least `min_value` for the absolute form.
- On fail, if attempts remain and the budget allows, the fixer (same agent, separate prompt) receives `diagnosis.json`: val metrics, per-class or per-field table, top confusions with examples, invalid-output examples, training loss curve, dataset stats, and every previous attempt's patch and result.
- The fixer returns one or more fixes from this closed menu, each with a reason. Code turns them into a config patch:

| Fix | Patch | Limits |
| --- | --- | --- |
| `more_epochs` | `train.epochs += k` | ≤ 6 total |
| `change_learning_rate` | `train.learning_rate = x` | within `tg fine-tuning model-limits` bounds |
| `use_more_data` | raise or remove `train.max_examples` | — |
| `bigger_model` | next entry in `fallback_models` | — |
| `add_examples` | frontier-label or generate N extra train examples for named weak classes | only if `targets.frontier_model` is set; needs a cost estimate |
| `drop_suspect_labels` | remove train rows where the label looks wrong (frontier second opinion) | max 5% of train; listed in the report |
| `stop` | end the loop with a reason | — |

- The fixer can never change the test set, the metric, the pass bar or the baselines.
- Final model = best attempt by val score. Its **test** numbers go in the report. Test predictions are computed for every attempt while its endpoint is up, but are never shown to the fixer.

**Done when:** on BANKING77 with a deliberately weak config (`max_examples: 300`, `epochs: 1`, `learning_rate: 1.0e-6`), the loop reaches a passing model within 3 attempts with no human input. Also covered by a dry-run test where `FakeTuned` accuracy improves when the right fix is applied.

---

## 14. Email cleaner (M5)

### 14.1 What it is

A model that predicts whether the user will treat a new email as important, read it, or skip it, trained on their own inbox history, and a daily job that labels new mail in Gmail.

### 14.2 Gmail connector

- OAuth desktop flow (`google-auth-oauthlib`); the user creates `credentials.json` in Google Cloud Console (ask them; give step-by-step instructions in the CLI output). Token cached at `~/.config/ft/gmail_token.json`.
- Scopes: `gmail.readonly` for `ft email pull`; `gmail.modify` requested separately, only for `ft email apply`.
- Pull the last 12 months (configurable). Skip drafts, chats and messages the user sent. Skip messages younger than 7 days (their outcome isn't settled yet).
- Per message: id, thread id, date, sender name and address, number of recipients, subject, snippet, plain-text body (first 2,000 chars, HTML stripped), whether a `List-Unsubscribe` header exists, Gmail category, label IDs.

### 14.3 task.yaml

```yaml
name: email
source: {type: gmail, months: 12, min_age_days: 7}
task:
  type: classify
  input: {kind: text, field: rendered}     # built by the connector, see below
  output: {field: behavior, labels: [important, read, skip]}
  instruction: Predict how this user will treat this email: important, read, or skip.
targets:
  from: behavior
  rules:                      # first match wins; boolean expressions over connector features
    - important: replied_after OR starred
    - read: opened
    - skip: NOT opened
splits: {by_time: date, val_fraction: 0.1, seed: 42}
baselines:
  - {model: claude-sonnet-5-5, effort: low, thinking: between_tools, context_file: examples/email/about_me.md}
  - {model: claude-haiku-5-5, effort: low, thinking: disabled, context_file: examples/email/about_me.md}
train: {provider: together, base_model: Qwen/Qwen3.5-4B, fallback_models: [Qwen/Qwen3.5-9B], lora: true, epochs: 3, learning_rate: 1.0e-4}
eval:
  primary_metric: macro_f1
  pass_bar: {compare_to: best_baseline, max_gap_points: 0}   # tuned must at least match the best baseline on val
budget: {max_spend_usd: 30, max_attempts: 3}
```

`rendered` input format:

```
From: <sender token> (<sender domain>)
Recipients: <n>
Category: <gmail category>
Unsubscribe link: yes|no
Subject: <subject>

<first 300 words of body>
```

`context_file` adds the user's self-description to the baseline's system prompt, so the frontier model gets a fair shot.

### 14.4 Behavior rules

- `opened` = the message lacks the `UNREAD` label.
- `starred` = has `STARRED`.
- `replied_after` = the thread contains a message from the user (label `SENT`) dated after this message.
- Rules are evaluated in order, first match wins. Expressions support `AND`, `OR`, `NOT` and parentheses over the connector's boolean features. Use a small safe parser; never `eval()`.
- Label noise is expected (preview panes mark mail as read). The report should show the class balance and note this.

### 14.5 Privacy

- Data stays in `runs/` (gitignored).
- Before anything leaves the machine (training upload, baseline calls), run redaction (`ft/redact.py`, Presidio analyzer + anonymizer): replace PERSON, EMAIL_ADDRESS, PHONE_NUMBER, LOCATION, CREDIT_CARD, IBAN_CODE, US_SSN and similar with type placeholders.
- Replace each sender address with a stable salted hash token (`sender_7f3a9c`) and keep the domain. The model still learns "this sender matters" without exposing who it is. Salt stored locally in `~/.config/ft/salt`.
- The report states exactly what was uploaded and where.

### 14.6 Apply

`ft email apply --run <run_id>`:
1. Fetch messages received since the last apply checkpoint (stored in `~/.config/ft/email_state.json`).
2. Render and redact them the same way as training data.
3. Deploy the run's best model, classify, tear down.
4. Add a Gmail label `ft/Probably skip` (create if missing) to messages predicted `skip`, and `ft/Important` to `important`. **Never archive, delete or mark as read.**
5. Save the checkpoint and a log of what was labeled.

Scheduling: document a cron line for the user's machine. Note in `NOTES.md` that each daily run pays for endpoint provisioning plus a few minutes.

### 14.7 Stretch

Download the LoRA adapter or merged weights (`client.fine_tuning.content(ft_id=..., checkpoint="adapter" | "merged")`) and run locally for the daily job, removing endpoint cost. Only after the MVP.

**Done when:** the tuned model matches or beats the best baseline's macro-F1 on the time-split test set (a tie counts; v0 uses `max_gap_points: 0`), and `ft email apply` labels a day of new mail correctly with teardown confirmed. The report words the result per the verdict rule in 9.11: "beat" / "fell short" only when the CI excludes 0, otherwise "ahead by" / "behind by X points, not statistically significant".

---

## 15. Logs mode (M6)

### 15.1 Log format

One JSON object per line, OpenAI-compatible, so a future proxy can write it directly:

```json
{"id": "...", "timestamp": "2026-10-17T14:30:00Z",
 "request": {"model": "claude-haiku-5-5", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]},
 "response": {"content": "card_arrival", "usage": {"input_tokens": 0, "output_tokens": 0}},
 "latency_ms": 412, "metadata": {}}
```

`connectors/logs.py` also accepts Anthropic Messages request/response shapes and normalizes them to this.

### 15.2 Simulating logs

`ft simulate-logs examples/banking77 --model claude-haiku-5-5 --n 5000` runs BANKING77 inputs through Claude with an "app-style" system prompt (a plausible product prompt, different from the baseline prompt) and writes `data/banking77-logs/logs.jsonl`. True labels go to a separate sidecar file used **only** for evaluation.

### 15.3 task.yaml

```yaml
name: banking77-logs
source: {type: logs, path: data/banking77-logs/logs.jsonl, truth_sidecar: data/banking77-logs/truth.jsonl}
task:
  type: classify
  input: {kind: text, field: last_user_message}
  output: {field: response_content, labels: auto}
  instruction: Classify the banking customer message into its intent label.
targets: {from: logs}
splits: {val_fraction: 0.1, test_max: 1000, seed: 42}
baselines: []                    # the logged responses are the baseline; no extra Claude calls
train: {provider: together, base_model: Qwen/Qwen3.5-4B, lora: true, epochs: 3, learning_rate: 1.0e-4}
eval:
  primary_metric: agreement      # tuned vs logged responses
  pass_bar: {min_value: 95.0}
budget: {max_spend_usd: 30, max_attempts: 3}
```

- Primary metric is **agreement** with the logged frontier answers ("does it behave like what you use today?"). It's computed like accuracy, with the logged responses as targets.
- The logged frontier model appears in the report as a pseudo-baseline: its cost per 1,000 requests comes from the logged usage, and, when a truth sidecar exists, its accuracy against truth sits next to the tuned model's.
- MVP assumes one task per log file. Grouping mixed logs by system prompt into separate tasks is a later agent feature.
- Before training on frontier outputs for anything beyond personal research, the user must check the provider's terms (open decision in the brief). Print this reminder when `targets.from` is `logs` or `frontier`.

**Done when:** the simulated-logs run reaches ≥ 95% agreement on test, and the report shows both agreement and truth accuracy.

---

## 16. Milestones

Stop after each milestone, show results, and wait for approval.

| # | Milestone | Done when |
| --- | --- | --- |
| M0 | Skeleton and dry run | Repo, CLI, config schema, run folder, events, spend guard, fakes. `ft run examples/banking77 --dry-run` produces a full report offline; tests pass. |
| M1 | BANKING77, real | Real report with Haiku and Sonnet baselines and a tuned Qwen3.5-4B; endpoint torn down (verified via `ft status`); spend recorded within budget. |
| M2 | CORD (images, extraction) | Vision and extraction support added to `tasks/` and providers; BANKING77 and CORD both run from the same code path with only `task.yaml` differing. |
| M3 | The agent | Section 12 done-when. |
| M4 | Retry loop | Section 13 done-when. |
| M5 | Email cleaner | Section 14 done-when. |
| M6 | Logs mode | Section 15 done-when. |

**MVP complete** = M0–M6 done, and a final run of all four proof-set datasets shows zero human interventions in their reports.

---

## 17. Engineering conventions

- Python 3.12, `uv` for environments, `pyproject.toml`, `ruff` for lint/format, `pytest`.
- Libraries: `typer`, `pydantic>=2`, `pyyaml`, `anthropic`, `together[cli]`, `openai`, `httpx`, `tenacity`, `jinja2`, `rich`, `python-dotenv`, `datasets`, `pillow`, `scikit-learn` (metrics). M5 adds `presidio-analyzer`, `presidio-anonymizer` (plus a spaCy English model), `google-api-python-client`, `google-auth-oauthlib`, `html2text`.
- Secrets in `.env` (`ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`); `.env.example` committed with empty values. Never log keys or full email bodies.
- `.gitignore`: `runs/`, `data/`, `.env`, `credentials.json`, `*token*.json`.
- Stages contain no provider SDK imports; task-type logic (prompts, parsing, normalization, scoring) lives only in `ft/tasks/`.
- All randomness seeded from `task.yaml`.
- Tests:
  - Unit: splitting (stratified, by time, leakage check), normalization and scoring for both task types, bootstrap CI, cost and break-even math, config validation, log normalization, behavior rules, redaction.
  - Integration: full dry-run of BANKING77 and CORD fixtures; resume after a simulated crash mid-train; teardown runs when predict raises; spend guard stops an over-budget run.
  - Live tests marked `@pytest.mark.live`, skipped by default, each with a tiny dataset and a hard spend cap.

---

## 18. Verify before the first real run

Do these at the start of M1 (and M2 for vision) and record results in `NOTES.md` and `models.yaml`:

1. `tg beta models configs Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`: both deployable? Which hardware configs, and what hourly price?
2. `tg fine-tuning model-limits Qwen/Qwen3.5-4B`: learning-rate bounds, LoRA rank limits.
3. Together vision fine-tuning docs: is `Qwen/Qwen3-VL-8B-Instruct` (or another small VLM) tunable **and** deployable? Exact image format in the training file.
4. Together pricing page: LoRA training price per million tokens for these models.
5. CORD on Hugging Face: field names and `gt_parse` structure match section 9.1.
6. Anthropic: structured outputs with a 77-value enum compile without a complexity error; `effort` and `thinking` combinations in `baselines` are accepted.

If any check fails, stop and ask the user before substituting.

---

## 19. Reference links

- Together fine-tuning overview: https://docs.together.ai/docs/fine-tuning/overview
- Together quickstart (upload, create, poll, deploy, call): https://docs.together.ai/docs/fine-tuning/quickstart
- Together deploy a fine-tuned model: https://docs.together.ai/docs/fine-tuning/deployment
- Together supported models: https://docs.together.ai/docs/fine-tuning/supported-models
- Together vision fine-tuning: https://docs.together.ai/docs/fine-tuning/vision
- Together data preparation: https://docs.together.ai/docs/fine-tuning/data-preparation
- Claude models: https://platform.claude.com/docs/en/models/overview
- Claude pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Claude effort: https://platform.claude.com/docs/en/build-with-claude/effort
- Claude structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Claude prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Claude batch processing: https://platform.claude.com/docs/en/build-with-claude/batch-processing
- Claude tool use: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- BANKING77 CSVs: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data (train.csv: 10,003 rows; test.csv: 3,080 rows; columns `text`, `category`; 77 labels)
- CORD v2: https://huggingface.co/datasets/naver-clova-ix/cord-v2
- Fireworks LoRA serving (for a later provider): https://docs.fireworks.ai/faq-new/billing-pricing/are-there-extra-fees-for-serving-fine-tuned-models
