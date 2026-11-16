# Heron

Automated LLM fine-tuning.

**Give it a labeled dataset. An agent plans the run, fine-tunes a small open model on real GPUs, fixes its
own failed attempts, and benchmarks the result against a frontier model on held-out data with confidence
intervals.**

See the finished runs: **[heron-finetune.vercel.app](https://heron-finetune.vercel.app/)**.

## Results (real runs, held-out test sets)

| Dataset | Fine-tuned Qwen3.5-4B | Best GPT baseline (gpt-6-luna) | Difference, 95% CI | Latency (p50) | $ / 1k requests |
|---|---|---|---|---|---|
| BANKING77 (77 intents, 3,080 test) | **92.4%** accuracy | 86.0% (1-shot) | **+6.4** [+5.2, +7.5] | 0.24 s vs 1.21 s | $0.025 vs $0.046 |
| LEDGAR (100 contract clause types, 5,000 test) | **84.1%** accuracy | 73.6% (zero-shot) | **+10.4** [+9.3, +11.5] | 0.48 s vs 1.18 s | $0.046 vs $0.106 |
| BANKING77 from a deliberately broken config | **90.6%** | 85.8% (1-shot) | **+4.8** [+3.7, +5.9] | 0.39 s vs 1.03 s | $0.038 vs $0.041 |
| Enron email folders (time split, 840 test) | 48.7 macro-F1 | 52.9 (3-shot) | −4.2 [−8.1, +1.7], not significant | 0.70 s vs 2.20 s | $0.092 vs $0.337 |

Every row is a run with **zero human edits after approval**.

## What the agent does

1. **Reads the data:** profiles columns, samples rows, finds the input and the label, and checks there are
   enough examples and that outputs can be graded automatically.
2. **Writes the run config:** task type, labels, metric and splits, validated against the schema, with no
   hand-written YAML. Datasets that can't be graded automatically are flagged before any spend.
3. **Plans under rules enforced in code:** smallest deployable model, zero-shot and few-shot GPT baselines,
   test-set size, metric choice, cost and time estimate, budget. A human checkpoint shows the plan before
   any spend.
4. **Trains and benchmarks:** LoRA on Together or Fireworks, paired-bootstrap comparison on test, and cost per
   1,000 requests.
5. **Fixes itself:** if the model misses the bar, a fixer agent reads a validation-only diagnosis (loss
   curve, confusions, invalid outputs, data warnings) and picks fixes from a closed menu.
   - From a broken config (learning rate 1e-6, 1 epoch, 300 rows) it went from **0.8% to 88.7%** in one
     retry.
   - Its reasons were "learning rate far below the usual LoRA range", "only 300 of 8,993 rows used", and
     "free-form outputs: list the labels".

## Engineering highlights

- **Pass bars calibrated to the frontier model itself.** When training on GPT's logged answers, the bar is
  set relative to GPT's own self-agreement (93.75% on re-asking), so imitation is graded against what is
  actually achievable.
- **Retrains on the newest data before shipping.** For time-split tasks the chosen config is retrained on
  train + val, which lifted Enron test macro-F1 from 42.8 to 48.7.
- **Tells serving problems apart from model problems.** The retry loop recognises a broken endpoint and
  stops, so it never pays for training fixes that can't help.
- **Root-causes provider issues cheaply.** A $0.27 base-model probe isolated an image-serving fault on the
  provider's side, confirmed against the tokenized training data, and the agent now routes around it.

## Run it

```bash
uv venv && uv pip install -e ".[together,demo]"
cp .env.example .env            # OPENAI_API_KEY, TOGETHER_API_KEY, FT_TOTAL_BUDGET_USD
ft run data/my-folder           # the agent plans, you approve at the checkpoint, then it runs
ft run examples/banking77 --dry-run   # the whole pipeline offline with fake models, no spend
```

- **Tests:** `uv sync && uv run pytest` (offline, no keys needed).
- **Web demo locally:** `FT_DEMO_DRY_RUN=1 uvicorn demo.server:app` (fake models), or without the flag for
  real runs.
- **Deploying the static site:** see [demo/README.md](demo/README.md).

## Stack

Python, FastAPI, Typer, Pydantic. OpenAI Responses API for the agent and the baselines; Together or Fireworks
(`train.provider` in the task file) for LoRA training and dedicated endpoints. 186 tests, with fake providers so the whole pipeline runs offline.
