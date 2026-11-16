# Auto Fine-Tuning Pipeline: Idea Brief

Oct 16, 2026

## The idea

Data in, deployed model out, no ML engineer. An agent turns a company's raw data into a small tuned open model that does one task as well as a frontier API, for far less.

- **The product is the automation.** The agent makes every call an ML engineer would: what the task is, where the right answers come from, which base model to use, and whether the result is good enough.
- **A retry loop fixes weak runs on its own.** When a tuned model misses the bar, the agent reads its mistakes, changes the data or the model, and trains again.
- **Text or images.** Support messages, contracts, scanned receipts, camera frames: the same pipeline with a different base model.
- **Not a training company.** GPUs, training and hosting are rented from existing providers.
- **What the customer gets:** a one-page report (accuracy, cost, speed vs. their current API) and a drop-in API endpoint.
- **Stage:** personal project first; decide later whether it becomes a business.

Two kinds of win:

1. **Cheaper:** matches a frontier model on a narrow task at a fraction of the cost.
2. **Better:** beats a frontier model when the task depends on knowledge only the customer's data holds, such as their own categories, customers or habits.

Core claim to prove: one pipeline takes very different datasets from raw data to a passing model with zero human interventions.

## When it fits

Fine-tuning wins on narrow, repeated, high-volume tasks with clear right answers. It loses on changing facts and open-ended work.

**Good fit**

- Classify, route, extract, produce structured output, keep a fixed tone or format, call a fixed set of tools
- Text or images: messages, documents, scanned pages, camera frames
- Outcomes already recorded: clicks, replies, approvals, routing decisions, edits. When behavior is logged, labeling is free.
- Past examples of good outputs exist, or can be generated
- Outputs can be graded automatically
- At least one pressure: cost, speed, data residency or language

**Poor fit**

- Facts that change: use retrieval (RAG) instead
- Broad, open-ended reasoning: keep a frontier model
- Low volume: savings are too small to matter
- No way to judge a good output

Rule: the agent's first job is to decide fit, and to say no when it isn't one.

## How customers plug in

Two ways in, one output: a tested model behind an OpenAI-compatible endpoint.

**Logs mode (first to sell)**

- Customer already uses GPT or Claude for the task and routes those API calls through the pipeline.
- Prompts and responses are logged; they become labeled training data.
- A small model is trained to match, tested, then traffic switches over.
- Zero data work for the customer.
- Check that the source model's license allows training on its outputs; OpenRouter flags which models do.

**Data mode**

- For tasks not yet using AI, or where the history already holds the answers.
- Customer connects a database, inbox or file store; the agent proposes tasks from historical records.
- Strongest when outcomes are already recorded: a ticket's final routing, an approved invoice, an email that got a reply. Those become labels at no cost.
- Customer approves; the pipeline trains.

**Output (both modes)**

- Switching = changing the base URL and model name in existing code.
- Per-customer models are cheap: Fireworks serves hundreds of LoRA adapters on one deployment at the base model's price.
- One human checkpoint: the customer approves the task and the test set.

## How it's built

The agent decides and plain code executes. The agent writes a task config; ordinary scripts run every stage from it, so any bad run shows exactly what was decided.

&#91;embedded content: pipeline flow · 8 steps, 1 decision, 1 retry loop\]

The two highlighted steps are the product; the rest are rented services run from the agent's config. A weak result loops back through the agent, not to a person.

A config the agent might write for BANKING77:

```yaml
task: classify
input: text              # or image
output: one of 77 labels
labels_from: dataset     # or behavior, or frontier_model
base_model: Qwen3-8B
test_size: 500
pass_if: within 2 points of frontier accuracy
```

| Stage | Use | You build |
| --- | --- | --- |
| Pull data | Files and Gmail for v1; dlt for databases and APIs | Thin connectors |
| Define the task | LLM agent | The product |
| Get the right answers | Recorded outcomes, or a frontier model through its batch API | Agent picks the source |
| Clean, remove personal data, balance | NeMo Data Designer, Presidio | Config |
| Train | Together for v1; Fireworks or Tinker later | One SDK call |
| Test against the frontier model | Eval Protocol or Braintrust | Agent writes the test set |
| Retry when it misses | LLM agent | The product |
| Deploy | Provider's endpoint | Nothing |
| Retrain on a schedule | Modal or similar | A schedule |

- **The agent:** reads the schema, sample rows or sample images; proposes the task; picks where labels come from; writes the test set and the config.
- **The retry loop:** when the tuned model misses the bar, the agent reads its errors and picks a fix (more examples of weak categories, relabeling, a bigger base model), within a cap on attempts and spend.
- **Together for v1:** small text models (LoRA up to 16B) and Qwen3-VL-8B for images in one place, so text and image tasks share one code path.
- **Vision fine-tuning is available on all three candidates:** [Fireworks](https://www.fireworks.ai/blog/vlm-tuning) (Qwen 2.5 VL), [Together](https://together.ai/blog/fine-tuning-update) (Qwen, Google and Meta models) and [Tinker](https://thinkingmachines.ai/news/tinker-general-availability/) (Qwen3-VL).
- **One folder per run:** config, data splits, results, report and the agent's reasoning, so any run can be inspected or replayed.
- **dlt:** loads from APIs, databases and files; infers schemas; pulls only new records.
- **NeMo Data Designer:** builds datasets from seed data and scores quality with an LLM judge.
- **Dropped: the distil labs shortcut.** It automates the same steps this product exists to automate; wrapping it means never building the part that matters.

## Costs

A full text run costs about $20–75. Serving a tuned small model costs about $0.20 per million tokens.

**Training prices**

- Fireworks: $0.50 per 1M training tokens, LoRA, models under 16B
- Together: $0.48 per 1M training tokens, LoRA, models up to 16B
- Training tokens = dataset tokens × epochs

Example run: 5,000 examples × \~600 tokens × 3 epochs ≈ 9M training tokens.

| Item | Cost per run (USD) |
| --- | --- |
| Training | \~$5 |
| Labeling or synthetic data with a frontier model | \~$10–60 (estimate) |
| Testing, baseline vs. tuned | A few dollars (estimate) |
| Agent calls | A few dollars (estimate) |
| Total | \~$20–75 |

- **Behavior-labeled runs skip the labeling line.** When outcomes are already recorded, a run is mostly training and testing (estimate).
- **Image runs cost more.** Each image adds many tokens to training and testing; measure the real figure on CORD.

**Serving**

- Fireworks charges nothing extra to serve a fine-tuned model vs. its base model.
- 4–16B models: \~$0.20 per 1M tokens. 1B tokens a month ≈ $200.
- Avoid dedicated endpoints early: Together bills them per minute, even when idle.

Prototype budget: low hundreds of dollars in API credits.

## Timeline

Each customer run takes hours. Building v1 takes about 2–4 weekends.

**Per customer run**

- Data pull and testing: minutes
- Data generation and training: usually under a couple of hours each (estimate)
- Retry loop: adds one training run per attempt
- Reference point: distil labs says about 8 hours end to end
- Slowest parts: getting data access and human approval of the test set

**Building it (estimates, using coding agents)**

- v1, one task and one data source, end to end: 2–4 weekends
- Agent reliable on unfamiliar customer data: another 1–2 months
- Milestones and their done-when checks are under Build milestones.

## Markets and customers

The best customers repeat one text or image task at high volume and feel at least one pressure: cost, speed, data residency or language.

| Customer | Example tasks | Why they'd pay |
| --- | --- | --- |
| AI startups with big API bills | Whatever their product does at volume | Margin; logs mode cuts cost directly |
| Vertical SaaS (dental, vet, legal, property management, field service, freight, construction, restaurants, auto dealers, insurance agencies) | Ticket triage, note summaries, form extraction, plain-language search | Differentiation; AI cost they can't pass on to small-business customers |
| Outsourcing firms (Philippines, India, LatAm, Eastern Europe) | Support replies, back-office data entry, call scoring | Huge volume, thin margins |
| Banks, insurers, hospitals, government | Claims triage, KYC documents, medical coding, citizen request routing | Data can't go to a US API; open models run in-country or on-premises |
| Non-English markets (Southeast Asia, India, Africa, Middle East, LatAm) | Support, routing, document extraction in local languages | Frontier models weaker there; non-English text uses more tokens |
| Camera and document-image tasks (factories, retail, insurance, property, logistics) | Defect checks, empty-shelf detection, damage photos, scanned forms | Images use many tokens per call, so frontier vision bills grow fastest; small models can run on-site |
| E-commerce and marketplaces | Product categorization, attribute extraction, listing moderation, search relevance | Millions of products; small savings per item add up |
| Fintech, payments, accounting | Transaction categorization, merchant name cleanup, invoice extraction | Mandatory e-invoicing (Brazil, Mexico, India) creates huge extraction volume |
| Trade and logistics | Customs (HS) code classification, shipping document extraction | Every cross-border shipment needs it |
| Healthcare admin (not clinical) | Medical coding, prior authorization, scheduling | High volume, expensive staff time |
| Legal | Clause extraction, contract review, discovery sorting | Long documents make frontier APIs expensive |
| Edge and offline (field service, factories, checkout systems, cars, low-connectivity apps) | Voice and text commands, on-device assistants | Small models run with no connection |
| HR and recruiting | Resume parsing, job matching | Volume, consistent structured output |

**Order of attack**

1. AI startups: already understand the problem; logs mode needs no integration.
2. Vertical SaaS.
3. International: data residency plus non-English languages, where frontier APIs are weakest.

## Competitive landscape

Generic “upload data, get a model” is crowded and consolidating. The opening is starting from raw data, plus vertical depth.

**Who's there**

- Acquired: OpenPipe by CoreWeave, Sep 2025; Predibase by Rubrik, $100M+, Jun 2025
- Training and inference clouds: Together, Fireworks (incl. managed reinforcement fine-tuning), Baseten
- Training API: Tinker by Thinking Machines, generally available since Dec 2025
- Automated small-model builder: distil labs
- Enterprise continuous learning: Adaptive ML, $20M seed; Applied Compute, $80M, ex-OpenAI founders
- Data platforms: Snowflake Cortex Fine-tuning and Cortex Training, preview Jun 2026; Databricks fine-tunes on governed tables
- Hyperscalers: AWS Bedrock trains from API logs; OpenAI offers reinforcement fine-tuning

**Where the gap is**

- **Starting before the dataset exists.** Every training provider assumes the customer shows up with a clean, labeled dataset. This product starts at raw data and makes the decisions an ML engineer would.
- Vertical templates: task definitions, data mappings and test sets per industry
- Companies whose data sits in their own database, not Snowflake or Databricks
- Non-English and data-residency markets

Risk: a thin layer of connectors plus a config screen is easy for providers to copy. The agent, the retry loop and vertical templates are the moat.

## Proof plan

The idea holds if one pipeline takes four very different datasets from raw data to a passing model with zero human interventions.

Success metric: human interventions per run, target zero. Every intervention a run needs is the next thing to automate.

**Core proof set**

| Dataset | Task | What it proves |
| --- | --- | --- |
| BANKING77 | Customer messages → one of 77 intents | Easy text baseline with known right answers |
| CORD | Receipt photos → structured fields | Images run through the same pipeline |
| Simulated logs | Any dataset run through a frontier model, saved as logs | Logs mode works end to end |
| Own email (Gmail) | Sender, subject, opening text → important, read or skip | Behavior labels; a tuned model beats a frontier model |

Stretch goals: CUAD (clauses from long contracts, hard) and MASSIVE (one non-English language).

**Email cleaner demo**

- Labels from behavior: replied or starred = important; opened = read; never opened, then archived or deleted = skip.
- Baseline: a frontier model given a description of the user. The tuned model should win because it learned real habits.
- Output: a daily job applies a Gmail label such as “Probably skip.” It labels, never deletes.
- Watch: noisy labels (emails read in the preview pane) and privacy. Strip names and addresses with Presidio before upload.

**More public datasets (Hugging Face or Kaggle)**

| Use case | Dataset | Task |
| --- | --- | --- |
| Support triage | Bitext Customer Support | Sort messages by intent |
| Legal | LEDGAR | Classify contract provisions |
| EU documents, 23 languages | MultiEURLEX | Classify EU laws |
| Receipts and invoices | SROIE | Extract fields from scanned receipts |
| Finance | FiNER-139 | Tag numbers in financial filings |
| Ask your data | BIRD, Spider | Questions to SQL, graded by running the query |
| Agents and tool use | Salesforce xLAM function-calling 60k | Pick the right API call and fill its arguments |
| E-commerce | Amazon ESCI Shopping Queries | Judge search relevance |
| Visual inspection | MVTec AD | Defect yes or no on factory part photos |
| Healthcare, no privacy issues | MTSamples, Synthea | Classify specialty; extract from synthetic records |
| Logs mode | WildChat | Real chatbot conversations; can the agent spot tasks worth tuning? |

Note: check licenses before a commercial demo; several datasets are research-only, MVTec AD included.

## Build milestones

Seven milestones, each with a check that says it's done. Start with BANKING77, not email: known right answers mean a bad result points at the pipeline, not the data.

1. **BANKING77, hardcoded, end to end.** Record baseline numbers from a frontier API first. Done when a real report comes out.
2. **Config-driven.** Move every hardcoded choice into the config. Done when CORD runs by editing only the config.
3. **Agent writes the config.** Done when BANKING77 and CORD both run with zero manual edits.
4. **Retry loop.** Done when a deliberately weak first run (too little data, too small a model) gets fixed automatically.
5. **Email cleaner.** Done when a daily job labels new mail and the tuned model beats the frontier baseline.
6. **Logs mode.** Done when simulated logs train a replacement and one changed line switches traffic to it.
7. **Show it to people.** Show the reports to 3–5 people at vertical SaaS companies; ask if they'd pay for it on their own data.

From milestone 3 on, every hand fix a new dataset needs is the next thing to automate.

## Open decisions and next steps

The v1 backend is decided: Together. Data privacy needs an answer before real customer data; the vertical and personal-vs-business can wait until after the demo.

- [ ] Start milestone 1: baseline numbers on BANKING77 from a frontier API
- [ ] List which frontier models' licenses allow training on their outputs (needed for logs mode)
- [ ] Decide whether uploading customer data to a training provider is acceptable, or whether some customers need training run elsewhere
- [ ] Check dataset licenses before any commercial demo
- [ ] After the demo: pick the first vertical (ideally one you know from the inside)
- [ ] After the demo: decide personal project vs. business (sets how much to invest in connectors)

## Sources

Added in this revision:

- [Fireworks: Introducing Vision-Language Model Fine-tuning](https://www.fireworks.ai/blog/vlm-tuning)
- [Together AI: fine-tuning update with tool calling, reasoning and vision](https://together.ai/blog/fine-tuning-update)
- [Together AI: vision fine-tuning docs](https://docs.together.ai/docs/fine-tuning-vlm)
- [Thinking Machines: Tinker general availability and vision input](https://thinkingmachines.ai/news/tinker-general-availability/)

From the first version of this brief:

- Sacra: OpenPipe
- Rubrik to acquire Predibase
- Thinking Machines / Tinker (FutureSearch)
- Adaptive ML seed round
- Applied Compute launch
- Snowflake Cortex Fine-tuning
- Snowflake Cortex Training (Atlan)
- Databricks: fine-tuning on governed data
- dlt on GitHub
- dltHub profile
- NeMo Data Designer
- Fireworks Build SDK
- Eval Protocol
- distil labs
- distil labs talk
- OpenRouter: distillable models
- Inference hosting platforms 2026
- Fireworks pricing (UsagePricing)
- Fireworks pricing
- Fireworks (Beri)
- Fireworks pricing calculator (CostBench)
- Fine-tuning platforms 2026 (DevToolLab)
- Together AI fine-tuning pricing
