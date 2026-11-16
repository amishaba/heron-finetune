Bug report sent to Together on 2026-10-17.

**Subject:** Dedicated Qwen3.5-4B endpoints treat image inputs as noise (base model, no fine-tune)

Hi Together team,

Dedicated endpoints running Qwen/Qwen3.5-4B receive images and count their tokens, but the model answers as if each image were blank or noise. This happens with the base model, no fine-tune involved, so it looks like a serving problem.

**Setup**
- Model: `Qwen/Qwen3.5-4B` (base), on a dedicated endpoint with its only deploy config, `cr_CfHSNAes5gNmGHLdTQSVv` (1× H100).
- Endpoint: deployed 2026-10-17 at about 21:56 UTC and deleted after the test.
- Request: OpenAI-compatible chat completions, sent with the OpenAI Python SDK 3.33.0.
  - One user message: a text part, plus an `image_url` part holding a base64 JPEG data URL (receipt photos, about 850×1280).
  - `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`.

**What happens**
- **"What is the total?"** I asked this about 5 different receipts (true totals 45,500 / 23,000 / 20,000 / 28,000 / 41,000). The endpoint answered `10.00` all 5 times.
- **Extraction prompt** (receipt totals and line items):
  - It returned the same made-up one-item receipt ("Coke", 10.00), or an empty one.
  - Once it replied: *"The image appears to be corrupted or contains only noise, making it impossible to read any text or identify the receipt details."*
- **The image does arrive:** prompt tokens varied with image size, from 329 to 1,238 per call.

**What works**
- I sent the identical request (same code, same images) to serverless `Qwen/Qwen3.5-9B`, and it read the receipt total correctly.

**Also seen on a fine-tune**
- I trained a vision LoRA fine-tune of Qwen3.5-4B:
  - job `<job id>`;
  - model `<account>/ft-cord-a1`;
  - served on its own dedicated endpoint.
- On that endpoint it returned one identical empty JSON template for all 200 test receipts.
- I don't think the training is at fault:
  - The tokenized data from `retrieve_tokenized_dataset` shows each image aligned with its answer.
  - The training loss fell normally.
- So I believe the fine-tune hits the same serving problem as the base model.

**Questions**
1. Are image inputs supported on dedicated endpoints for Qwen3.5-4B? The deploy config's certification lists only text benchmarks.
2. If they are, is there a different request format or deploy config we should use?
3. If this is a bug, is there a timeline for a fix, or a config (for example one with `adapter_mode`) that serves vision correctly in the meantime?

I'm happy to share the request script and the test images.

Thanks
