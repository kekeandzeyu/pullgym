# v0.2 frontier matrix (Phase 4)

This page reports the uniform v0.2 frontier evaluation: 9 models from 6 labs, plus
the oracle and prior baselines, all under one harness. Everything below can be
regenerated from the job directories with `scripts/phase4_analysis.py` and
`pullgym report`.

**Headline.** When frontier tool-using agents write a valid answer, they reach
oracle-level calibration on Tier 5 as well as on Tiers 1–4. Tier 5 means
unflagged glitches, unmodelled drift, look-elsewhere scans and k = 2 datasheet
conventions, with no hint of any of them in the note. Seed-paired against the
exact reference analysis:

- **Reward gap.** Every model is within 0.08 nats of the oracle on valid answers.
- **Coverage.** No model's 68% coverage differs significantly from the oracle
  (exact McNemar, p ≥ 0.18).
- **Width.** Sharpness is about 1.0, so the intervals are not inflated.

What separates frontier models is answer validity. Some models spend the whole
20-turn budget exploring and never write an answer. Qwen3-8B models stay far
below the frontier on Tier 5 (cov68 13–47%, Phase 2), so Tier 5 is a
small-model and RLVR benchmark, not a frontier-separating one.

## Setup

- **Harness:** `0.2.0.dev0`, `pullgym run-fast`, `tools` mode
  (`run_python` / `run_bash` function calling).
  - Budget: `max_turns = 20` and `max_tokens = 16384` per call.
  - Every observation carries `turn k/N` and a schema-only `answer_status`.
  - Budget warnings come 3 turns before the end and on the final turn.
- **Seeds:** dev indices 0–24 for every family, the same seeds for every model.
  The oracle and prior run all 16 families × 25 seeds × both variants.
- **Design per model:**

  | Stage | Seeds | Episodes |
  |---|---|---|
  | Tier 5 `calibrated` | 25 seeds × 4 families | 100 |
  | Tier 5 `neutral` | 25 seeds × 4 families | 100 |
  | Tiers 1–4 `calibrated` saturation check | 5 seeds × 12 families | 60 |

  The expensive models ran a subset under hard budget caps. Their seeds were
  run chunk-major across families, so a capped model still has the same seeds
  in every family.
- **Sampling:** temperature as recommended by each vendor (1.0 for all rows
  here), provider-default reasoning settings.
- **Endpoints:**
  - Volcengine Ark: DeepSeek, Kimi, GLM.
  - OpenRouter: Anthropic, OpenAI, Google. Claude requests carry an ephemeral
    `cache_control` breakpoint on the newest message.
  - Recorded per model in `runs/phase4/models.json`.
- **Statistics:**
  - Wilson intervals for coverage.
  - Paired bootstrap CIs for reward and pinball differences against the oracle
    on shared tasks.
  - Exact McNemar tests for coverage differences.

### Why `tools` mode

A first attempt used `code` mode (bare `python` fences, no tools offered), which
is cheaper. On tool-native reasoning models it failed for harness reasons, not
statistical ones:

| Model | What it did in `code` mode |
|---|---|
| DeepSeek-V4.1-Flash | Put the code block in `reasoning_content` and left the content empty |
| DeepSeek-V4-Pro | Wrote `<tool_calls><invoke name="exec">` markup as plain text |
| GLM-5.2 | Garbled the fence (`¡python`) |
| Kimi-K3.1-Pro | Returned empty content |
| Gemini 3.x | Returned `MALFORMED_FUNCTION_CALL` with no content |
| gpt-oss-120b on Groq | Emitted a native `python` tool call that the provider rejected |

Scoring those as model failures would measure the harness, not calibration. That
attempt is archived under `runs/phase4/codemode_v1/` and excluded from every
table.

## Tier 5, `calibrated` prompt

Δ columns are model − oracle on the same tasks. Δrew_v is computed on tasks
both answered validly, and the Δcov68 counts are tasks covered only by the
model / only by the oracle.

| Model | n | Valid | cov68 [Wilson 95%] | cov95 | sharp (gm) | rew_v | rew_all | Δrew_v [95% CI] | Δcov68 (+/−, p) |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| Claude Haiku 5.5 | 100 | 100% | 72.0% [63, 80] | 97.0% | 1.15 | −0.079 | −0.079 | −0.05 [−0.11, +0.01] | 7/2, p = 0.18 |
| Claude Sonnet 5.5 | 100 | 100% | 71.0% [61, 79] | 98.0% | 1.20 | −0.107 | −0.107 | −0.08 [−0.14, −0.01] | 8/4, p = 0.39 |
| GPT-6 Luna | 100 | 100% | 66.0% [56, 75] | 96.0% | 1.08 | −0.086 | −0.086 | −0.06 [−0.11, −0.00] | 5/6, p = 1 |
| GPT-6.2 Sol | 100 | 100% | 67.0% [57, 75] | 96.0% | 1.05 | −0.063 | −0.063 | −0.03 [−0.08, +0.01] | 4/4, p = 1 |
| Kimi-K3.1-Pro † | 48 | 100% | 66.7% [53, 78] | 95.8% | 1.04 | −0.001 | −0.001 | −0.02 [−0.09, +0.05] | 1/3, p = 0.62 |
| Gemini 3.8 Flash † | 34 | 94% | 71.9% [55, 84] | 93.8% | 1.05 | +0.043 | +0.021 | −0.02 [−0.12, +0.07] | 2/2, p = 1 |
| DeepSeek-V4.1-Flash (economy) | 100 | 96% | 70.8% [61, 79] | 95.8% | 1.16 | −0.101 | −0.178 | −0.05 [−0.10, +0.00] | 7/2, p = 0.18 |
| DeepSeek-V4-Pro † | 51 | **69%** | 74.3% [58, 86] | 94.3% | 1.01 | +0.050 | **−1.020** | −0.01 [−0.05, +0.03] | 0/0 |
| GLM-5.3-Flash | 100 | **56%** | 73.2% [60, 83] | 96.4% | 1.10 | −0.058 | **−1.365** | +0.01 [−0.07, +0.11] | 4/1, p = 0.38 |
| *oracle* | 100 | 100% | 67.0% [57, 75] | 95.0% | 1.00 | −0.029 | −0.029 | — | — |
| *prior* | 100 | 100% | 72.0% [63, 80] | 97.0% | 12.90 | −2.214 | −2.214 | −2.19 [−2.45, −1.91] | 22/17, p = 0.52 |

† Budget-capped subset of seeds, chunk-major, so the seeds are the same in
every family: Kimi seeds 0–11, Gemini 3.8 Flash and DeepSeek-V4-Pro about 0–12.

### Per family, pooled over the 9 frontier models

| Family | Valid | cov68 frontier / oracle | cov95 frontier / oracle | Mean pull frontier / oracle | rew_v frontier / oracle |
|---|---:|---|---|---|---|
| `outliers` | 91% | 75% / 76% | 100% / 100% | −0.07 / −0.07 | −0.002 / +0.095 |
| `unmodeled_drift` | 89% | 67% / 60% | 94% / 92% | −0.39 / −0.36 | −0.256 / −0.150 |
| `look_elsewhere` | 90% | 73% / 68% | 94% / 92% | −0.08 / −0.13 | +0.014 / −0.032 |
| `datasheet_metrology` | 93% | 65% / 64% | 96% / 96% | −0.20 / −0.21 | −0.023 / −0.028 |

Each family has a shortcut analysis that fails it. In the slow tests in
`tests/test_tier5.py`, the shortcuts lose at least 1.5 nats of mean reward
against the reference on `outliers`, 5.0 on `unmodeled_drift`, 0.3 on
`look_elsewhere` and 2.0 on `datasheet_metrology`. The frontier agents do not
take the shortcuts. Without being told, they:

- down-weight or mix out the glitches;
- model the drift;
- account for the scan;
- apply k = 2 and the rectangular-tolerance conventions.

`unmodeled_drift` has the largest reward gap (−0.11 nats), but its coverage
matches the oracle.

## Prompt variant: `calibrated` vs `neutral` (Tier 5)

| Model | cov68 cal → neu | cov95 cal → neu | Valid cal → neu |
|---|---|---|---|
| Claude Haiku 5.5 | 72% → 69% | 97% → 95% | 100% → 100% |
| Claude Sonnet 5.5 | 71% → 71% | 98% → 98% | 100% → 100% |
| GPT-6 Luna | 66% → 64% | 96% → 93% | 100% → 100% |
| DeepSeek-V4.1-Flash | 71% → 68% | 96% → 96% | 96% → 95% |
| GLM-5.3-Flash | 73% → 70% | 96% → 95% | 56% → 59% |

Removing the hint that intervals are scored for calibration changes nothing
beyond seed noise.

## Tiers 1–4 under v0.2 (saturation check, 5 seeds × 12 families)

Paired against the oracle per tier, valid answers from Claude Haiku 5.5, GPT-6
Luna and DeepSeek-V4.1-Flash are task-for-task the reference analysis:

- **Coverage:** at most 1 discordant task per tier.
- **Reward:** every per-tier Δrew_v is within ±0.10. The largest is Haiku on
  Tier 1, −0.10 [−0.22, −0.01].
- **Validity:** DeepSeek's only loss is on Tier 2 (87% valid). v0.1's
  conclusion holds under the v0.2 harness.

## How frontier agents lose: turn budget, not statistics

| Model | Valid | Turn budget exhausted | Stopped without valid answer | Median turns | Median prompt tokens (cached) |
|---|---:|---:|---:|---:|---|
| Claude Sonnet 5.5 | 200/200 | 0 | 0 | 5 | 12k (78%) |
| GPT-6.2 Sol | 100/100 | 0 | 0 | 6 | 16k (73%) |
| Claude Haiku 5.5 | 260/260 | 0 | 0 | 6 | 17k (78%) |
| GPT-6 Luna | 260/260 | 0 | 0 | 8 | 18k (73%) |
| Kimi-K3.1-Pro | 48/48 | 0 | 0 | 10 | 83k (81%) |
| Gemini 3.8 Flash | 32/34 | 2 | 0 | 20 | 110k (46%) |
| DeepSeek-V4.1-Flash | 249/260 | 10 | 1 | 20 | 271k (88%) |
| DeepSeek-V4-Pro | 35/51 | 16 | 0 | 20 | 299k (86%) |
| GLM-5.3-Flash | 111/196 | 55 | 30 | 20 | 362k (86%) |

Of the 116 invalid frontier episodes on dev and test, 113 never wrote
`artifacts/answer.json`. The other 3, all GLM, ended on turn 20 with a
schema-invalid file: a missing key twice, a NaN once. The budget warnings at
turns 17 and 19 do not stop the long explorers. The same models are also the
most expensive per episode, by roughly 20×.

## Held-out `test` split (dev-overfitting check)

The test seeds are `SHA-256(secret:family:i)` for i = 0–24, and the secret is
never committed. Family parameters were tuned only on calibrate, probe and
scratch seeds.

Tier 5, `calibrated` prompt, on the test split:

| Model | n | Valid | cov68 [Wilson 95%] | cov95 | rew_v | Δrew_v vs oracle [95% CI] | Δcov68 (+/−, p) |
|---|---:|---:|---|---:|---:|---|---|
| Claude Haiku 5.5 | 100 | 100% | 69.0% [59, 77] | 98.0% | −0.223 | −0.26 [−0.68, −0.02] * | 6/1, p = 0.12 |
| GPT-6 Luna | 100 | 100% | 67.0% [57, 75] | 99.0% | −0.008 | −0.04 [−0.09, +0.01] | 5/2, p = 0.45 |
| DeepSeek-V4.1-Flash (economy) | 100 | 97% | 66.0% [56, 75] | 99.0% | −0.011 | −0.03 [−0.08, +0.01] | 4/1, p = 0.38 |
| *oracle* | 100 | 100% | 64.0% [54, 73] | 97.0% | +0.033 | — | — |

\* Haiku's mean comes from one episode.

- **The episode:** on `datasheet_metrology`, Haiku reported the resistance
  ×1000 too large: q50 ≈ 264,000 Ω against a true 268 Ω, a pull of +58. That
  answer scores the −20 floor and alone moves the mean by about −0.2.
- **Without it:** Δrew_v = −0.055 over the other 99 tasks, and the median
  paired difference over all 100 is −0.03.
- **How rare it is:** it is the only unit-scale error among roughly 350 valid
  frontier `datasheet_metrology` answers. The unit trap does catch frontier
  models, but rarely.

The test split reproduces the dev conclusions: oracle-level coverage and reward
on valid answers, with validity the only loss. Tuning families on scratch seeds
did not overfit the dev seeds.

## Cost

| Endpoint | Model | Episodes | Cost |
|---|---|---:|---|
| OpenRouter | Claude Haiku 5.5 | 260 dev + 100 test | $0.76 |
| OpenRouter | GPT-6 Luna | 260 dev + 100 test | $1.12 |
| OpenRouter | Claude Sonnet 5.5 | 200 | $5.05 |
| OpenRouter | GPT-6.2 Sol | 100 | $4.01 |
| OpenRouter | Gemini 3.8 Flash | 34 | $2.55 |
| Ark | DeepSeek-V4.1-Flash (economy) | 260 dev + 100 test | 38.9 CNY (list) |
| Ark | DeepSeek-V4-Pro | 51 | 61.4 CNY (list) |
| Ark | Kimi-K3.1-Pro | 48 | 53.5 CNY (list) |
| Ark | GLM-5.3-Flash | 200 | 42.1 CNY (list) |

OpenRouter costs are its billed `usage.cost`. Ark costs are computed from
recorded tokens at list prices, with cached prompt tokens at the cached price
and before free quotas. The matrix rows cost $13.49 on OpenRouter ($14.55 billed in total, including
pilots and episodes lost to dropped connections) and 196 CNY at Ark list
prices, excluding pilots.

Prefix caching matters: 73–88% of prompt tokens were cache hits.

## Limitations

- **Small subsets.** The capped expensive models (Kimi, DeepSeek-V4-Pro, Gemini
  3.8 Flash) have 34–51 Tier-5 episodes. Their per-family numbers are
  indicative only.
- **One draw per task.** Phase 2 showed that validity flips between draws for
  8B models. Frontier validity is near 100% for most rows, so this matters
  mainly for the DeepSeek-V4-Pro and GLM validity numbers.
- **Gemini 3.5 Flash-Lite** (free tier) is not included. Its free quota is
  20–hundreds of requests a day, and the row was still accumulating.
- **Google's top model is not in the table.** Gemini 4 Hydrogen cost $0.33 per
  episode in the pilot, too much for this budget.
- **Provider routing.** The rows depend on Ark and OpenRouter serving the
  model IDs in `models.json`; provider-side quantisation or routing is not
  controlled.
