# pullgym — Technical Audit, Bottlenecks & ML/RL Roadmap

## 1. Current Status Snapshot (~92% Complete for `v0.1`)

### Completed & Verified
* **Core Benchmark & Calibrated Reference Suite (`src/pullgym/`):** 12 experiment families across 4 tiers (`line_fit` through `dead_time`, `correlated_norm`, `no_signal`), five-quantile proper `log_score_reward`, verified on 2,000+ seeds (`pullgym calibrate`, `pullgym probe`), plus secret-seed held-out Harbor export (`pullgym export`).
* **Milestone 1 — Fast Zero-Docker Multi-Turn Runner (`pullgym run-fast`, Done):** Ephemeral read-only workspaces keeping `truth` and `ref_sigma` strictly in parent memory, stdlib OpenAI-compatible multi-turn `run_python` / `run_bash` loop with `--resume` and parallel concurrency (`src/pullgym/runner.py`, exact Harbor parity verified in `tests/test_runner.py`).
* **Milestone 2 — Controlled Distributional RLVR vs. Binary GRPO (`Qwen3-8B`, Done):** Two controlled 2-arm GRPO rounds (1,600 evaluated episodes across `calibrated` and `neutral`, documented in [docs/RLVR_RESULTS.md](RLVR_RESULTS.md)):
  * **ID Calibration Without Width Inflation:** `log_score_r2` (`lr=5e-5, beta=0.01`, 150 steps) reached **100% valid (`120/120`)** on the 8 training families with **`−0.528` mean reward** (vs. `−4.558` base, `−1.738` `binary_r2`) and **exact `1.00` Tier-1 sharpness** (`line_fit=1.00`, `weighted_mean=1.00`, `counting_rate=1.03`).
  * **Binary RLVR Blindness & Gradient Starvation:** `binary_r2` hit `80%` zero-std batch starvation by step 150 (vs. `5%` for `log_score_r2`) and left `10×` unit bugs (`counting_rate` sharpness `10.04`) and ignored systematics (`voltmeter_gain` `−9.03` vs. `log_score_r2` `−2.28`).
  * **Partial OOD Transfer & Single-Turn Bottleneck:** `log_score_r2` beat `binary_r2` by `+11.8` pts coverage / `+2.04` reward on paired valid OOD (`n=34`) and lifted `dead_time` cov68 from `0.0% → 33.3%`, while single-turn `enable_thinking=False` schema errors (`53%` OOD valid) remain the main bottleneck for `v0.2` multi-turn RLVR.

### Remaining Before Tagging `v0.1`
1. **Commercial & Open Frontier Calibration Table (`pullgym run-fast`, ~\$8–\$15):** Run 3–5 multi-turn frontier models (`DeepSeek`, `Claude Sonnet`, `Gemini`, `GPT`) across `calibrated` and `neutral` (`12 families × 25 seeds = 300 episodes/variant`) to publish the headline frontier leaderboard alongside the RLVR results.

---

## 2. Two Upgrades That Add Genuine Technical & ML Novelty

### Milestone 1: Fast Zero-Docker Worker Mode (`pullgym run-fast`)
Keep `pullgym export` (Harbor separate-container mode) as the gold-standard untrusted leaderboard runner, and add a lightweight local/Colab runner (`src/pullgym/runner.py`):
* Materializes only `data/` and `apparatus_note.md` in an ephemeral read-only directory per episode; keeps `truth` and `ref_sigma` exclusively in the parent process memory (never written to disk before subprocess exit).
* Runs an OpenAI-compatible multi-turn tool loop (`bash` / `python3` with timeout and clean environment) against any endpoint (`vLLM`, DeepSeek, OpenAI, Anthropic) with configurable async concurrency.
* **Payoff:** Evaluates 300–600 episodes in minutes on a MacBook or Colab GPU and serves directly as the rollout environment for GRPO training.

### Milestone 2: Proper-Scoring-Rule RLVR vs. Binary-Reward Overconfidence
Almost all current Reinforcement Learning with Verifiable Rewards (RLVR) on math and code optimizes **binary `0/1` correctness rewards**, which rewards confident guessing and degrades epistemic calibration.

`pullgym` enables a controlled, zero-judge RL experiment testing whether **distributional RLVR** teaches calibrated scientific data analysis:
* **Setup:** Train an open tool-using model (e.g. **Qwen3-4B** with QLoRA via Unsloth + `TRL` `GRPOTrainer` on a single Colab L4/A100 GPU) on **8 training families**, holding out **4 unseen systematic/boundary families** (`dead_time`, `correlated_norm`, `errors_xy`, `no_signal`) strictly for out-of-distribution evaluation.
* **Controlled ablation (identical base model, tool loop, and episode budget):**
  1. **Point-Accuracy Binary GRPO:** Reward $R = \mathbb{I}(|q_{50} - \theta| \le k \cdot \sigma_{\text{ref}})$ (plus valid-ordering format check).
  2. **Proper Distributional GRPO:** Reward $R = \text{log\_score\_reward}(q, \theta, \sigma_{\text{ref}})$ from `src/pullgym/scoring.py#L66-L72`.
* **Headline Falsifiable Hypothesis:**
  * Binary RLVR improves central-value accuracy (`q50`) while **shrinking reported intervals or leaving systematic uncertainties ignored** (severe under-coverage on held-out systematics).
  * Log-Score GRPO induces emergent verification behaviors in generated Python code (explicitly reading `apparatus_note.md` certificates, propagating nuisance/calibration errors, running Monte Carlo / bootstrap checks, and reporting asymmetric intervals near physical boundaries), moving held-out 68%/95% coverage toward nominal without blowing up sharpness.

---

## 3. Low-Cost Evaluation & Compute Plan (~$8–$18 Total API Spend)

| Step | Status | Scope (`12 families × 25 seeds = 300 eps/variant`; `calibrated` + `neutral`) | Compute / API Strategy | Est. Spend |
| :--- | :--- | :--- | :--- | :--- |
| **1. `pullgym run-fast`** | **Done** | Lightweight isolated subprocess runner with exact Harbor parity & `--resume`. | Local MacBook | **$0** |
| **2. `Qwen3-8B` Base + Binary vs. Log-Score GRPO** | **Done** | Round 1 (`KL=0.011`) + Round 2 (`KL=0.076`, 150 steps) on 8 families; 1,600 episodes evaluated across ID & OOD (`docs/RLVR_RESULTS.md`). | RTX 4090 48 GB | **Done** |
| **3. Commercial Frontier Leaderboard** | **Next** | Run `deepseek-chat` / `deepseek-reasoner`, `claude-sonnet`, `gemini-flash/pro`, and `gpt` via `pullgym run-fast` (`--concurrency 6 --resume`). | DeepSeek API + OpenRouter / AI Studio | **~$8–$15** |
| **4. `v0.1` Release & Flagship Post** | **Next** | Merge frontier leaderboard into `README.md`, tag `v0.1` on GitHub/PyPI, and publish technical post. | Local | **$0** |
| **5. `v0.2` Multi-Turn RLVR & New Trap Families** | Planned | Multi-turn execution-feedback GRPO (fixing OOD schema validity) + `outliers`, `look_elsewhere`, and unit-mismatch families. | GPU VM / Colab | TBD |

---

## 4. Independent Positioning & Reward-Hacking Audit Checklist

### Framing & Prior-Art Guardrail (Critical for Paper / Blogpost)
* **Do NOT claim to be the first to put proper scoring rules into GRPO.** 2025–2026 papers (**RLCR**, **CAPO**, Brier-score GRPO) already optimize proper scoring rules inside GRPO for *scalar verbalized confidence* ("I am 80% confident my final math/QA token is right").
* **Cite RLCR / CAPO upfront and state the exact novelty boundary:** Prior calibration RL rescales a self-rated scalar probability on discrete correctness; `pullgym` studies **procedural uncertainty propagation in code-writing data-analysis agents**, testing whether distributional RLVR (`log_score_reward` over five continuous predictive quantiles) induces **code-level modeling changes** (parsing `apparatus_note.md` calibration certificates, modeling nuisance/dead-time systematics, using Monte Carlo / profile likelihoods, and respecting physical $r \ge 0$ boundaries).

### Three Technical Warnings to Audit on Every RL Run
1. **Never Raise `invalid_penalty` Above `floor` (`src/pullgym/rl.py#L318-L334`):**
   * In official leaderboard scoring (`scoring.py`), missing/malformed `answer.json` gets `prior_score - 1` (`~ -3`), which is higher than a valid `10×` overconfident systematic miss (`-12.6` to `-20`).
   * `rl.py` correctly overrides this during GRPO (`floor = -10.0`, `invalid_penalty = -12.0`) so crashing never beats submitting quantiles. Keep that strict inequality intact if you ever widen `floor` (e.g. `floor = -20.0` requires `invalid_penalty <= -21.0`).
   * **Reporting corollary:** Because `evaluate_completions_to_trials()` uses `scoring.score_episode()` (`prior - 1` on invalid), a model whose `valid%` rises from `56%` (`Qwen3-8B` base) to `95%` can see its raw `reward_mean` shift solely from fewer `-3` fallbacks or more `-12` attempts on hard tasks. Always report **`valid%`**, **valid-episode coverage/pull/sharpness**, and **all-episode reward** side by side.
2. **Audit Tier-1 Sharpness Against Lazy Global Width Inflation:**
   * Because `10×` overconfident intervals cost `~ -12` while `10×` conservative intervals cost only `~ -1.8`, GRPO can raise OOD coverage the "lazy way" by multiplying every `scipy` standard error by `5×–8×` globally without reading `apparatus_note.md`.
   * **Mandatory post-training check:** Compare **ID Tier-1 sharpness** (`line_fit`, `weighted_mean`, `counting_rate` — should stay near `~1.0`) against **OOD Tier-3/4 sharpness & coverage** (`dead_time`, `correlated_norm`, `errors_xy`, `no_signal`), and diff generated Python scripts on `dead_time` / `correlated_norm` to verify genuine systematic propagation vs. constant inflation. (Either outcome—true systematic synthesis or exposing lazy width inflation in 8B QLoRA—is an honest, publishable finding!)
3. **Headline Viral Hook First:**
   * Don't wait on RL ablations to showcase `pullgym`: running the **commercial frontier table** (`Claude Sonnet`, `DeepSeek`, `Gemini`, `GPT` on `neutral` vs. `calibrated` via `pullgym run-fast`, `~$10–$15`) is the highest-impact empirical hook and validates whether `100B+` frontier agents commit the exact same systematic shortcuts as `Qwen3-8B` (`33.9%` OOD 68%-coverage).
