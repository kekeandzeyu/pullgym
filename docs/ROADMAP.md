# pullgym — Technical Audit, Bottlenecks & ML/RL Roadmap

## 1. Technical Audit: Idea Novelty vs. Technical/ML Novelty

### What Already Works Well (~80% Complete)
* **Gap Filled:** Existing benchmarks test task-level confidence, zero-data priors (`OpenEstimate`), static stats QA (`StatABench`), or law/code reproduction (`PhysGym`, `Collider-Bench`). None score **empirical interval coverage (68% / 95%) and pull calibration** of tool-using data-analysis agents under realistic systematic uncertainties—computed purely from known ground truth $\theta$ with **zero LLM-as-a-judge**.
* **Rigorous Verification & Scoring (`src/pullgym/scoring.py`):**
  * Five-quantile reporting (`q025, q16, q50, q84, q975`) interpolated piecewise-linearly in Gaussian $z$-space (`log_density`, `src/pullgym/scoring.py#L53-L63`).
  * Strictly proper clipped log-density reward (`log_score_reward`) that averages `~0` for the calibrated human reference analysis and penalizes overconfident error bars quadratically (`~ -pull²/2`, down to `-20`).
  * All 12 families across 4 tiers (`line_fit` through `dead_time`, `correlated_norm`, and `no_signal`) are verified calibrated on 2,000+ seeds and beat degenerate width/shift shortcuts (`pullgym calibrate`, `pullgym probe`).
  * Contamination-resistant secret-seed held-out splits and isolated Harbor verification (`src/pullgym/export.py`).

### Current Technical Limitations & Bottlenecks
1. **Classical Internals Only (No Model Results or ML/RL Loop Yet):**
   Right now `families/*.py` and `scoring.py` combine clean `numpy`/`scipy` synthetic generators with classical proper scoring rules (Gneiting & Raftery, 2007). Without frontier model baselines and a training experiment, the repository reads as a well-crafted static test suite rather than a new AI/RL result.
2. **Per-Episode Dual-Container Overhead (`src/pullgym/export.py#L146-L185`):**
   `export_task()` generates **two Dockerfiles per episode** (`environment/Dockerfile` + `tests/Dockerfile`). Running `12 families × 50 seeds × 5 models = 3,000 episodes` via per-episode container builds on a laptop or Colab instance introduces heavy I/O and startup overhead when episodes only need an isolated Python worker.

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

| Step | Scope (e.g. 12 families × 25 seeds = 300 eps/model; `neutral` vs. `calibrated` prompt ablation) | Compute / API Strategy | Est. Spend |
| :--- | :--- | :--- | :--- |
| **1. `pullgym run-fast`** | Build lightweight isolated subprocess runner; verify exact parity with Harbor oracle on dev seeds. | Local MacBook | **$0** |
| **2. Open-Weight Baselines** | Evaluate `Qwen3-8B` & `Qwen3-32B-AWQ` (or `DeepSeek-R1-Distill`) served locally on Colab `vLLM`. | Colab Pro L4/A100 | **$0** |
| **3. Commercial Frontier Table** | Evaluate `deepseek-flash` & `deepseek-v4-pro` (off-peak 50% discount, cached prefixes) + 1 non-DeepSeek closed baseline (e.g. Claude Haiku / GPT-mini, plus optional 120-episode subset on Sonnet). | Commercial APIs | **~$8–$18** |
| **4. Binary vs. Log-Score GRPO** | Train Qwen3-4B on 8 families (`run-fast` rollouts) under both rewards; report OOD pull width, 68%/95% coverage, and sharpness on the 4 held-out families. | Colab Pro L4/A100 | **$0** |
| **5. v0.1 Release & Writeup** | Publish model leaderboard, `neutral` vs. `calibrated` prompt sensitivity, shortcut-trap breakdown, and the Binary-vs-Proper RLVR calibration curves. | Local | **$0** |
