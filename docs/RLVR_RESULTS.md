# pullgym — Distributional RLVR (`log_score`) vs. Binary GRPO (`binary`) Results

## Positioning & Scope

Recent calibration-aware RL papers (**RLCR**, **CAPO**, Brier-score GRPO) optimize proper scoring rules inside GRPO to calibrate a model's *scalar verbalized confidence* (*"I am 80% sure my final answer is right"*) on discrete QA or math correctness. `pullgym` studies a distinct setting: **procedural uncertainty propagation in code-writing scientific data-analysis agents**, testing whether proper distributional RLVR (`log_score_reward` over five continuous predictive quantiles `q025, q16, q50, q84, q975`) induces **code-level modeling changes** (reading calibration certificates from `apparatus_note.md`, propagating systematics/nuisance parameters, and respecting physical boundaries) rather than overconfident point estimation or lazy width inflation.

> **Scope note:** All runs below train and evaluate **single-turn Python code synthesis** (`scripts/train_grpo.py`, `src/pullgym/rl.py`) from full `apparatus_note.md` + CSV previews, with isolated subprocess execution and zero LLM judge. Transfer to multi-turn interactive `pullgym run-fast` / Harbor agents is tracked separately.

---

## Round 1 Experimental Setup (RTX 4090 48 GB)

Controlled two-arm GRPO ablation from the exact same base model, dataset split, rollout budget, and evaluation sampling parameters:

- **Base model:** `unsloth/Qwen3-8B-unsloth-bnb-4bit` (4-bit QLoRA, `r=32`, `alpha=32`, 87.3M trainable params, `enable_thinking=False`)
  - *Why `enable_thinking=False`?* A 20-episode OOD probe of `Qwen3-8B` with `enable_thinking=True` and `max_tokens=3072` hit `20/20` truncations (`0%` valid) before emitting a complete Python script.
- **Training split (8 families × 250 seeds = 2,000 episodes, `variant="calibrated"`):** `line_fit`, `weighted_mean`, `counting_rate`, `decay_background`, `peak_background`, `low_count_rate`, `voltmeter_gain`, `underestimated_errors`.
- **Held-out OOD test split (4 unseen systematic/boundary families × 25 seeds = 100 episodes):** `errors_xy`, `correlated_norm`, `dead_time`, `no_signal` (+ 120 ID eval episodes across the 8 training families, evaluated on both `calibrated` and `neutral` OOD prompts — **960 total evaluated episodes**).
- **Shared GRPO hyperparameters:** `steps=150`, `num_generations=8`, `grad_accum=4` (effective batch size 32; 4,800 total rollouts per arm), `lr=1e-5` (cosine), `beta=0.04`, `max_grad_norm=0.2`, `max_completion_len=1536`, rollout sampling `temp=0.85, top_p=0.95, top_k=20, presence_penalty=1.0`; eval sampling `temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.0, seed=42`.
- **Reward arms:**
  - **Arm A (`binary`):** `+1.0` if valid and `|q50 − truth| ≤ 1.0 × ref_sigma`, `0.0` if valid miss, `−1.0` if invalid/crash.
  - **Arm B (`log_score`):** `distributional_log_score_reward` with `floor = −10.0`, `cap = +5.0`, `invalid_penalty = −12.0` (strictly maintaining `invalid_penalty < floor` so crashing never beats reporting a wide interval).

---

## Round 1 Headline Results

> **Valid-rate vs. fallback audit note (`docs/ROADMAP.md` §4):** Official `pullgym report` scores missing/invalid answers as `prior − 1` (`~ −3`), which is milder than a `−20` overconfident systematic miss. Every table below therefore reports **`valid%`**, **valid-only point accuracy (`|q50 − θ| ≤ σ_ref`)**, **valid-only coverage / pull width / sharpness / mean reward (`rew_v`)**, and **all-episode mean reward (`rew_all`)** side by side, followed by the **paired valid-on-all-3** intersection to eliminate selection bias.

### 1. In-Distribution (8 Train Families, `calibrated`, `n = 120`)

| Model | Valid | Pt Acc (valid) | Pt Acc (all) | 68% Cov *(68%)* | 95% Cov *(95%)* | Pull Width *(1.0)* | Sharpness *(1.0)* | Reward (valid) | Reward (all) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3-8b-base` | 58.3% | 45.7% | 26.7% | 41.4% | 61.4% | 2.07 | 1.00 | −4.669 | −4.558 |
| `qwen3-8b_binary_full` | **74.2%** | 47.2% | 35.0% | 47.2% | 59.6% | 2.87 | 1.00 | −5.395 | −5.160 |
| `qwen3-8b_log_score_full` | 67.5% | **55.6%** | **37.5%** | **49.4%** | **63.0%** | **2.22** | 1.00 | **−4.382** | **−4.304** |

### 2. ID Breakdown by Tier & Lazy Global Width-Inflation Audit

| Slice | Model | Valid | Pt Acc (valid) | 68% Cov | 95% Cov | Pull Width | Sharpness | Reward (valid) | Reward (all) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Tier-1 Clean** (`line_fit`, `weighted_mean`, `counting_rate`, `n=45`) | `base` | 77.8% | 51.4% | 54.3% | 71.4% | 1.66 | **1.00** | −3.796 | −4.035 |
| | `binary` | **95.6%** | **62.8%** | **67.4%** | **79.1%** | **0.97** | **1.00** | **−2.384** | **−2.460** |
| | `log_score` | 86.7% | 59.0% | 56.4% | 71.8% | 1.65 | **1.00** | −3.738 | −3.910 |
| **Tier-2 Nuisance** (`decay_bg`, `peak_bg`, `low_count_rate`, `n=45`) | `base` | 31.1% | 21.4% | 28.6% | 50.0% | 2.13 | 1.22 | −8.153 | −5.286 |
| | `binary` | **53.3%** | 25.0% | 29.2% | 41.7% | 3.87 | 1.29 | −9.490 | −7.073 |
| | `log_score` | 46.7% | **42.9%** | **47.6%** | **52.4%** | 3.68 | 1.15 | **−7.170** | **−5.208** |
| **Tier-3 ID Systematics** (`voltmeter_gain`, `underestimated_errors`, `n=30`) | `base` | 70.0% | 52.4% | 28.6% | 52.4% | 2.89 | 0.50 | −3.803 | −4.249 |
| | `binary` | **73.3%** | 40.9% | 27.3% | 40.9% | 3.98 | 0.37 | −6.813 | −6.339 |
| | `log_score` | 70.0% | **61.9%** | **38.1%** | **57.1%** | **2.04** | **0.53** | **−2.789** | **−3.537** |

Notable ID family wins for `log_score` over `binary`:
- **`decay_background`:** `log_score` reached **75.0% cov68 / 75.0% cov95**, pull width **0.98**, sharpness **0.96**, valid reward **−2.930** (vs. `binary`: 27.3% cov68, pull width 3.25, valid reward −9.423).
- **`underestimated_errors`:** `log_score` improved `cov68` **14.3% → 37.5%**, `cov95` **42.9% → 62.5%**, sharpness **0.36 → 0.81**, valid reward **−3.879 → −1.269** (vs. `binary`: 14.3% cov68 / 14.3% cov95, sharpness 0.36, valid reward **−7.450**).

### 3. Held-Out Out-of-Distribution (4 Unseen Families, `n = 100` per variant)

| Variant | Model | Valid | Pt Acc (valid) | Pt Acc (all) | 68% Cov *(68%)* | 95% Cov *(95%)* | Pull Width *(1.0)* | Sharpness *(1.0)* | Reward (valid) | Reward (all) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `calibrated` | `qwen3-8b-base` | 56.0% | **33.9%** | **19.0%** | **33.9%** | **44.6%** | 14.76 | 1.41 | **−9.286** | **−6.518** |
| `calibrated` | `qwen3-8b_binary_full` | **68.0%** | 16.2% | 11.0% | 19.1% | 29.4% | 17.81 | 0.74 | −12.491 | −9.473 |
| `calibrated` | `qwen3-8b_log_score_full` | 55.0% | 21.8% | 12.0% | 30.9% | 38.2% | **7.91** | 1.15 | −10.150 | −7.047 |
| `neutral` | `qwen3-8b-base` | 53.0% | 15.1% | 8.0% | **37.7%** | **45.3%** | **8.44** | 1.46 | **−9.533** | **−6.463** |
| `neutral` | `qwen3-8b_binary_full` | 57.0% | 22.8% | 13.0% | 22.8% | 33.3% | 126.46 | 0.94 | −11.144 | −7.650 |
| `neutral` | `qwen3-8b_log_score_full` | **62.0%** | **25.8%** | **16.0%** | 30.6% | 38.7% | 15.04 | 1.05 | −10.080 | −7.387 |

### 4. Paired Valid-on-All-3 Subset (Zero Selection-Bias / Fallback Confounding)

| Split | Model | `n` | Pt Acc | 68% Cov | 95% Cov | Pull Width | Sharpness | Mean Reward |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All (`calibrated`) | `qwen3-8b-base` | 81 | **49.4%** | **44.4%** | **63.0%** | **2.16** | 1.00 | **−4.225** |
| All (`calibrated`) | `qwen3-8b_binary_full` | 81 | 39.5% | 40.7% | 54.3% | 2.99 | 1.00 | −6.273 |
| All (`calibrated`) | `qwen3-8b_log_score_full` | 81 | 44.4% | 43.2% | 55.6% | 2.69 | 1.00 | −5.074 |
| OOD (`calibrated`) | `qwen3-8b-base` | 29 | **44.8%** | **44.8%** | **62.1%** | **2.73** | 1.63 | **−6.150** |
| OOD (`calibrated`) | `qwen3-8b_binary_full` | 29 | 20.7% | 27.6% | 41.4% | 6.45 | 1.01 | −10.408 |
| OOD (`calibrated`) | `qwen3-8b_log_score_full` | 29 | 31.0% | 34.5% | 44.8% | 6.53 | 1.21 | −8.029 |

![OOD Calibration Comparison (Round 1)](assets/ood_calibration_calibrated.png)

---

## Mandatory Audits & Honest Takeaways

### 1. Binary RLVR Severe Overconfidence Collapse Confirmed
Across both `calibrated` and `neutral` prompts, optimizing binary point-accuracy rewards (`binary`) reliably increased code execution validity (`57.3% → 71.4%` overall), **while severely harming epistemic calibration**:
- On **ID Tier-3 systematics**, `binary` shrank median sharpness (`0.50 → 0.37`), worsened pull width (`2.89 → 3.98`), and dropped valid-only reward by `−3.01` (`−3.803 → −6.813`).
- On **OOD (`calibrated`)**, `binary` shrank median sharpness nearly 2× (`1.41 → 0.74`), collapsed `cov68` by **−14.8 points** (`33.9% → 19.1%`), and dropped valid-only reward from `−9.286 → −12.491`.
- On **OOD (`neutral`)**, the exact same collapse replicated (`cov68`: `37.7% → 22.8%`, pull width `8.44 → 126.46`).

### 2. Lazy Global Width-Inflation Audit: Passed (No Constant Multiplier Hack)
- On Tier-1 clean families, `log_score` preserved exact median sharpness **`1.00`** on both `line_fit` (`1.00`) and `weighted_mean` (`1.00`), ruling out lazy global error-bar inflation.
- On `counting_rate`, both `binary` (`sharpness=8.69`) and `log_score` (`sharpness=19.97`) shifted away from a buggy Gamma-prior script in `base` (which predicted `4.26e6` instead of `26.2`) to an identical simpler Poisson script (`std_dev = np.sqrt(mean_rate)` missing `/ sqrt(total_time)`). Because *both* arms independently adopted this simpler non-crashing template, it reflects crash avoidance rather than log-score width hacking.

### 3. Why OOD Did Not Yet Surpass Base in Round 1 (Diagnosis for Round 2)
Code-level inspection and training telemetry revealed two concrete bottlenecks in Round 1:
1. **Tiny policy shift (`KL ≈ 0.011`):** With `lr=1e-5`, `beta=0.04`, and `max_grad_norm=0.2` over 150 steps, `KL` plateaued at `0.011` (`log_score`) / `0.012` (`binary`) and `log_score` batch valid rate rose only modestly (`66.6% → 72.2%`). On several seeds (e.g., `underestimated_errors-dev-00000`), all three models generated character-for-character identical scripts.
2. **Physics-formula & implementation bugs under single-turn `enable_thinking=False`:** On unseen `dead_time`, all three models wrote wrong dead-time formulas (`true_rate * 10 * (1 - dead_time/10)` instead of $R_{\text{obs}} = R_{\text{true}} / (1 + R_{\text{true}}\tau)$), yielding `0.0%` coverage on `calibrated`. On `correlated_norm-dev-00000`, `log_score` actually attempted a daily log-normal Monte Carlo calibration loop (`np.random.lognormal(0, 0.005)`) rather than `base`/`binary`'s buggy `density * avg_density` (`850,000`), but hit a 1D broadcast shape bug and crashed.

### Next Step: Round 2 Stronger Policy Shift (`lr=5e-5`, `beta=0.01`, `max_grad_norm=1.0`)
To test whether pushing GRPO out of the near-frozen `KL ≈ 0.011` regime into genuine policy adaptation (`KL ≈ 0.05–0.15`) unlocks true OOD systematic generalization—or triggers lazy Tier-1 width inflation in 8B QLoRA—Round 2 runs both arms with `lr=5e-5, beta=0.01, max_grad_norm=1.0` over 150 steps on the RTX 4090.
