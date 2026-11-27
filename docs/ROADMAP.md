# pullgym — Technical Audit, Bottlenecks & ML/RL Roadmap

## 1. Current Status Snapshot (`v0.1.0` Released → `v0.2.0` In Progress)

### Completed & Verified (`v0.1.0`)
* **Core Benchmark & Calibrated Reference Suite (`src/pullgym/`):** 12 experiment families across 4 tiers (`line_fit` through `dead_time`, `correlated_norm`, `no_signal`), five-quantile proper `log_score_reward`, verified on 2,000+ seeds (`pullgym calibrate`, `pullgym probe`), plus secret-seed held-out Harbor export (`pullgym export`).
* **Fast Zero-Docker Multi-Turn Runner (`pullgym run-fast`):** Ephemeral read-only workspaces keeping `truth` and `ref_sigma` strictly in parent memory, stdlib OpenAI-compatible multi-turn `run_python` / `run_bash` loop with `--resume`, `Retry-After` / `IncompleteRead` retries, turn-budget reminders, and parallel concurrency (`src/pullgym/runner.py`, exact Harbor parity verified in `tests/test_runner.py`).
* **Controlled Distributional RLVR vs. Binary GRPO (`Qwen3-8B`):** Two controlled 2-arm GRPO rounds (1,600 evaluated episodes across `calibrated` and `neutral`, documented in [docs/RLVR_RESULTS.md](RLVR_RESULTS.md)):
  * **ID Calibration Without Global Width Inflation:**
    * `log_score_r2` (`lr=5e-5, beta=0.01`, 150 steps) reached **100% valid (`120/120`)** on the 8 training families, with mean reward `−0.528` (vs. `−4.558` base, `−1.738` `binary_r2`).
    * Seed-paired against `binary_r2`: Δreward = **`+1.21` [`+0.53`, `+2.02`]** and Δpinball skill = **`+0.58` [`+0.24`, `+1.03`]**, with Δcov68 ≈ 0.
    * Tier-1 geometric-mean sharpness is `1.00` / `1.00` / `1.56` (`line_fit` / `weighted_mean` / `counting_rate`), against `7.32` on `counting_rate` for `binary_r2`.
  * **Binary RLVR Blindness & Gradient Starvation:** `binary_r2` hit `80%` zero-std batch starvation by step 150 (vs. `5%` for `log_score_r2`) and left `10×` unit bugs (`counting_rate` sharpness `10.04`) and ignored systematics (`voltmeter_gain` `−9.03` vs. `log_score_r2` `−2.28`).
  * **OOD Transfer Not Established:**
    * On 4 held-out families, `log_score_r2` − `binary_r2` gives Δreward = `+1.55` [`−0.33`, `+3.41`] (`n=100` paired).
    * On the 34 tasks both answered validly, Δcov68 = `+11.8 ± 10.8` pt (9/5 discordant, p = 0.42), and 18 of those 34 pairs are `dead_time`.
    * Single-turn runtime/boundary errors (`53%` OOD valid) and one training seed per arm motivate multi-turn RLVR with ≥3 seeds per arm in `v0.2.0`.
* **Commercial & Open Frontier Baselines (`pullgym run-fast`):** Evaluated `deepseek-flash` (360 eps across `calibrated` and `neutral`), `kimi-k3.1-pro`, `qwen3.8-omni-flash`, `qwen/qwen3.8-27b`, and `nemotron-3-ultra-550b` alongside seed-paired `oracle` and `prior`. **When a frontier agent produces a valid answer, it reproduces the reference analysis task by task.** Against the oracle, `deepseek-flash` has Δrew_v = `+0.01` [`−0.01`, `+0.03`] over 151 paired tasks, with 2 vs 0 discordant at 68%; `kimi-k3.1-pro` and `qwen3.8-27b` have no discordant tasks. v0.1's explicit systematics are saturated for frontier models, and their only remaining loss is answer validity (`deepseek-flash` Δrew_all = `−0.73` [`−1.01`, `−0.47`]).

---

## 2. Next Milestones (`v0.2.0`)

0. **Phase 1, harness and statistics: done in `0.2.0.dev0`.**
   * Single-source `__version__`, with `harness_version` in every `result.json`.
   * pandas and matplotlib are core dependencies.
   * `probe` scores the exact reference quantiles, with shape-preserving width probes and a `gaussian approx` diagnostic.
   * The runner reports `turn k/N` and a schema-only `answer_status` after every tool call, warns 3 turns before the end and on the final turn, and no longer ends an episode on an invalid `answer.json`.
   * With several quantities, `reward.json` averages the calibration keys.
   * `pullgym report` has `rew_v`/`rew_all`, `pt_acc`, Wilson CIs, geometric-mean sharpness, clipped pinball skill, `--by tier`, `--paired`, `--baseline` (paired bootstrap and McNemar), a harness-version warning, `--format md|latex|json`, and accepts several paths.
1. **Tier-5 Implicit Metrology & Unprompted Diagnostic Families: done in `0.2.0.dev0`.** Expanded from 12 to 16 families:
   * Unprompted residual diagnostics: `outliers` and `unmodeled_drift`. Their apparatus notes give no hint of the problem, and a test enforces this.
   * A trials-factor search: `look_elsewhere`.
   * BIPM/GUM datasheet and cross-unit conventions: `datasheet_metrology`.

   The checks:
   * Every reference is exact, deterministic and under 25 ms per episode.
   * All four pass `calibrate --n 2000` (cov68 66.1–68.0%, cov95 94.0–95.2%) and `probe --n 1000`.
   * The main shortcut fails, and an analysis that does not know the hidden hyperparameters stays close to the reference (`tests/test_tier5.py`).

   The new families are held out as `OOD_V02_FAMILIES`, and `TRAIN_FAMILIES` is unchanged.
1b. **Phase 2, zero-retraining multi-turn diagnostic of the R2 checkpoints: done** ([RLVR_RESULTS §6](RLVR_RESULTS.md)).
   * **Evaluation noise:** between two draws at temperature 0.7, 15–37% of OOD tasks flip validity. With two draws per task, `log_score_r2` beats `binary_r2` OOD (+2.02 [+0.49, +3.53]) but ties `base`, while `binary_r2` falls below `base`.
   * **Repair:** 3 repair turns raise OOD validity by 19–24 pt but lower `rew_all`, because the repaired answers are confidently wrong.
   * **Harness transfer:** preview-trained adapters collapse under the zero-preview `run-fast` harness (ID validity 120 → 55).
   * **Tier 5:** unsolved by all 8B arms (cov68 13–47%).
   * **Consequences for Phase 5:** train in the `run-fast` format, with ≥4 evaluation draws per task and ≥3 training seeds per arm.
2. **Uniform Frontier Leaderboard & Harbor Registry PR:** Evaluate cross-lab frontier models (`Claude`, `GPT`, `Gemini`, `DeepSeek` chat vs. reasoner, `Qwen3-Max`) at uniform `n = 25` seeds/family with paired CIs and `pinball_skill`, and submit `pullgym` to Harbor's community benchmark registry.
3. **Multi-Turn Execution-Feedback GRPO:** Extend controlled `base` vs. `binary` vs. `log_score` GRPO to multi-turn `run_python` rollouts (`Dr. GRPO` with turn-level `action_mask`) to eliminate OOD runtime traceback fragility on 8B open models.

---

## 3. Positioning & Reward-Hacking Audit Checklist

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
