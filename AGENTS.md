# AGENTS.md — Engineering & Scientific Guide for `pullgym`

`pullgym` is an evaluation benchmark and Reinforcement Learning with Verifiable Rewards (RLVR) environment testing whether code-writing AI agents report calibrated uncertainty intervals (`q025, q16, q50, q84, q975`) on simulated physics experiments, scored against exact ground truth via a strictly proper clipped log-density rule with **zero LLM judge**.

> **Local roadmap check:** Before starting multi-step feature work, check whether local gitignored planning files (such as `plan-v0.2.0.md`) exist in the repository root and follow their phase specifications. Never force-add (`git add -f`) `plan-*.md`, `paper*/`, `*.secret`, `runs/`, `tasks/`, `jobs/`, or `.DS_Store`.

---

## 1. Repository Architecture

- `src/pullgym/scoring.py` — **100% Python stdlib-only** answer parser and scorer (`load_answer`, `log_density`, `log_score_reward`, `pinball`, `score_episode`). Copied verbatim into standalone Harbor verifier containers by `export.py`.
- `src/pullgym/families/` — Procedural experiment generators (`base.py` `Family` ABC + concrete family modules registered in `families/__init__.py`).
- `src/pullgym/export.py` — Harbor `separate`-verifier exporter (`task.toml`, `instruction.md`, agent/verifier `Dockerfile`s, SHA-256 `heldout_seeds`).
- `src/pullgym/runner.py` — Zero-Docker local/API multi-turn runner (`pullgym run-fast`), keeping `truth` and `ref_sigma` strictly in parent memory while mounting `data/` read-only and scrubbing secret env vars from subprocesses.
- `src/pullgym/stats.py` & `src/pullgym/report.py` — Summary statistics, Wilson/SE intervals, degenerate-strategy `probe()`, and Harbor/`run-fast` leaderboard reporting (`pullgym report`).
- `src/pullgym/rl.py` & `scripts/train_grpo.py` — RLVR prompt/dataset builders (`TRAIN_FAMILIES` vs. held-out `OOD_*_FAMILIES`), isolated Python execution sandbox, and controlled two-arm GRPO (`binary` vs. `log_score` via Unsloth + TRL + vLLM).
- `src/pullgym/cli.py` — CLI entrypoint (`pullgym list | export | calibrate | probe | report | run-fast`).

---

## 2. Non-Negotiable Invariants (Must Never Break)

### 2.1 Verifier Isolation & Backwards Compatibility
1. **`scoring.py` is strictly stdlib-only:** Never import `numpy`, `scipy`, `pandas`, or any `pullgym.*` module inside `src/pullgym/scoring.py`.
2. **Invalid-answer contract:** When `valid` is `False`, `score_episode()` scores `fallback` minus `INVALID_PENALTY` (`1.0`) and **omits** calibration keys (`covered68`, `covered95`, `pull`, `sharpness`, `pinball_skill`) from `reward.json` (required by Harbor & `tests/test_report.py`).
3. **Zero truth leakage:** `truth`, `ref_sigma`, `truth_doc`, and held-out seeds must **never** appear in `data/`, `apparatus_note.md`, workspace files, subprocess environment variables, `test`-split `task.toml`, or live tool observations.
4. **Schema-only live feedback:** Mid-episode `answer_status` checks in `runner.py` or `rl.py` may call `scoring.load_answer(...)` (public JSON syntax, required keys, finite floats `< 1e30`, strict inequality `q025 < q16 < q50 < q84 < q975`) — **never** `score_episode()` or `truth`.
5. **Preserve `stats.summarize()` keys:** Keep `cov68_se` and `cov95_se` keys intact alongside any new CI keys (`cli.py` depends on them), and guard optional fields with `.get(...)` in `report.py` for minimal test fixtures.

### 2.2 Disjoint Seed Partition
Never overlap seed ranges across splits, calibration checks, and RL training:
- `0 .. 9_999`: public `dev` evaluation split (`--split dev`, default `--start 0`)
- `10_000 .. 10_399` & `20_000 .. 20_199`: `pytest -m slow` calibration & probe unit tests (`tests/test_families.py`)
- `100_000+`: RLVR training episodes (`DEFAULT_TRAIN_SEED_START` in `src/pullgym/rl.py`)
- `1_000_000+`: `pullgym calibrate` CLI default
- `2_000_000+`: `pullgym probe` CLI default
- `SHA-256(secret:family:i)`: opaque held-out `test` split (`--split test --secret-file ...`)

### 2.3 Experiment Family Metrological Standards
Every `Family` subclass in `src/pullgym/families/` must satisfy:
1. **Deterministic & fast `reference_analysis(files)` (`< 30 ms`/episode):** Use exact conjugate math or deterministic quadrature/`grid_quantiles` — **no unseeded Monte Carlo**. Watch tensor shapes (avoid multi-GB 4D grids; use sufficient statistics / local stencils).
2. **Exact Bayesian/pivotal calibration:** Must pass `uv run pytest -m slow`, `uv run pullgym calibrate --n 2000` (`cov68 ∈ [66%, 70%]`, `cov95 ∈ [94%, 96%]`, `pull_width_robust ∈ [0.95, 1.05]`), and outscore all degenerate strategies in `uv run pullgym probe --n 1000`.
3. **No unprompted spoilers & no secret hyperpriors:**
   - Unprompted diagnostic families (`outliers`, `unmodeled_drift`, etc.) must **not** hint at glitches or drift in `apparatus_note.md`.
   - `reference_analysis()` must never hardcode secret nuisance parameters withheld from `apparatus_note.md`; model-agnostic analyses must achieve near-reference coverage and reward.
4. **Single-quantity i.i.d. benchmark structure:** Keep 1 primary target quantity per main benchmark family so episodes are independent and equally weighted in aggregate coverage/CIs (test multi-quantity support via synthetic fixtures in `tests/`).

### 2.4 RLVR & Evaluation Rigor
1. **Strict crash penalty inequality:** In `rl.distributional_log_score_reward`, always enforce `invalid_penalty < floor` (default `floor = -10.0, invalid_penalty = -12.0`) so crashing never beats reporting a wide valid interval.
2. **Strict OOD family holdout:** Keep `TRAIN_FAMILIES` (the 8 training families) untouched and strictly disjoint from `OOD_TEST_FAMILIES` (`OOD_V01_FAMILIES` + `OOD_V02_FAMILIES`).
3. **Controlled multi-turn GRPO:**
   - Build `(completion_ids, action_mask)` turn-by-turn (`action_mask = 1` strictly on assistant tokens, `0` on environment observation/header tokens — never naively re-tokenize concatenated strings without verifying token-boundary invariance).
   - Truncate per-turn `STDOUT`/`STDERR` in RL rollouts to `~1,200` chars (head `400` + tail `800` preserving Python exception lines) and use `loss_type="dr_grpo"` with terminal-only episode rewards across `base`, `binary`, and `log_score`.
   - For `Qwen3` models, avoid greedy `temperature=0.0` (causes comment loops; use `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.0` with `enable_thinking=False` unless context budget is explicitly scaled). For `deepseek-reasoner`, pass `--max-tokens 32768` so CoT does not truncate before `tool_calls`.
4. **Anti-hacking & confounding audits:** Always report `valid%`, valid-only metrics (`rew_v`, `cov68`, `cov95`, `sharpness`, `pinball_skill`), all-episode reward (`rew_all`), **ID Tier-1 sharpness** (`line_fit`, `weighted_mean`, `counting_rate ≈ 1.00` to rule out global width inflation), and seed-paired comparisons (`--paired`), and never mix trials from different `harness_version`s in one leaderboard table.
5. **Double-blind safety:** Never commit author names, institutional emails, or anonymous-mirror URLs into tracked source/doc files.

---

## 3. Standard Development & Verification Commands

```bash
# Install / sync environment
uv sync

# Fast unit test suite (< 15s — run after every edit)
uv run pytest -q

# Slow statistical calibration & probe suite (mandatory after touching families/, scoring.py, or stats.py)
uv run pytest -m slow

# High-precision reference calibration & degenerate-strategy probe
uv run pullgym calibrate --n 2000
uv run pullgym probe --n 1000

# Local zero-Docker runner smoke check
uv run pullgym run-fast --family line_fit,voltmeter_gain --n 2 --agent oracle --out /tmp/pullgym_smoke
```
