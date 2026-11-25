# pullgym — Step-by-Step Colab Pro Training & Evaluation Guide

This checklist walks you through running the full **Proper-Scoring-Rule RLVR vs. Binary GRPO** experiment on **Google Colab Pro (L4 or A100 GPU)** using your Google AI Pro plan at **$0 extra compute cost**.

---

## The Big Picture (What You Are Testing)

Right now, your code lives locally on your Mac. You will use a Colab Pro NVIDIA GPU to run a controlled experiment with **`Qwen3-4B`**:

1. **Base model (`qwen3-4b-base`):** Measure how honest its error bars are *before* training.
2. **Arm A — Binary RL (`ARM = "binary"`):** Standard RLVR baseline. Rewards getting the central estimate `q50` close to truth, ignoring error bar width.
3. **Arm B — Log-Score RL (`ARM = "log_score"`):** Your method. Uses `pullgym`'s strictly proper `log_score_reward` so overconfident error bars are heavily penalized.
4. **Compare all three** on **4 hard physics experiments never seen during training** (`errors_xy`, `correlated_norm`, `dead_time`, `no_signal`).

Everything is wired into one notebook: `notebooks/train_grpo_colab.ipynb`.

---

## Step 1: Move Your Code from Your Mac to Google Drive

Colab runs on Google's servers and cannot read files directly from your Mac.

### Option A: Google Drive Upload (Easiest — No GitHub Fork Needed)
- [ ] Open [Google Drive](https://drive.google.com) signed into your **Google AI Pro** account.
- [ ] In **My Drive**, create a new folder named `pullgym`.
- [ ] On your computer, open your local `pullgym` directory and drag **only** these 4 items into `My Drive/pullgym/` *(do **not** upload `.venv` or `.git`)*:
  - `pyproject.toml`
  - `README.md`
  - `LICENSE`
  - `src/`
  - `tests/`
  - `notebooks/`
- [ ] Confirm your Google Drive contains:
  - `My Drive/pullgym/pyproject.toml`
  - `My Drive/pullgym/src/pullgym/rl.py`
  - `My Drive/pullgym/notebooks/train_grpo_colab.ipynb`

### Option B: GitHub Push (If You Own/Forked the Repo)
- [ ] Commit and push your local changes (`src/pullgym/runner.py`, `src/pullgym/rl.py`, `src/pullgym/cli.py`, `src/pullgym/export.py`, `tests/`, `notebooks/`).
- [ ] Note your repo URL (you will leave `USE_DRIVE_REPO = False` in Cell 2 and set `REPO_URL` to your repo).

---

## Step 2: Open the Notebook in Colab & Enable the GPU

- [ ] Open `My Drive/pullgym/notebooks/train_grpo_colab.ipynb` in Google Colab (double-click it in Drive, or go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → Google Drive**).
- [ ] In the Colab top menu, click **Runtime → Change runtime type**.
- [ ] Under **Hardware accelerator**, select **L4 GPU** (~5 compute units/hr — best for debugging and saving units), then click **Save**.
- [ ] Click **Connect** (top right) if it does not connect automatically.

---

## Step 3: Run the 10-Minute Sanity Check (Do This First!)

Before spending hours of GPU time, run a quick **20-step check** to verify that installation, data generation, code execution, and rewards all work cleanly.

- [ ] **Edit Cell 2:** If you used Google Drive (Option A), set:
  ```python
  USE_DRIVE_REPO = True
  ```
  *(If you used GitHub Option B, keep `USE_DRIVE_REPO = False` and update `REPO_URL`.)*
- [ ] **Check Cell 3:** Verify these two lines are set for the sanity check:
  ```python
  ARM = "log_score"
  SANITY_CHECK = True
  ```
- [ ] Run **Cell 1** (mount Google Drive → click **Allow**). Verify it prints your GPU (`NVIDIA L4 (24.0 GB VRAM)`).
- [ ] Run **Cell 2** (installs `pullgym`, `unsloth`, `vllm`, `trl`, and runs unit tests; takes ~3–5 min). Verify the last line prints `Environment ready!`.
- [ ] Run **Cell 3** (prints config: `Arm=log_score | Sanity=True | Steps=20`).
- [ ] Run **Cell 4** (builds train/eval datasets in ~10 seconds).
- [ ] Run **Cell 5** (loads 4-bit `Qwen3-4B` with Unsloth + vLLM; takes ~1–2 min).
- [ ] Run **Cell 6** (evaluates untrained **Base `Qwen3-4B`** on OOD and ID splits and caches results on Google Drive at `My Drive/pullgym_rlvr/eval_runs/`).
- [ ] Run **Cell 7** (20-step GRPO sanity run, ~5–10 min). Check the printed batch logs:
  ```text
  [batch    1 | voltmeter_gain      ] valid=83% mean=-3.120 std=4.210 min=-12.00 max=+0.45
  ```
  - [ ] Confirm `valid` is roughly **50%–100%** (model's Python code executes and writes valid JSON).
  - [ ] Confirm `std > 0.000` (different rollouts get different rewards so GRPO has real learning signal).
- [ ] Run **Cell 8** (evaluates the 20-step sanity checkpoint and prints the first comparison table).

---

## Step 4: Full Training — Arm B (`log_score`, Your Method)

Once Step 3 passes cleanly, run the full 350-step training for **Arm B**:

- [ ] *(Optional speed choice)* Stay on **L4 GPU** (~3–4 hrs, ~18 compute units) **or** switch to **A100 GPU** (~1.5–2 hrs, ~22 compute units). If you switch GPU types, re-run Cells 1–5 first.
- [ ] **Edit Cell 3:**
  ```python
  ARM = "log_score"
  SANITY_CHECK = False
  ```
- [ ] Run **Cell 3 → Cell 4 → Cell 5 → Cell 7 → Cell 8**.
  *(Cell 6 is automatically skipped if you run it, because the base model eval is already cached on Google Drive.)*
- [ ] Confirm Cell 8 finishes and prints results for `qwen3-4b-log_score_full`.

> **If Colab disconnects mid-training:**
> Checkpoints auto-save to `My Drive/pullgym_rlvr/checkpoints/log_score_full/` every 50 steps. Just reconnect, re-run **Cells 1–5**, and re-run **Cell 7** — it will print `resume_from_checkpoint=True` and continue from the latest checkpoint.

---

## Step 5: Full Training — Arm A (`binary`, Controlled Baseline)

Now train a second, clean copy of `Qwen3-4B` with standard point-accuracy binary RLVR so you have a fair apples-to-apples comparison:

- [ ] Click **Runtime → Restart session** *(important: guarantees you start from fresh base weights, not on top of Arm B)*.
- [ ] **Edit Cell 3:**
  ```python
  ARM = "binary"
  SANITY_CHECK = False
  ```
- [ ] Run **Cells 1, 3, 4, 5, 7, 8** *(run Cell 2 first if the session reset cleared installed pip packages)*.
- [ ] Confirm Cell 8 finishes and prints results for all three models side-by-side:
  - `qwen3-4b-base`
  - `qwen3-4b-binary_full`
  - `qwen3-4b-log_score_full`

---

## Step 6: Inspect Calibration Charts & Generated Code (Cell 9)

- [ ] Run **Cell 9**.
- [ ] Check the saved 3-panel figure at `My Drive/pullgym_rlvr/ood_calibration_calibrated.png`:
  - **OOD Empirical Coverage (`cov68`, `cov95`):** Closer to the dashed **68%** and **95%** target lines is better.
  - **OOD Robust Pull Width:** Closer to **`1.0`** is calibrated. Values much larger than `1.0` (e.g., `2.5`–`10.0`) mean severe overconfidence (error bars too small).
  - **OOD Mean Log-Score Reward:** Closer to **`0.0`** (human reference analyst) is better.
- [ ] Check `sharpness` in the Cell 8 table: confirm Arm B improved coverage **without** simply reporting absurdly wide error bars (`sharpness` near `1.0`–`1.5`, not `10+`).
- [ ] Read the printed side-by-side Python scripts for `dead_time-dev-00000` (a held-out OOD family) at the bottom of Cell 9: check whether Arm B explicitly reads and propagates the dead-time uncertainty from `apparatus_note.md`, whereas Base or Arm A ignores it.

---

## Step 7 (Optional): Add Commercial Frontier Model Baselines

Once your RLVR results are done, you can optionally evaluate commercial models into the exact same `eval_runs/` folder so they appear alongside your trained 4B models in `pullgym report`.

- [ ] **Gemini (Google AI Studio free/low-cost tier):**
  ```bash
  export OPENAI_API_KEY="$GEMINI_API_KEY"
  pullgym run-fast \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --model gemini-2.5-flash \
    --n 25 --variant calibrated --concurrency 6 \
    --out /content/drive/MyDrive/pullgym_rlvr/eval_runs --resume
  ```
- [ ] **DeepSeek (50% off-peak discount, ~$2–$5 total):**
  ```bash
  export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
  pullgym run-fast \
    --base-url https://api.deepseek.com/v1 \
    --model deepseek-chat \
    --n 25 --variant calibrated --concurrency 6 \
    --out /content/drive/MyDrive/pullgym_rlvr/eval_runs --resume
  ```
- [ ] **Generate the unified leaderboard table:**
  ```bash
  pullgym report /content/drive/MyDrive/pullgym_rlvr/eval_runs --by model,variant
  pullgym report /content/drive/MyDrive/pullgym_rlvr/eval_runs --by model,family,variant
  ```

---

## Quick Troubleshooting

| Issue in Colab | Fix |
| :--- | :--- |
| **`CUDA out of memory` on L4 (24 GB)** | In Cell 3, lower `NUM_GENERATIONS` from `6` to `4` and `GPU_MEM_UTIL` from `0.50` to `0.40`, then restart session and re-run from Cell 3. |
| **`valid=0%` in Cell 7** | Check the printed sample completion in Cell 7 to see why Python failed (e.g., missing package or formatting issue) and share the output with me. |
| **Colab disconnected mid-run** | Reconnect, run Cells 1–5, then re-run Cell 7. It automatically resumes from the latest checkpoint on Google Drive. |
| **Package conflict after Cell 2 `pip install`** | Click **Runtime → Restart session**, skip re-installing in Cell 2 (or let it check quickly), and run from Cell 3. |
