# pullgym — Training Reproducibility & Handoff Guide

> **Completed Status:** Both Round 1 (`lr=1e-5, beta=0.04, steps=150`) and Round 2 (`lr=5e-5, beta=0.01, max_grad_norm=1.0, steps=150`) `Qwen3-8B` GRPO experiments (`log_score` vs. `binary`, 1,600 evaluated episodes) are **complete and audited** in [docs/RLVR_RESULTS.md](RLVR_RESULTS.md). Keep this guide for reproducing or extending those runs (for Round 2 effective adaptation, pass `--steps 150 --lr 5e-5 --beta 0.01 --num-generations 8 --grad-accum 4`).

Use this file to reproduce or extend the **Proper-Scoring-Rule RLVR (`log_score`) vs. Binary GRPO (`binary`)** experiment with zero wasted setup.

> **Fastest way to resume in a new chat:** tell the agent:
> *"Read `docs/RESUME_TRAINING.md` and run Option A on Colab A100."*

---

## 1. Current State Snapshot

- **Git:** All core code, tests, scripts, and notebook are pushed to `origin/main` (`3f7db7f`), with **86/86 fast tests passing** (`uv run pytest`).
- **Colab Account:** `google-colab-cli` (`colab`) is installed and authenticated; **~188.10 compute units** remaining, **0 active VMs**.
- **Saved Baseline Artifact (already computed — do NOT re-run):**
  - `runs/qwen3_8b_base_eval.tar.gz` — 220 evaluated episodes of untrained `unsloth/Qwen3-8B` (100 OOD + 120 ID, `variant="calibrated"`).

### Already-Measured `Qwen3-8B` Base Results (`runs/qwen3_8b_base_eval.tar.gz`)

| Split | `n` | Valid | 68% Cov *(target 68%)* | 95% Cov *(target 95%)* | Pull Width *(target 1.0)* | Sharpness *(ref 1.0)* | Mean Reward *(ref 0.0)* |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OOD (4 unseen systematic families)** | 100 | 56% | **33.9%** | **44.6%** | **14.76** | 1.41 | **−6.518** |
| **All 12 families combined** | 220 | 57% | **38.1%** | **54.0%** | **3.33** | 1.00 | **−5.449** |

Per-family OOD traps on `qwen3-8b-base`:
- `dead_time`: 32% valid, **0.0%** cov68 / **0.0%** cov95, sharpness 0.18, reward **−8.79**
- `no_signal`: 48% valid, **16.7%** cov68 / **25.0%** cov95, sharpness 0.11, reward **−7.37**
- `errors_xy`: 52% valid, **23.1%** cov68 / **38.5%** cov95, pull width 1.91, reward **−4.87**
- `correlated_norm`: 92% valid, **60.9%** cov68 / **73.9%** cov95, reward **−5.05**

---

## 2. Hard-Won Colab Fixes (Already Baked into the Code)

All of these are already fixed in `src/pullgym/rl.py`, `scripts/train_grpo.py`, and `notebooks/train_grpo_colab.ipynb`:

1. **Model choice:** Use **`unsloth/Qwen3-8B`** (4-bit QLoRA). `Qwen3-4B` was too weak at `numpy`/`scipy` coding; `Qwen3-8B` gets 57% zero-shot valid (92–100% on clean/medium families) with massive room to improve calibration.
2. **Disable `<think>` mode (`enable_thinking=False`):** Default Qwen3 thinking mode exhausts 1,536 tokens before outputting Python code.
3. **Never use greedy decoding (`temperature=0.0`) on Qwen3:** Causes infinite comment repetition loops. Eval uses Qwen3's recommended `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.0` (`0` repetition loops across all 220 episodes).
4. **Colab install command:** Use `pip install --prefer-binary -q unsloth vllm vllm-bnb-plugin` (avoid `uv pip install` on Colab which tries to build CUDA extensions from source; `vllm >= 0.28` requires `vllm-bnb-plugin` for 4-bit `fast_inference`, and `train_grpo.py` must run in a fresh subprocess after install so Pillow & `vllm_bnb_plugin` load cleanly).
5. **Avoid namespace shadowing:** Extract/clone repo into `/content/pullgym_repo` (never `/content/pullgym`, because `/content` is on `sys.path[0]` in Colab).
6. **Colab disconnect lesson:** Ephemeral `/content` on CLI VMs is wiped if Colab reclaims an unattended VM after ~90 min. Either run on **A100** while your Mac stays awake (~45–55 min per arm, Option A) **or** run in the Colab web UI with **Google Drive mounted** (Option B).

---

## 3. Option A — Fast A100 via CLI (~45–55 min/arm, ~7 units/arm)

Best when your Mac will stay awake for ~1.5–2 hours. A100-40GB costs **`8.16 units/hr`** (~14–16 units total for both arms).

### Step-by-step
1. **Provision A100** (retry if Colab returns a transient `503`):
   ```bash
   colab new -s pullgym-a100 --gpu A100
   ```
2. **Package & upload code + cached base evals:**
   ```bash
   COPYFILE_DISABLE=1 tar --exclude='.venv' --exclude='.git' --exclude='__pycache__' \
     --exclude='.pytest_cache' --exclude='*.pyc' --exclude='runs' \
     -czf /tmp/pullgym_upload.tar.gz pyproject.toml README.md LICENSE src tests scripts

   colab upload -s pullgym-a100 /tmp/pullgym_upload.tar.gz /content/pullgym_upload.tar.gz
   colab upload -s pullgym-a100 runs/qwen3_8b_base_eval.tar.gz /content/qwen3_8b_base_eval.tar.gz
   ```
3. **Unpack, restore cached base evals, and install dependencies** (run via `colab exec -s pullgym-a100 --timeout 600 -f ...`):
   - Extract `/content/pullgym_upload.tar.gz` → `/content/pullgym_repo`
   - Extract `/content/qwen3_8b_base_eval.tar.gz` → `/content/pullgym_rlvr/` (restores `eval_runs/`)
   - Run `pip install --prefer-binary -q -e /content/pullgym_repo pytest pandas matplotlib`
   - Run `pip install --prefer-binary -q unsloth vllm vllm-bnb-plugin`
4. **Train Arm B (`log_score`, 100 steps) + evaluate + archive immediately:**
   ```bash
   python3 -u /content/pullgym_repo/scripts/train_grpo.py \
     --model unsloth/Qwen3-8B --arm log_score --steps 100 \
     --num-generations 4 --grad-accum 4 --max-completion-len 1536 \
     --gpu-mem-util 0.60 --lr 1e-5 --skip-base-eval \
     --out-dir /content/pullgym_rlvr
   ```
   - Immediately tar & download `/content/pullgym_rlvr/{eval_runs,loras,ood_calibration_calibrated.png}` to `runs/arm_b_log_score.tar.gz` before starting Arm A so Arm B is safely backed up on your Mac!
5. **Train Arm A (`binary`, 100 steps) + evaluate + archive:**
   ```bash
   python3 -u /content/pullgym_repo/scripts/train_grpo.py \
     --model unsloth/Qwen3-8B --arm binary --steps 100 \
     --num-generations 4 --grad-accum 4 --max-completion-len 1536 \
     --gpu-mem-util 0.60 --lr 1e-5 --skip-base-eval \
     --out-dir /content/pullgym_rlvr
   ```
   - Tar & download `/content/pullgym_rlvr/{eval_runs,loras,ood_calibration_calibrated.png}` to `runs/both_arms_final.tar.gz`.
6. **Stop the VM immediately to stop billing:**
   ```bash
   colab stop -s pullgym-a100
   ```

---

## 4. Option B — Browser Colab with Google Drive Persistence (Disconnect-Safe)

Best if you want to leave training unattended or close your laptop.

1. Sync the latest `notebooks/train_grpo_colab.ipynb` (already updated with `Qwen3-8B`, `ENABLE_THINKING=False`, `temp=0.7`, `presence_penalty=1.0`, `vllm-bnb-plugin`, `lr=1e-5`, `GRAD_ACCUM=4`, `MAX_STEPS=100`) to Google Drive (or open it directly from GitHub `kekeandzeyu/pullgym` with `USE_DRIVE_REPO = False`).
2. Select **Runtime → Change runtime type → A100 GPU** (or **L4 GPU**).
3. Run Cell 1 & Cell 2, then **Runtime → Restart session** once if Pillow was upgraded (skip Cell 2 after restart).
4. Set `SANITY_CHECK = False` and `ARM = "log_score"` in Cell 3 → run Cells 3–8. Checkpoints save to `/content/drive/MyDrive/pullgym_rlvr/` every 50 steps and auto-resume after any disconnect.
5. Restart session, set `ARM = "binary"` in Cell 3 → run Cells 1, 3, 4, 5, 7, 8, 9.

---

## 5. After Both Arms Finish

1. Extract `runs/both_arms_final.tar.gz` locally and generate the full comparison tables:
   ```bash
   uv run pullgym report runs/eval_runs --by model,variant
   uv run pullgym report runs/eval_runs --by model,family,variant
   ```
2. Inspect `ood_calibration_calibrated.png` and side-by-side generated Python scripts on held-out `dead_time`, `correlated_norm`, `errors_xy`, and `no_signal`:
   - **Lazy width-inflation check:** Compare ID Tier-1 `sharpness` (`line_fit`, `weighted_mean` — should remain near `~1.0`) against OOD Tier-3/4 `sharpness` & `cov68`/`cov95`. Verify from generated scripts that OOD coverage gains come from parsing `apparatus_note.md` systematics rather than multiplying every `scipy` error bar by a constant `5×–8×`.
   - **Valid-rate vs. fallback check:** Note that `train_grpo.py` trains with `floor=-10.0, invalid_penalty=-12.0` (`rl.py`), while `pullgym report` scores missing/invalid answers via `scoring.py` (`prior - 1 ≈ -3`). Always tabulate `valid%`, valid-only coverage/sharpness, and all-episode mean reward together.
   - **Prior-art framing:** Cite 2025–2026 scalar verbalized-confidence GRPO papers (`RLCR`, `CAPO`) and frame `pullgym` around **procedural code-level uncertainty propagation** across continuous quantiles under instrument systematics.
3. Run commercial/frontier baselines into the same `runs/eval_runs` directory via `pullgym run-fast` (Gemini Flash / Claude Sonnet / DeepSeek off-peak, `neutral` vs. `calibrated` variants) and publish the v0.1 calibration table + RLVR writeup in `README.md`.
