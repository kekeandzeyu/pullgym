# Phase 5 runbook: multi-turn GRPO (`base-mt` / `binary-mt` / `log_score-mt`)

These are the decisions behind this runbook. The rationale is in [ROADMAP §2–2b](ROADMAP.md).
* **Training rollouts use the evaluation harness.** They run through the exact `pullgym run-fast --tool-mode code` loop:
  * 4 turns of at most 1,280 tokens each.
  * STDOUT/STDERR cut to the first 400 and last 800 characters.
  * A 15 s tool timeout.
* **One model server.** Training and evaluation talk to the same vLLM server. It serves the 4-bit Qwen3-8B with the template `src/pullgym/templates/qwen3_nothink_mt.jinja`.
* **Runs:**
  * `binary` and `log_score` are trained with 3 seeds each.
  * `base` is evaluated without training.
* **Evaluation draws:**
  * The 8 held-out families (25 tasks each) get 4 draws per task, on both the calibrated and neutral prompts.
  * The 8 training families (15 tasks each) get 2 draws per task, on the calibrated prompt only.

## 0. One-time setup on an A100-40GB VM

```bash
cd /root/autodl-tmp && source venv/bin/activate      # the venv from Phase 2 (vLLM + bitsandbytes)
source /etc/network_turbo 2>/dev/null || true        # AutoDL network accelerator, if available
pip install --prefer-binary peft safetensors        # trainer dependencies (transformers comes with vLLM)
cd pullgym && git pull && pip install -e .           # or rsync the repo from the laptop
# Model weights are in /root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit. If they are missing:
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download unsloth/Qwen3-8B-unsloth-bnb-4bit \
#       --local-dir /root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit
pytest -q tests/test_rl_multiturn.py tests/test_train_grpo_mt.py   # CPU tests, including the trainer loss
```

## 1. Server (training: shares the GPU with the trainer)

```bash
MODEL=/root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit GPU_UTIL=0.42 \
  nohup bash scripts/phase5_serve.sh > /root/autodl-tmp/serve.log 2>&1 &
```

## 2. Smoke test (5 steps): run this first

```bash
python scripts/train_grpo_mt.py --arm log_score --seed 0 --steps 5 --run-name smoke \
  --model-path /root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit 2>&1 | tee /root/autodl-tmp/smoke.log
```

At startup, the trainer checks that the server's prompt tokens match the Phase 5 template, and that the server returns completion token ids. It aborts if either check fails.

Each line of `steps.jsonl` reports:
* `rollout_sec` and `step_sec`;
* `peak_mem_gb`;
* `valid_rate`;
* `prefix_breaks` (should be about 0);
* `zero_std_groups`;
* `rollout_errors` (must be 0).

Multiply `step_sec` by 150 to get the length of a full run. If memory is tight, lower `--rollout-workers` or the server's `GPU_UTIL`.

## 3. Full training runs (one at a time per GPU; resumable)

```bash
for arm in log_score binary; do for s in 0 1 2; do
  python scripts/train_grpo_mt.py --arm $arm --seed $s --steps 150 \
    --model-path /root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit \
    2>&1 | tee -a /root/autodl-tmp/train_${arm}_s$s.log
done; done
```

If a run is interrupted, rerun the same command: it resumes from the last 10-step checkpoint. With two VMs, give each VM one arm, or split the seeds between them.

## 4. Evaluation (no trainer on the GPU, so the server can use most of it)

```bash
MODEL=/root/autodl-tmp/models/Qwen3-8B-unsloth-bnb-4bit GPU_UTIL=0.88 \
  nohup bash scripts/phase5_serve.sh > /root/autodl-tmp/serve_eval.log 2>&1 &
A=runs/phase5/train
python scripts/phase5_eval.py --models qwen3-8b-base \
  $(for arm in binary log_score; do for s in 0 1 2; do
      n=qwen3-8b_${arm}_mt_s$s; echo --adapter $n=$A/$n/final_adapter; done; done)
```

The command can be rerun safely: finished episodes are skipped and failed ones retried. It exits non-zero while any episode has an infrastructure error.

## 5. Tables

```bash
python scripts/phase5_analysis.py --format md > runs/phase5/phase5_tables.md
```

The analysis averages each arm over its training seeds and draws for every task, then runs a bootstrap over tasks for `log_score − binary`, `log_score − base` and `binary − base`. It also prints the per-seed spread and the Tier-1 width-inflation audit.

Claim OOD transfer only if the paired CI excludes zero. Otherwise the claim is "log score does no harm OOD".
