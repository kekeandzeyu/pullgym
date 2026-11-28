#!/usr/bin/env bash
# Phase 5 vLLM server: 4-bit Qwen3-8B with the Phase 5 chat template, runtime
# LoRA loading (the trainer hot-loads its adapter every step) and prefix caching.
# Training and evaluation both talk to a server started by this script.
#
#   bash scripts/phase5_serve.sh                 # foreground
#   GPU_UTIL=0.85 bash scripts/phase5_serve.sh   # evaluation only (no trainer on the GPU)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-unsloth/Qwen3-8B-unsloth-bnb-4bit}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.42}"   # leaves the rest of a 40 GB A100 to the trainer
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec vllm serve "$MODEL" \
  --served-model-name qwen3-8b-base \
  --quantization bitsandbytes \
  --enable-lora --max-lora-rank 32 --max-loras 8 \
  --max-model-len 10240 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --chat-template "$HERE/../src/pullgym/templates/qwen3_nothink_mt.jinja" \
  --enable-prefix-caching \
  --port "$PORT" --seed 0 "$@"
