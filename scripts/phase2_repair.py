"""Phase 2A: same-prompt multi-turn repair diagnostic on saved RL checkpoints.

Turn 1 is exactly the single-turn RL evaluation prompt (``rl.build_rl_prompt``:
apparatus note + CSV previews) with the Round 2 decoding settings. If the
script crashes or ``artifacts/answer.json`` fails the public schema check, the
policy sees its own truncated STDOUT/STDERR plus ``answer_status`` and may
repair it, up to ``--max-turns`` turns. Nothing derived from the truth is ever
shown to the policy.

Each episode is run once; its outcome after turn 1 (K=1) and after the last
turn (K=max) are written as two job directories, so the K=1 vs K=max comparison
is exactly paired and isolates the effect of execution feedback.

Usage (against an OpenAI-compatible vLLM server serving the base model and the
LoRA adapters under their model names):

    python scripts/phase2_repair.py --base-url http://127.0.0.1:8000/v1 \\
        --models qwen3-8b-base,qwen3-8b_binary_full_r2,qwen3-8b_log_score_full_r2 \\
        --out runs/phase2/2A
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from pullgym.rl import (
    OOD_V01_FAMILIES,
    OOD_V02_FAMILIES,
    TRAIN_FAMILIES,
    build_rl_records,
    repair_episode,
    write_rl_trial,
)
from pullgym.runner import openai_http_chat

# Same evaluation sets as scripts/train_grpo.py (Round 2), plus the new
# Tier-5 families as an extra held-out set.
SETS = {
    "id_calibrated": (TRAIN_FAMILIES, 15, "calibrated"),
    "ood_calibrated": (OOD_V01_FAMILIES, 25, "calibrated"),
    "ood_neutral": (OOD_V01_FAMILIES, 25, "neutral"),
    "ood_v02_calibrated": (OOD_V02_FAMILIES, 25, "calibrated"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--models", required=True, help="comma list of served model names")
    p.add_argument(
        "--sets", default="id_calibrated,ood_calibrated,ood_neutral", help="comma list of: " + ", ".join(SETS)
    )
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--out", default="runs/phase2/2A")
    p.add_argument("--concurrency", type=int, default=48)
    # Round 2 evaluation decoding (scripts/train_grpo.py defaults on A100).
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=15.0, help="per-script execution timeout")
    args = p.parse_args()

    chat = openai_http_chat(args.base_url, api_key="EMPTY", timeout_sec=600.0)
    out = Path(args.out)
    k_max = args.max_turns

    for set_name in args.sets.split(","):
        families, n, variant = SETS[set_name]
        records = build_rl_records(
            families, n_per_family=n, seed_start=0, split="dev", variant=variant, shuffle=False
        )
        for model in args.models.split(","):
            dirs = {k: out / set_name / f"{model}_K{k}" for k in (1, k_max)}
            if all(len(list(d.rglob("verifier/details.json"))) == len(records) for d in dirs.values()):
                print(f"skip {set_name} {model}: done", flush=True)
                continue

            def generate(messages, model=model):
                resp = chat(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "presence_penalty": args.presence_penalty,
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                return (resp["choices"][0]["message"].get("content") or "")

            t0 = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = {
                    pool.submit(repair_episode, r, generate, max_turns=k_max, timeout_sec=args.timeout): r
                    for r in records
                }
                episodes = []
                for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                    episodes.append(fut.result())
                    if i % 50 == 0:
                        print(f"  {set_name} {model}: {i}/{len(records)}", flush=True)
            elapsed = time.monotonic() - t0

            turn_log = out / set_name / f"{model}_turns.jsonl"
            turn_log.parent.mkdir(parents=True, exist_ok=True)
            with turn_log.open("w") as f:
                for ep in episodes:
                    f.write(json.dumps({
                        "task": ep.record["task_name"],
                        "family": ep.record["family"],
                        "turns": len(ep.turns),
                        "status": [t.answer_status or "no code" for t in ep.turns],
                        "exit_code": [(t.exec_result or {}).get("exit_code") for t in ep.turns],
                    }) + "\n")
            for k, d in dirs.items():
                for ep in episodes:
                    used = min(k, len(ep.turns))
                    # Transcript up to and including the k-th assistant turn.
                    n_msgs = len(ep.record["prompt"]) + 2 * used - 1
                    write_rl_trial(
                        d,
                        ep.record,
                        ep.outcome_after(k),
                        ep.messages[:n_msgs],
                        model_name=f"{model}|K{k}",
                        elapsed_sec=elapsed / len(records),
                        turns=used,
                    )
            v1 = sum(ep.outcome_after(1).valid for ep in episodes)
            vk = sum(ep.outcome_after(k_max).valid for ep in episodes)
            print(
                f"{set_name} {model}: valid K1={v1}/{len(records)} K{k_max}={vk}/{len(records)} "
                f"({elapsed:.0f}s)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
