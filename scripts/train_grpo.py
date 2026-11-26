#!/usr/bin/env python3
"""Standalone GRPO training & evaluation script for pullgym (Unsloth + TRL + vLLM).

Designed to run directly on a Colab Pro L4 (24 GB) or A100 (40 GB) GPU over SSH
via ``google-colab-cli`` (or inside a Colab terminal).

Examples:
    # 10-minute 20-step sanity check (Arm B: Proper Log-Score RLVR)
    python3 scripts/train_grpo.py --arm log_score --sanity

    # Full 350-step Arm B (Proper Log-Score RLVR)
    python3 scripts/train_grpo.py --arm log_score --steps 350

    # Full 350-step Arm A (Binary Point-Accuracy RLVR baseline)
    python3 scripts/train_grpo.py --arm binary --steps 350
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Import unsloth BEFORE trl/transformers/peft so all RL patches apply cleanly.
from unsloth import FastLanguageModel  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from pullgym.report import load_episodes, report  # noqa: E402
from pullgym.rl import (  # noqa: E402
    OOD_TEST_FAMILIES,
    TRAIN_FAMILIES,
    build_rl_records,
    completion_text,
    evaluate_completions_to_trials,
    make_binary_reward_fn,
    make_log_score_reward_fn,
)
from pullgym.stats import summarize  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train & evaluate pullgym GRPO arms.")
    p.add_argument("--arm", choices=["log_score", "binary"], default="log_score")
    p.add_argument("--sanity", action="store_true", help="Run a fast 20-step sanity check")
    p.add_argument("--base-only", action="store_true", help="Evaluate untrained base model only and exit")
    p.add_argument("--steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--model", default="unsloth/Qwen3-8B")
    p.add_argument("--bf16", action="store_true", help="Use 16-bit LoRA instead of 4-bit QLoRA")
    p.add_argument("--variant", choices=["calibrated", "neutral"], default="calibrated")
    p.add_argument("--n-train", type=int, default=250, help="Train episodes per family (x8)")
    p.add_argument("--n-eval-ood", type=int, default=25, help="OOD eval episodes per family (x4)")
    p.add_argument("--n-eval-id", type=int, default=15, help="ID eval episodes per family (x8)")
    p.add_argument("--out-dir", default="/content/pullgym_rlvr", help="Root output directory")
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.2, help="Gradient clipping norm")
    p.add_argument("--run-suffix", default="", help="Optional suffix for run/checkpoint/eval names (e.g. r2)")
    p.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank")
    p.add_argument("--save-steps", type=int, default=None, help="Checkpoint save interval")
    p.add_argument("--max-completion-len", type=int, default=None)
    p.add_argument("--gpu-mem-util", type=float, default=None)
    p.add_argument("--train-temp", type=float, default=0.85, help="GRPO rollout temperature")
    p.add_argument("--train-top-p", type=float, default=0.95)
    p.add_argument("--train-top-k", type=int, default=20)
    p.add_argument("--eval-temp", type=float, default=0.7, help="Eval temperature (Qwen3 recommends 0.7)")
    p.add_argument("--eval-top-p", type=float, default=0.8)
    p.add_argument("--eval-top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.0, help="Prevents Qwen3 comment loops")
    p.add_argument("--exec-workers", type=int, default=16, help="Parallel Python sandbox workers")
    p.add_argument("--eval-only", action="store_true", help="Skip training and evaluate saved LoRA")
    p.add_argument("--skip-base-eval", action="store_true")
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 <think> mode (off by default so completions don't truncate before code)",
    )
    return p.parse_args()


def model_slug(model_name: str) -> str:
    tail = model_name.rstrip("/").split("/")[-1].lower()
    for suf in ("-unsloth-bnb-4bit", "-bnb-4bit"):
        if tail.endswith(suf):
            tail = tail[: -len(suf)]
    return tail


def render_prompt(tokenizer, prompt_msgs: list[dict[str, str]] | str, enable_thinking: bool) -> str:
    if isinstance(prompt_msgs, str):
        return prompt_msgs
    try:
        return tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )


def eval_with_vllm(
    model,
    tokenizer,
    records: list[dict],
    eval_dir: Path,
    *,
    job_id: str,
    model_label: str,
    max_completion_len: int,
    enable_thinking: bool = False,
    lora_request=None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.0,
    exec_workers: int = 16,
    skip_if_exists: bool = True,
) -> Path:
    target_dir = eval_dir / job_id
    if (
        skip_if_exists
        and target_dir.exists()
        and len(list(target_dir.rglob("verifier/details.json"))) == len(records)
    ):
        print(f"Skipping {job_id}: already evaluated ({len(records)} trials).", flush=True)
        print(report(target_dir, group_by="model,family,variant", truth_bins=1), flush=True)
        return target_dir

    prompts = [render_prompt(tokenizer, r["prompt"], enable_thinking) for r in records]
    sp_kwargs = {
        "temperature": temperature,
        "max_tokens": max_completion_len,
        "seed": 42,
    }
    if temperature > 0:
        sp_kwargs["top_p"] = top_p
        sp_kwargs["top_k"] = top_k
    if presence_penalty:
        sp_kwargs["presence_penalty"] = presence_penalty
    sampling = SamplingParams(**sp_kwargs)
    outputs = model.fast_generate(
        prompts,
        sampling_params=sampling,
        lora_request=lora_request,
    )
    completions = [o.outputs[0].text for o in outputs]
    job_dir = evaluate_completions_to_trials(
        records,
        completions,
        eval_dir,
        job_id=job_id,
        model_name=model_label,
        timeout_sec=15.0,
        max_workers=exec_workers,
    )
    print(f"\n=== {model_label} ({job_id}) ===", flush=True)
    print(report(job_dir, group_by="model,family,variant", truth_bins=1), flush=True)
    return job_dir


def save_ood_plot(eval_dir: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, _ = load_episodes(eval_dir)
    ood_rows = [r for r in rows if r["family"] in OOD_TEST_FAMILIES]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in ood_rows:
        by_model[r["model"]].append(r)
    models = sorted(by_model)
    if not models:
        return

    stats = {m: summarize(by_model[m]) for m in models}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = list(range(len(models)))

    axes[0].bar([i - 0.18 for i in x], [stats[m].get("cov68", 0) * 100 for m in models], width=0.35, label="68% cov")
    axes[0].bar([i + 0.18 for i in x], [stats[m].get("cov95", 0) * 100 for m in models], width=0.35, label="95% cov")
    axes[0].axhline(68, color="C0", linestyle="--", alpha=0.7)
    axes[0].axhline(95, color="C1", linestyle="--", alpha=0.7)
    axes[0].set_title("OOD Empirical Coverage (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].legend()

    axes[1].bar(x, [stats[m].get("pull_width_robust", 0) for m in models], color="C2")
    axes[1].axhline(1.0, color="black", linestyle="--", label="Calibrated (1.0)")
    axes[1].set_title("OOD Robust Pull Width")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].legend()

    axes[2].bar(x, [stats[m].get("reward_mean", 0) for m in models], color="C3")
    axes[2].axhline(0.0, color="black", linestyle="--", label="Reference (0.0)")
    axes[2].set_title("OOD Mean Log-Score Reward")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved OOD calibration chart to {out_png}", flush=True)


def main() -> int:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA GPU is required."
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    is_a100 = vram_gb >= 35.0

    lora_rank = args.lora_rank or (32 if is_a100 else 16)
    max_prompt_len = 1536
    max_comp_len = args.max_completion_len or (2048 if is_a100 else 1536)
    max_seq_len = max_prompt_len + max_comp_len
    gpu_mem_util = args.gpu_mem_util or (0.60 if is_a100 else 0.50)
    num_gens = args.num_generations or (8 if is_a100 else 6)
    max_steps = args.steps if args.steps is not None else (20 if args.sanity else 350)
    save_steps = args.save_steps or (10 if args.sanity else 50)

    slug = model_slug(args.model)
    out_root = Path(args.out_dir)
    tag = "sanity" if args.sanity else "full"
    if args.run_suffix:
        tag = f"{tag}_{args.run_suffix}"
    run_name = f"{slug}_{args.arm}_{tag}"
    ckpt_dir = out_root / "checkpoints" / run_name
    lora_dir = out_root / "loras" / run_name
    eval_dir = out_root / "eval_runs"
    for d in (ckpt_dir, lora_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"GPU={gpu_name} ({vram_gb:.1f}GB) | Model={args.model} | Arm={args.arm} | "
        f"BaseOnly={args.base_only} | Sanity={args.sanity} | Steps={max_steps} | "
        f"Gens={num_gens} | GradAccum={args.grad_accum} | MaxComp={max_comp_len} | "
        f"EvalTemp={args.eval_temp} | PresPen={args.presence_penalty} | 4bit={not args.bf16}",
        flush=True,
    )

    train_records = build_rl_records(
        TRAIN_FAMILIES,
        n_per_family=args.n_train,
        seed_start=100_000,
        split="train",
        variant=args.variant,
        shuffle=True,
        shuffle_seed=42,
    )
    eval_ood_records = build_rl_records(
        OOD_TEST_FAMILIES,
        n_per_family=args.n_eval_ood,
        seed_start=0,
        split="dev",
        variant=args.variant,
        shuffle=False,
    )
    eval_id_records = build_rl_records(
        TRAIN_FAMILIES,
        n_per_family=args.n_eval_id,
        seed_start=0,
        split="dev",
        variant=args.variant,
        shuffle=False,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=max_seq_len,
        load_in_4bit=not args.bf16,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_mem_util,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Pre-render prompts with enable_thinking=False by default so Qwen3 writes
    # concise reasoning + Python code without truncating inside <think>.
    for rec in train_records:
        rec["prompt"] = render_prompt(tokenizer, rec["prompt"], args.thinking)
    train_dataset = Dataset.from_list(train_records)

    eval_kw = dict(
        max_completion_len=max_comp_len,
        enable_thinking=args.thinking,
        temperature=args.eval_temp,
        top_p=args.eval_top_p,
        top_k=args.eval_top_k,
        presence_penalty=args.presence_penalty,
        exec_workers=args.exec_workers,
    )

    if not args.skip_base_eval:
        eval_with_vllm(
            model,
            tokenizer,
            eval_ood_records,
            eval_dir,
            job_id=f"{slug}_base_ood_{args.variant}",
            model_label=f"{slug}-base",
            skip_if_exists=not args.base_only,
            **eval_kw,
        )
        eval_with_vllm(
            model,
            tokenizer,
            eval_id_records,
            eval_dir,
            job_id=f"{slug}_base_id_{args.variant}",
            model_label=f"{slug}-base",
            skip_if_exists=not args.base_only,
            **eval_kw,
        )

    if args.base_only:
        print("\n================ BASE-ONLY SUMMARY ================", flush=True)
        print(report(eval_dir, group_by="model,variant", truth_bins=1), flush=True)
        return 0

    if not args.eval_only:
        if args.arm == "log_score":
            base_reward_fn = make_log_score_reward_fn(
                floor=-10.0,
                invalid_penalty=-12.0,
                timeout_sec=8.0,
                max_workers=args.exec_workers,
            )
            invalid_val = -12.0
        else:
            base_reward_fn = make_binary_reward_fn(
                k=1.0,
                invalid_penalty=-1.0,
                timeout_sec=8.0,
                max_workers=args.exec_workers,
            )
            invalid_val = -1.0

        counter = {"n": 0}
        batch_log_path = out_root / f"batch_metrics_{run_name}.jsonl"

        def logged_reward_fn(completions, files_json, truth_json, family=None, **kwargs):
            rewards = base_reward_fn(
                completions, files_json=files_json, truth_json=truth_json, **kwargs
            )
            counter["n"] += 1
            arr = np.asarray(rewards, dtype=float)
            valid_rate = float(np.mean(arr > invalid_val))
            fam_str = family[0] if family else "?"
            rec = {
                "batch": counter["n"],
                "family": fam_str,
                "valid_rate": valid_rate,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
            with batch_log_path.open("a", encoding="utf-8") as bf:
                bf.write(json.dumps(rec) + "\n")
            if counter["n"] <= 5 or counter["n"] % 10 == 0:
                print(
                    f"[batch {counter['n']:4d} | {fam_str:<21s}] "
                    f"valid={valid_rate:.0%} mean={arr.mean():+.3f} std={arr.std():.3f} "
                    f"min={arr.min():+.2f} max={arr.max():+.2f}",
                    flush=True,
                )
            if counter["n"] in {1, 20} or counter["n"] % 50 == 0:
                best_i = int(np.argmax(arr))
                print(
                    f"--- Sample best completion (reward={arr[best_i]:+.3f}) ---\n"
                    f"{completion_text(completions[best_i])[:800]}\n"
                    "---------------------------------------------------",
                    flush=True,
                )
            return rewards

        grpo_kwargs = dict(
            use_vllm=True,
            learning_rate=args.lr,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_steps=max(2, int(0.05 * max_steps)),
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            beta=args.beta,
            temperature=args.train_temp,
            top_p=args.train_top_p,
            top_k=args.train_top_k,
            logging_steps=5,
            per_device_train_batch_size=num_gens,
            gradient_accumulation_steps=args.grad_accum,
            num_generations=num_gens,
            max_completion_length=max_comp_len,
            max_steps=max_steps,
            save_steps=save_steps,
            save_total_limit=6,
            max_grad_norm=args.max_grad_norm,
            report_to="none",
            output_dir=str(ckpt_dir),
        )
        if args.presence_penalty:
            import inspect

            if "generation_kwargs" in inspect.signature(GRPOConfig.__init__).parameters:
                grpo_kwargs["generation_kwargs"] = {
                    "presence_penalty": args.presence_penalty
                }
        training_args = GRPOConfig(**grpo_kwargs)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[logged_reward_fn],
            args=training_args,
            train_dataset=train_dataset,
        )
        has_ckpt = any(ckpt_dir.glob("checkpoint-*"))
        print(f"Starting GRPO training (resume_from_checkpoint={has_ckpt})...", flush=True)
        trainer.train(resume_from_checkpoint=has_ckpt)
        (out_root / f"trainer_history_{run_name}.json").write_text(
            json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
        )
        model.save_lora(str(lora_dir))
        print(f"Saved trained LoRA to {lora_dir}", flush=True)

    trained_label = run_name
    lora_req = model.load_lora(str(lora_dir))
    eval_with_vllm(
        model,
        tokenizer,
        eval_ood_records,
        eval_dir,
        job_id=f"{run_name}_ood_{args.variant}",
        model_label=trained_label,
        lora_request=lora_req,
        skip_if_exists=False,
        **eval_kw,
    )
    eval_with_vllm(
        model,
        tokenizer,
        eval_id_records,
        eval_dir,
        job_id=f"{run_name}_id_{args.variant}",
        model_label=trained_label,
        lora_request=lora_req,
        skip_if_exists=False,
        **eval_kw,
    )

    print("\n================ COMBINED SUMMARY ================", flush=True)
    print(report(eval_dir, group_by="model,variant", truth_bins=1), flush=True)
    print("\n================ PER-FAMILY SUMMARY ================", flush=True)
    print(report(eval_dir, group_by="model,family,variant", truth_bins=1), flush=True)
    save_ood_plot(eval_dir, out_root / f"ood_calibration_{args.variant}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
