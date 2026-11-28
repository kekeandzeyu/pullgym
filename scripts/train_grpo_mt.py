#!/usr/bin/env python3
"""Phase 5: multi-turn Dr. GRPO through the exact `run-fast --tool-mode code` harness.

Two processes on one GPU:

1. A vLLM OpenAI server (started by `scripts/phase5_serve.sh`) that serves the
   4-bit base model with the Phase 5 chat template, runtime LoRA loading, and
   `return_token_ids`.
2. This trainer. Each step it hot-loads the current LoRA into the server, runs
   `prompts_per_step x num_generations` training episodes through
   `pullgym.rl_multiturn.rollout_episode` (the run-fast agent loop), and
   updates the LoRA on the recorded token streams.

The loss is Dr. GRPO: token-summed policy gradient with a constant denominator
(`prompts x generations x max_turns x max_tokens`) and an optional k3 KL term
to the base model. Advantages are group mean-centred and std-scaled, so the
`binary` and `log_score` arms train with the same advantage scale and differ
only in the reward. Rewards are terminal (`floor=-10`, `invalid_penalty=-12`).
Episodes whose group has zero reward spread carry no gradient and are skipped.

    # 5-step smoke test (prints time per step and peak memory)
    python scripts/train_grpo_mt.py --arm log_score --seed 0 --steps 5 --run-name smoke

    # Full run
    python scripts/train_grpo_mt.py --arm log_score --seed 0 --steps 150
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from pullgym.rl_multiturn import (
    MT_MAX_TOKENS,
    MT_MAX_TURNS,
    TEMPLATE_PATH,
    Rollout,
    group_advantages,
    rollout_episode,
    sampling_extra,
    train_schedule,
)
from pullgym.runner import openai_http_chat

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=["log_score", "binary"], required=True)
    p.add_argument("--seed", type=int, required=True, help="training seed (data order, sampling, LoRA init)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--run-name", help="default: qwen3-8b_<arm>_mt_s<seed>")
    p.add_argument("--model-path", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    p.add_argument("--server", default="http://127.0.0.1:8000/v1")
    p.add_argument("--out", default="runs/phase5")
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.01, help="KL coefficient (0 disables the reference pass)")
    p.add_argument("--max-grad-norm", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-turns", type=int, default=MT_MAX_TURNS)
    p.add_argument("--max-tokens", type=int, default=MT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.0)
    p.add_argument("--rollout-workers", type=int, default=32)
    p.add_argument("--save-every", type=int, default=10, help="resumable checkpoint interval (steps)")
    p.add_argument("--logit-chunk", type=int, default=512)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# vLLM server control


class VLLMServer:
    def __init__(self, base_url: str, timeout_sec: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        self.chat = openai_http_chat(self.base_url, "EMPTY", timeout_sec=timeout_sec)

    def _post(self, path: str, body: dict[str, Any]) -> str:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read().decode()

    def wait_ready(self, timeout_sec: float = 1800.0) -> None:
        t0 = time.monotonic()
        while True:
            try:
                with urllib.request.urlopen(f"{self.root}/health", timeout=10) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            if time.monotonic() - t0 > timeout_sec:
                raise RuntimeError(f"vLLM server at {self.root} not ready after {timeout_sec}s")
            time.sleep(5)

    def load_lora(self, name: str, path: Path) -> None:
        self._post("/load_lora_adapter", {"lora_name": name, "lora_path": str(Path(path).resolve())})

    def unload_lora(self, name: str) -> None:
        try:
            self._post("/unload_lora_adapter", {"lora_name": name})
        except OSError as e:  # already gone: harmless
            print(f"warning: unload {name}: {e}", flush=True)


def check_server_template(server: VLLMServer, tokenizer, model_name: str) -> None:
    """Abort unless the server renders prompts with the Phase 5 template."""
    msgs = [
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "Again."},
    ]
    resp = server.chat({
        "model": model_name, "messages": msgs, "max_tokens": 4, "temperature": 0.0,
        "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False},
    })
    got = resp.get("prompt_token_ids")
    want = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, chat_template=TEMPLATE_PATH.read_text()
    )
    if hasattr(want, "input_ids"):
        want = want["input_ids"]
    if got is None or list(got) != list(want):
        raise SystemExit(
            "server prompt tokens differ from the Phase 5 template; start the server with "
            "scripts/phase5_serve.sh (--chat-template src/pullgym/templates/qwen3_nothink_mt.jinja)"
        )
    if resp["choices"][0].get("token_ids") is None:
        raise SystemExit("server does not return completion token_ids (vLLM too old?)")


# ---------------------------------------------------------------------------
# Policy model and loss


def load_policy(model_path: str, *, lora_rank: int, seed: int):
    """Load the (pre-quantized 4-bit) base model and wrap it with a fresh LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kw: dict[str, Any] = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    return wrap_lora(model, lora_rank), tokenizer


def wrap_lora(model, lora_rank: int):
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def _decoder(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def action_logprobs(model, input_ids: list[int], action_mask: list[int], *,
                    temperature: float, chunk: int = 512):
    """Log-probabilities (at the sampling temperature) of the tokens with action_mask 1.

    Logits are computed only at action positions, in checkpointed chunks, so a
    long multi-turn sequence never materialises a [T, vocab] tensor.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    dev = next(p for p in model.parameters()).device
    ids = torch.tensor(input_ids, device=dev).unsqueeze(0)
    mask = torch.tensor(action_mask, device=dev)
    if int(mask[0]) != 0:
        raise ValueError("the first token of a segment cannot be an action token")
    pos = torch.nonzero(mask[1:], as_tuple=False).squeeze(-1)  # token pos+1 is predicted from pos
    decoder, lm_head = _decoder(model)
    hidden = decoder(input_ids=ids).last_hidden_state[0]
    h = hidden.index_select(0, pos)
    targets = ids[0].index_select(0, pos + 1)

    def _chunk_logp(hc, tc):
        logits = lm_head(hc).float() / temperature
        return torch.log_softmax(logits, dim=-1).gather(-1, tc.unsqueeze(-1)).squeeze(-1)

    parts = []
    for i in range(0, h.shape[0], chunk):
        hc, tc = h[i:i + chunk], targets[i:i + chunk]
        if torch.is_grad_enabled():
            parts.append(checkpoint(_chunk_logp, hc, tc, use_reentrant=False))
        else:
            parts.append(_chunk_logp(hc, tc))
    return torch.cat(parts) if parts else h.new_zeros(0, dtype=torch.float32)


def segment_backward(model, input_ids, action_mask, advantage: float, *, temperature: float,
                     beta: float, norm: float, chunk: int) -> dict[str, float]:
    """Dr. GRPO loss of one segment (on-policy, one update per batch); calls backward."""
    import torch

    logp = action_logprobs(model, input_ids, action_mask, temperature=temperature, chunk=chunk)
    ratio = torch.exp(logp - logp.detach())  # == 1; keeps the PPO form for later off-policy use
    per_tok = -advantage * ratio
    kl_mean = 0.0
    if beta > 0:
        with torch.no_grad(), model.disable_adapter():
            ref = action_logprobs(model, input_ids, action_mask, temperature=temperature, chunk=chunk)
        d = ref - logp
        kl = torch.exp(d) - d - 1.0
        per_tok = per_tok + beta * kl
        kl_mean = float(kl.detach().mean())
    loss = per_tok.sum() / norm
    loss.backward()
    return {"loss": float(loss.detach()), "kl": kl_mean, "n_tok": int(logp.numel())}


# ---------------------------------------------------------------------------
# Training loop


def _save_adapter(model, path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    model.save_pretrained(str(path))
    return path


def collect_rollouts(server: VLLMServer, policy_name: str, prompts: list[tuple[str, int]],
                     step: int, args, eos_id: int, step_dir: Path) -> list[list[Rollout]]:
    G = args.num_generations

    def one(pi: int, g: int) -> Rollout:
        fam, seed = prompts[pi]
        req_seed = (args.seed * 1_000_003 + step * 1009 + pi * G + g) % (2**31 - 1)
        extra = sampling_extra(top_p=args.top_p, top_k=args.top_k,
                               presence_penalty=args.presence_penalty, seed=req_seed)
        ro = None
        for attempt in range(2):  # one retry for infrastructure failures
            ro = rollout_episode(
                fam, seed, server.chat, step_dir / f"p{pi}_g{g}_a{attempt}",
                model=policy_name, arm=args.arm, eos_id=eos_id,
                temperature=args.temperature, extra_body=extra,
                max_turns=args.max_turns, max_tokens=args.max_tokens,
            )
            if ro.error is None:
                break
        assert ro is not None
        return ro

    groups: list[list[Rollout | None]] = [[None] * G for _ in prompts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.rollout_workers) as pool:
        futs = {pool.submit(one, pi, g): (pi, g) for pi in range(len(prompts)) for g in range(G)}
        for fut in concurrent.futures.as_completed(futs):
            pi, g = futs[fut]
            groups[pi][g] = fut.result()
    return [[r for r in grp if r is not None] for grp in groups]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_cosine_schedule_with_warmup

    run = args.run_name or f"qwen3-8b_{args.arm}_mt_s{args.seed}"
    root = Path(args.out) / "train" / run
    adapters, ckpts, rollouts_dir = root / "adapters", root / "checkpoints", root / "rollouts"
    for d in (adapters, ckpts, rollouts_dir):
        d.mkdir(parents=True, exist_ok=True)
    (root / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_path = root / "steps.jsonl"

    server = VLLMServer(args.server)
    server.wait_ready()
    model, tokenizer = load_policy(args.model_path, lora_rank=args.lora_rank, seed=args.seed)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    served = json.loads(urllib.request.urlopen(f"{server.base_url}/models", timeout=30).read())
    base_name = served["data"][0]["id"]
    check_server_template(server, tokenizer, base_name)

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb

        opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr, betas=(0.9, 0.99),
                                       weight_decay=args.weight_decay)
    except ImportError:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, max(2, int(args.warmup_frac * args.steps)), args.steps)

    start = 0
    done = sorted(ckpts.glob("step_*"))
    if done:
        last = done[-1]
        saved = load_file(str(last / "adapter" / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, saved)
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        b_norm = sum(float(p.detach().float().norm()) for n, p in model.named_parameters()
                     if "lora_B" in n)
        # A fresh LoRA has B = 0, so a zero norm means the checkpoint did not load.
        if unexpected or b_norm == 0.0:
            raise SystemExit(f"resume from {last} failed: {len(unexpected)} unexpected keys, "
                             f"lora_B norm {b_norm}")
        state = torch.load(last / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start = int(state["step"])
        print(f"resumed from {last} (step {start}, {len(saved)} tensors, lora_B norm {b_norm:.4f})",
              flush=True)

    schedule = train_schedule(args.steps, args.prompts_per_step, shuffle_seed=1000 + args.seed)
    norm = float(args.prompts_per_step * args.num_generations * args.max_turns * args.max_tokens)
    prev_name: str | None = None

    for step in range(start, args.steps):
        t0 = time.monotonic()
        name = f"{run}-s{step}"
        server.load_lora(name, _save_adapter(model, adapters / f"step_{step:04d}"))
        if prev_name:
            server.unload_lora(prev_name)
        prev_name = name

        groups = collect_rollouts(server, name, schedule[step], step, args, eos_id,
                                  rollouts_dir / f"step_{step:04d}")
        t_roll = time.monotonic() - t0

        model.train()
        stats = {"loss": 0.0, "kl": [], "n_tok": 0, "segments": 0}
        rewards_all, valid_all, turns_all, breaks, zero_std, errors = [], [], [], 0, 0, 0
        for grp in groups:
            ok = [r for r in grp if r.error is None]
            errors += len(grp) - len(ok)
            if len(ok) < 2:
                zero_std += 1
                continue
            adv = group_advantages([r.reward for r in ok])
            if all(a == 0.0 for a in adv):
                zero_std += 1
            for r, a in zip(ok, adv):
                rewards_all.append(r.reward)
                valid_all.append(r.valid)
                turns_all.append(r.turns)
                breaks += r.prefix_breaks
                if a == 0.0:
                    continue
                for seg in r.segments:
                    s = segment_backward(model, seg.input_ids, seg.action_mask, a,
                                         temperature=args.temperature, beta=args.beta,
                                         norm=norm, chunk=args.logit_chunk)
                    stats["loss"] += s["loss"]
                    stats["kl"].append(s["kl"])
                    stats["n_tok"] += s["n_tok"]
                    stats["segments"] += 1
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        t_total = time.monotonic() - t0

        rec = {
            "step": step + 1,
            "reward_mean": float(np.mean(rewards_all)) if rewards_all else None,
            "valid_rate": float(np.mean(valid_all)) if valid_all else None,
            "turns_mean": float(np.mean(turns_all)) if turns_all else None,
            "zero_std_groups": zero_std,
            "rollout_errors": errors,
            "prefix_breaks": breaks,
            "segments": stats["segments"],
            "action_tokens": stats["n_tok"],
            "loss": stats["loss"],
            "kl_mean": float(np.mean(stats["kl"])) if stats["kl"] else 0.0,
            "grad_norm": grad_norm,
            "lr": sched.get_last_lr()[0],
            "rollout_sec": round(t_roll, 1),
            "step_sec": round(t_total, 1),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if errors > len(groups) * args.num_generations // 2:
            raise SystemExit(f"step {step + 1}: {errors} rollout errors; check the vLLM server")

        # Keep only the adapter the server is using now.
        for old in adapters.glob("step_*"):
            if old.name != f"step_{step:04d}":
                shutil.rmtree(old, ignore_errors=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            ck = ckpts / f"step_{step + 1:04d}"
            _save_adapter(model, ck / "adapter")
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step + 1},
                       ck / "trainer_state.pt")
            for old in sorted(ckpts.glob("step_*"))[:-2]:
                shutil.rmtree(old, ignore_errors=True)

    final = _save_adapter(model, root / "final_adapter")
    print(f"saved {final}", flush=True)
    if prev_name:
        server.unload_lora(prev_name)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
