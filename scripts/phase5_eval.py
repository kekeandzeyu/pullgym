#!/usr/bin/env python3
"""Phase 5 evaluation: `run-fast --tool-mode code` episodes against the Phase 5 vLLM server.

The harness is exactly the training one (`pullgym.rl_multiturn` constants):
4 turns, 1280 tokens per turn, observations cut to head 400 + tail 800 chars,
15 s tool timeout, the Phase 5 chat template. Decoding is the Qwen3 evaluation
setting (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.0, thinking
off); each draw uses its own sampling seed.

Sets (decided 2026-11-28, see plan-v0.2.0.md Phase 5 amendment):
  id   TRAIN_FAMILIES x 15 dev seeds, calibrated, 2 draws
  ood  OOD_V01 + OOD_V02 x 25 dev seeds, calibrated and neutral, 4 draws

Output: <out>/<set>_<variant>/draw<d>/<model>/<task>/..., resumable (rerun the
same command; finished episodes are skipped, failed ones retried).

    python scripts/phase5_eval.py --models qwen3-8b-base \\
        --adapter qwen3-8b_log_score_mt_s0=runs/phase5/train/qwen3-8b_log_score_mt_s0/final_adapter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from pullgym.families import get_family
from pullgym.rl import OOD_V01_FAMILIES, OOD_V02_FAMILIES, TRAIN_FAMILIES
from pullgym.rl_multiturn import (
    MT_MAX_TOKENS,
    MT_MAX_TURNS,
    MT_OBS_TRUNCATE,
    MT_TOOL_TIMEOUT_SEC,
    sampling_extra,
)
from pullgym.runner import OpenAICompatibleAgent, run_suite

SETS = {
    # name: (families, tasks per family, variants, draws)
    "id": (TRAIN_FAMILIES, 15, ("calibrated",), 2),
    "ood": (OOD_V01_FAMILIES + OOD_V02_FAMILIES, 25, ("calibrated", "neutral"), 4),
}


def _post(url: str, body: dict) -> None:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=300).read()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="http://127.0.0.1:8000/v1")
    p.add_argument("--models", default="", help="comma list of already-served model names")
    p.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH",
                   help="LoRA adapter to load into the server and evaluate (repeatable)")
    p.add_argument("--sets", default="id,ood")
    p.add_argument("--draws", type=int, help="override the number of draws for every set")
    p.add_argument("--out", default="runs/phase5/eval")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.0)
    args = p.parse_args(argv)

    models = [m for m in args.models.split(",") if m]
    served = {m["id"] for m in json.loads(
        urllib.request.urlopen(f"{args.server}/models", timeout=30).read())["data"]}
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_dir():
            print(f"bad --adapter {spec!r} (need NAME=existing_dir)", file=sys.stderr)
            return 2
        if name not in served:
            _post(f"{args.server}/load_lora_adapter",
                  {"lora_name": name, "lora_path": str(Path(path).resolve())})
        models.append(name)
    if not models:
        print("nothing to evaluate: pass --models and/or --adapter", file=sys.stderr)
        return 2

    failures = 0
    for set_name in args.sets.split(","):
        fams, n, variants, draws = SETS[set_name]
        draws = args.draws or draws
        families = [get_family(f) for f in fams]
        for variant in variants:
            for d in range(draws):
                for model in models:
                    agent = OpenAICompatibleAgent(
                        model, base_url=args.server, api_key="EMPTY",
                        max_turns=MT_MAX_TURNS, temperature=args.temperature,
                        max_tokens=MT_MAX_TOKENS, tool_mode="code",
                        extra_body=sampling_extra(top_p=args.top_p, top_k=args.top_k,
                                                  presence_penalty=args.presence_penalty,
                                                  seed=1000 + d),
                        obs_truncate=MT_OBS_TRUNCATE,
                    )
                    t0 = time.monotonic()
                    job_dir, outcomes = run_suite(
                        families, split="dev", n=n, start=0, variant=variant, agent=agent,
                        out_dir=Path(args.out) / f"{set_name}_{variant}" / f"draw{d}",
                        job_id=model, concurrency=args.concurrency,
                        tool_timeout_sec=MT_TOOL_TIMEOUT_SEC, resume=True,
                    )
                    errs = [o for o in outcomes if o.error]
                    failures += len(errs)
                    valid = sum(1 for o in outcomes if (o.rewards or {}).get("valid"))
                    print(f"[{time.strftime('%H:%M:%S')}] {set_name}_{variant} draw{d} {model}: "
                          f"{len(outcomes)} episodes, {valid} valid, {len(errs)} errors, "
                          f"{time.monotonic() - t0:.0f}s", flush=True)
                    for o in errs[:3]:
                        print(f"   error {o.task_name}: {o.error[:200]}", flush=True)
    print(f"done; {failures} failed episodes (rerun the same command to retry them)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
