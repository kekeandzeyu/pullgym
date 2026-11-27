"""Phase 4 frontier matrix on free and low-cost endpoints, resumable across days.

    set -a; . ./api_keys.secret; set +a
    caffeinate -i uv run python scripts/phase4_matrix.py --models glm-5.3-flash,qwen3.5-397b

Every (model, variant, family) job is run with ``resume=True`` into
``runs/phase4/<split>_<variant>/<label>``, so re-running the script after a daily
quota reset continues where it stopped. Stages go in priority order: Tier 5
``calibrated`` first (the new result), then Tiers 1-4, then ``neutral``.

Quota exhaustion is detected per family: if most attempted episodes of a family
end in an API exception (no verifier output), the model is parked for this
invocation. Those episodes have no ``reward.json`` and are retried on the next
run; they are never scored. Model-caused provider errors after turn 1 (HTTP 400)
are scored by the runner (see ``runner.MODEL_ATTRIBUTABLE_HTTP``).

All rows use ``code`` mode with ``max_turns=10`` unless a model cannot (Gemini
answers a bare-code prompt with MALFORMED_FUNCTION_CALL, so it runs ``tools``);
the mode is recorded in ``runs/phase4/models.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from pullgym.families import FAMILIES, get_family
from pullgym.runner import OpenAICompatibleAgent, run_suite

MS = "https://api-inference.modelscope.cn/v1"
ZHIPU = "https://open.bigmodel.cn/api/paas/v4"
GROQ = "https://api.groq.com/openai/v1"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER = "https://openrouter.ai/api/v1"
DEEPSEEK = "https://api.deepseek.com/v1"

# label -> endpoint settings. Temperatures follow each vendor's recommendation.
MODELS: dict[str, dict] = {
    "glm-5.3-flash": dict(base=ZHIPU, key="ZHIPU_API_KEY", model="glm-5.3-flash", temp=1.0, conc=2),
    "glm-5.2": dict(base=MS, key="MODELSCOPE_API_KEY", model="ZhipuAI/GLM-5.2", temp=1.0, conc=1),
    "qwen3.5-397b": dict(base=MS, key="MODELSCOPE_API_KEY", model="Qwen/Qwen3.5-397B-A17B", temp=0.6, conc=2),
    "qwen3.8-27b": dict(base=MS, key="MODELSCOPE_API_KEY", model="Qwen/Qwen3.8-27B", temp=0.6, conc=2),
    "deepseek-v4-pro": dict(base=MS, key="MODELSCOPE_API_KEY", model="deepseek-ai/DeepSeek-V4-Pro-0813", temp=1.0, conc=1),
    "deepseek-v4.1-flash": dict(base=MS, key="MODELSCOPE_API_KEY", model="deepseek-ai/DeepSeek-V4.1-Flash-1017", temp=1.0, conc=1),
    "gemini-3.5-flash-lite": dict(base=GEMINI, key="GEMINI_API_KEY", model="gemini-3.5-flash-lite", temp=1.0, conc=1, mode="tools"),
    "nemotron-3-ultra": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="nvidia/nemotron-3-ultra-550b-a55b:free", temp=0.6, conc=1),
    # Paid (off-peak 00:30-08:30 BJT is half price); use only to fill gaps.
    "deepseek-flash-paid": dict(base=DEEPSEEK, key="DEEPSEEK_API_KEY", model="deepseek-flash", temp=1.0, conc=4, paid=True),
}

TIER5 = [f.name for f in FAMILIES.values() if f.tier == 5]
TIER14 = [f.name for f in FAMILIES.values() if f.tier < 5]
STAGES = {
    "t5-cal": ("calibrated", TIER5),
    "t14-cal": ("calibrated", TIER14),
    "t5-neu": ("neutral", TIER5),
    "t14-neu": ("neutral", TIER14),
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def _quota_like(err: str) -> bool:
    e = err.lower()
    return any(k in e for k in ("429", "quota", "insufficient balance", "rate limit", "503", "402"))


def run_model(label: str, args: argparse.Namespace, secret: str | None) -> dict:
    cfg = MODELS[label]
    key = os.environ.get(cfg["key"])
    if not key:
        log(f"[{label}] {cfg['key']} not set; skipped")
        return {"label": label, "status": "no key"}
    agent = OpenAICompatibleAgent(
        model=cfg["model"],
        base_url=cfg["base"],
        api_key=key,
        tool_mode=cfg.get("mode", "code"),
        max_turns=args.max_turns,
        temperature=cfg["temp"],
        max_tokens=args.max_tokens,
        http_timeout_sec=300.0,
        extra_body=cfg.get("extra"),
    )
    tally: Counter = Counter()
    for stage in args.stages.split(","):
        variant, fams = STAGES[stage]
        out = Path(args.out) / f"{args.split}_{variant}"
        for fam_name in fams:
            _, outcomes = run_suite(
                [get_family(fam_name)],
                split=args.split,
                n=args.n,
                start=args.start,
                variant=variant,
                secret=secret,
                agent=agent,
                out_dir=out,
                job_id=label,
                concurrency=cfg["conc"],
                resume=True,
            )
            ran = [o for o in outcomes if not o.skipped]
            errs = [o for o in ran if o.error]
            tally["scored"] += sum(1 for o in outcomes if o.rewards is not None)
            tally["new"] += len(ran) - len(errs)
            tally["errors"] += len(errs)
            if ran:
                log(f"[{label}] {stage} {fam_name}: {len(ran) - len(errs)} new, {len(errs)} errors")
            if len(ran) >= 2 and len(errs) >= 0.6 * len(ran) and all(_quota_like(o.error) for o in errs):
                log(f"[{label}] quota or rate limit hit ({errs[-1].error[:160]}); parked until next run")
                return {"label": label, "status": "parked", **tally}
    return {"label": label, "status": "done", **tally}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of labels from MODELS")
    ap.add_argument("--stages", default="t5-cal,t14-cal,t5-neu,t14-neu")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--secret-file", default="pullgym_v02.secret")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default="runs/phase4")
    ap.add_argument("--allow-paid", action="store_true")
    args = ap.parse_args()

    labels = args.models.split(",")
    for lb in labels:
        if lb not in MODELS:
            ap.error(f"unknown model {lb!r}; known: {', '.join(MODELS)}")
        if MODELS[lb].get("paid") and not args.allow_paid:
            ap.error(f"{lb} is billed; pass --allow-paid")
    secret = Path(args.secret_file).read_text().strip() if args.split == "test" else None

    meta_path = Path(args.out) / "models.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    for lb in labels:
        c = MODELS[lb]
        meta[lb] = {"endpoint": c["base"], "model": c["model"], "temperature": c["temp"],
                    "tool_mode": c.get("mode", "code"), "max_turns": args.max_turns,
                    "max_tokens": args.max_tokens}
    meta_path.write_text(json.dumps(meta, indent=2))

    # One thread per model; models that share an endpoint run side by side.
    results: list[dict] = []
    threads = [threading.Thread(target=lambda lb=lb: results.append(run_model(lb, args, secret)))
               for lb in labels]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for r in results:
        log(f"summary {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
