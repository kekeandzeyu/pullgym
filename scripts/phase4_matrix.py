"""Phase 4 frontier matrix on free and low-cost endpoints, budget-capped and resumable.

    set -a; . ./api_keys.secret; set +a
    caffeinate -i uv run python scripts/phase4_matrix.py --models deepseek-v4.1-flash-eco,glm-5.3-flash
    uv run python scripts/phase4_matrix.py --spend      # spend so far per model, no API calls

Each model has a plan of (stage, seeds per family) and, for billed endpoints, a hard
spending cap in CNY. Spend is computed from the token usage recorded in every
``result.json`` (cached prompt tokens at the cached price), so it survives restarts.
The cap is checked before every chunk of ``conc`` episodes; a model stops once its
spend reaches the cap. Seeds are run chunk-major across families, so a model that
stops early still has the same seeds in every family (paired with the oracle).

Every job runs with ``resume=True`` into ``runs/phase4/<split>_<variant>/<label>``.
If most episodes of a chunk end in an API exception that looks like a quota or rate
limit, the model is parked for this invocation; those episodes have no
``reward.json``, are never scored, and are retried on the next run. Model-caused
provider errors after turn 1 (HTTP 400) are scored by the runner.

All rows use the same harness: ``tools`` mode (OpenAI function calling), 20 turns.
An earlier ``code``-mode attempt (archived in ``runs/phase4/codemode_v1``) failed for
harness reasons on tool-native reasoning models (code placed in the reasoning
field, pseudo tool-call markup in plain text, empty content), so it is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pullgym.families import FAMILIES, get_family
from pullgym.runner import OpenAICompatibleAgent, run_suite

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
ARK = "https://ark.cn-beijing.volces.com/api/v3"  # Volcengine Ark, pay as you go
ZHIPU = "https://open.bigmodel.cn/api/paas/v4"
MS = "https://api-inference.modelscope.cn/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
# Hard ceiling on the OpenRouter key's real usage (USD, read from /api/v1/key before
# every chunk). The key itself is limited to $20 by OpenRouter; this leaves room for
# chunks already in flight.
OR_TOTAL_CAP_USD = 18.5
_OR_BLOCKED_LOC = {"CN", "HK", "MO"}

TIER5 = [f.name for f in FAMILIES.values() if f.tier == 5]
TIER14 = [f.name for f in FAMILIES.values() if f.tier < 5]
# stage -> (variant, families)
STAGES = {
    "t5-cal": ("calibrated", TIER5),
    "t5-neu": ("neutral", TIER5),
    "t14-cal": ("calibrated", TIER14),
    "t14-neu": ("neutral", TIER14),
}
# Reduced Phase 4 design, about 260 episodes. Tiers 1-4 are saturated in v0.1;
# 5 seeds/family re-check that under the v0.2 harness, paired with the oracle.
REDUCED = [("t5-cal", 25), ("t5-neu", 25), ("t14-cal", 5)]

# CNY per million tokens: (input, cached input, output); Ark price page 2026-11.
# Peak prices apply Mon-Fri 09-12 and 14-18 Beijing time where two are listed.
ARK_PRICES = {
    "deepseek-v4-1-flash-economy-261017": {"offpeak": (0.8, 0.016, 3.2), "peak": (1.6, 0.032, 6.4)},
    "deepseek-v4-1-flash-261017": {"offpeak": (1.0, 0.02, 4.0), "peak": (2.0, 0.04, 8.0)},
    "deepseek-v4-pro-ga-260813": {"flat": (9.0, 0.30, 27.0)},
    "kimi-k3-1-pro-261024": {"flat": (13.0, 1.30, 68.0)},
    "glm-5-2-260617": {"flat": (8.0, 2.0, 28.0)},
    "glm-5-3-flash-260828": {"flat": (0.8, 0.23, 2.8)},
}

# label -> endpoint settings. Temperatures follow each vendor's recommendation.
MODELS: dict[str, dict] = {
    "deepseek-v4.1-flash-eco": dict(base=ARK, key="ARK_API_KEY", model="deepseek-v4-1-flash-economy-261017",
                                    temp=1.0, conc=12, cap=45.0, plan=REDUCED),
    "glm-5.3-flash": dict(base=ARK, key="ARK_API_KEY", model="glm-5-3-flash-260828",
                          temp=1.0, conc=12, cap=40.0, plan=REDUCED),
    "deepseek-v4-pro": dict(base=ARK, key="ARK_API_KEY", model="deepseek-v4-pro-ga-260813",
                            temp=1.0, conc=4, cap=60.0, plan=[("t5-cal", 15)]),
    "kimi-k3.1-pro": dict(base=ARK, key="ARK_API_KEY", model="kimi-k3-1-pro-261024",
                          temp=1.0, conc=4, cap=100.0, plan=[("t5-cal", 12)]),
    # OpenRouter (prepaid, USD). New accounts have a per-model requests-per-minute
    # limit, so concurrency is 1-2. Cost is the exact usage.cost OpenRouter reports.
    # Pilot costs per Tier-5 episode: Haiku $0.002, Luna $0.007, Sonnet $0.04,
    # GPT-6.2 Sol $0.12, Gemini 4 Hydrogen $0.33 (dropped: too few episodes per $).
    "claude-haiku-5.5": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="anthropic/claude-haiku-5.5",
                             temp=1.0, conc=4, cap_usd=1.5, plan=REDUCED, cache=True,
                             extra={"usage": {"include": True}}),
    "gpt-6-luna": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="openai/gpt-6-luna",
                       temp=1.0, conc=4, cap_usd=2.5, plan=REDUCED,
                       extra={"usage": {"include": True}}),
    "claude-sonnet-5.5": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="anthropic/claude-sonnet-5.5",
                              temp=1.0, conc=2, cap_usd=5.5, plan=[("t5-cal", 25), ("t5-neu", 25)], cache=True,
                              extra={"usage": {"include": True}}),
    "gpt-6.2-sol": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="openai/gpt-6.2-sol",
                        temp=1.0, conc=2, cap_usd=4.5, plan=[("t5-cal", 25)],
                        extra={"usage": {"include": True}}),
    "gemini-3.8-flash": dict(base=OPENROUTER, key="OPENROUTER_API_KEY", model="google/gemini-3.8-flash",
                             temp=1.0, conc=2, cap_usd=2.5, plan=[("t5-cal", 25)],
                             extra={"usage": {"include": True}}),
    "gemini-3.5-flash-lite": dict(base=GEMINI, key="GEMINI_API_KEY", model="gemini-3.5-flash-lite",
                                  temp=1.0, conc=1, cap=None, plan=REDUCED),
}

_print_lock = threading.Lock()
_BJT = ZoneInfo("Asia/Shanghai")


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def _is_peak(ts: float) -> bool:
    t = datetime.fromtimestamp(ts, _BJT)
    return t.weekday() < 5 and (9 <= t.hour < 12 or 14 <= t.hour < 18)


def episode_cost(model_id: str, usage: dict, ts: float) -> float:
    prices = ARK_PRICES.get(model_id)
    if not prices:
        return 0.0
    p_in, p_cached, p_out = prices.get("flat") or prices["peak" if _is_peak(ts) else "offpeak"]
    cached = int(usage.get("cached_tokens") or 0)
    fresh = max(int(usage.get("prompt_tokens") or 0) - cached, 0)
    return (fresh * p_in + cached * p_cached + int(usage.get("completion_tokens") or 0) * p_out) / 1e6


def spend(label: str, out: Path) -> tuple[float, int]:
    """CNY spent so far by a model (list prices, before any free quota) and episode count."""
    model_id = MODELS[label]["model"]
    total, n = 0.0, 0
    files = [f for split in ("dev", "test") for v in ("calibrated", "neutral")
             for f in (out / f"{split}_{v}" / label).glob("*/result.json")]
    for f in files:
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        u = r.get("usage") or {}
        total += float(u.get("cost") or 0.0) if MODELS[label].get("cap_usd") else episode_cost(model_id, u, f.stat().st_mtime)
        n += 1
    return total, n


def openrouter_guard(key: str) -> str | None:
    """Read-only pre-chunk check for OpenRouter: exit location and real key usage.

    Returns a reason to stop, or None. Never changes any network setting.
    """
    import urllib.request

    loc, last_err = "", None
    for attempt in range(3):  # a single TLS timeout through the proxy is common
        try:
            trace = urllib.request.urlopen("https://openrouter.ai/cdn-cgi/trace", timeout=30).read().decode()
            loc = next((ln[4:] for ln in trace.splitlines() if ln.startswith("loc=")), "")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(10 * (attempt + 1))
    else:
        return f"could not verify exit location after 3 tries ({last_err})"
    if not loc or loc in _OR_BLOCKED_LOC:
        return f"exit location is {loc or 'unknown'}; proxy not active"
    usage = None
    for attempt in range(3):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                         headers={"Authorization": f"Bearer {key}"})
            usage = float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(10 * (attempt + 1))
    if usage is None:
        return f"could not read key usage after 3 tries ({last_err})"
    if usage >= OR_TOTAL_CAP_USD:
        return f"OpenRouter key usage ${usage:.2f} >= ${OR_TOTAL_CAP_USD:.2f}"
    return None


def _quota_like(err: str) -> bool:
    e = err.lower()
    return any(k in e for k in ("429", "quota", "insufficient balance", "rate limit", "503", "402",
                                "accountoverdue", "arrear"))


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
        tool_mode="tools",
        max_turns=args.max_turns,
        temperature=cfg["temp"],
        max_tokens=args.max_tokens,
        http_timeout_sec=300.0,
        extra_body=cfg.get("extra"),
        cache_breakpoint=cfg.get("cache", False),
    )
    out = Path(args.out)
    cap = cfg.get("cap_usd") or cfg.get("cap")
    if cap is not None and args.cap_scale is not None:
        cap *= args.cap_scale
    unit = "USD" if cfg.get("cap_usd") else "CNY"
    tally: Counter = Counter()
    quota_streak = 0  # consecutive quota-like failures across chunks
    plan = args.plan or cfg["plan"]
    for stage, n_seeds in plan:
        variant, fams = STAGES[stage]
        job_root = out / f"{args.split}_{variant}"
        conc = cfg["conc"]
        # A chunk spans every family of the stage (about 2x conc episodes), so all
        # workers stay busy instead of waiting for one family's slowest episode.
        # Seeds still advance together across families.
        step = max(1, -(-2 * conc // len(fams)))
        for chunk_start in range(args.start, args.start + n_seeds, step):
            chunk_n = min(step, args.start + n_seeds - chunk_start)
            if cfg["base"] == OPENROUTER:
                reason = openrouter_guard(key)
                if reason:
                    log(f"[{label}] OpenRouter guard: {reason}; stopping")
                    return {"label": label, "status": "guard", **tally}
            if cap is not None:
                spent, _ = spend(label, out)
                if spent >= cap:
                    log(f"[{label}] budget cap reached: {spent:.2f} >= {cap:.2f} {unit}; stopping")
                    return {"label": label, "status": "cap", "spent": round(spent, 2), **tally}
            _, outcomes = run_suite(
                [get_family(f) for f in fams],
                split=args.split,
                n=chunk_n,
                start=chunk_start,
                variant=variant,
                secret=secret,
                agent=agent,
                out_dir=job_root,
                job_id=label,
                concurrency=conc,
                resume=True,
            )
            ran = [o for o in outcomes if not o.skipped]
            errs = [o for o in ran if o.error]
            tally["new"] += len(ran) - len(errs)
            tally["errors"] += len(errs)
            if ran:
                spent, n_ep = spend(label, out)
                log(f"[{label}] {stage} seeds {chunk_start}-{chunk_start + chunk_n - 1} x {len(fams)} families: "
                    f"{len(ran) - len(errs)} new, {len(errs)} errors; spend {spent:.2f} {unit} over {n_ep} episodes")
            for o in errs:
                log(f"[{label}]   error {o.task_name}: {o.error[:200]}")
            if errs and len(errs) == len(ran) and all(_quota_like(o.error) for o in errs):
                quota_streak += len(errs)
            elif ran:
                quota_streak = 0
            if quota_streak >= 3:
                log(f"[{label}] quota, balance or rate limit hit; parked until next run")
                return {"label": label, "status": "parked", **tally}
    return {"label": label, "status": "done", **tally}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma list of labels from MODELS")
    ap.add_argument("--spend", action="store_true", help="print spend so far and exit")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--secret-file", default="pullgym_v02.secret")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--plan", help="override every model's plan, e.g. t5-cal:25,t5-neu:10")
    ap.add_argument("--cap-scale", type=float, help="multiply every budget cap (e.g. 0.1 for a smoke run)")
    ap.add_argument("--out", default="runs/phase4")
    args = ap.parse_args()

    if args.spend:
        total = 0.0
        for lb in MODELS:
            s, n = spend(lb, Path(args.out))
            usd = MODELS[lb].get("cap_usd")
            cap = usd or MODELS[lb].get("cap")
            unit = "USD" if usd else "CNY"
            if not usd:
                total += 0  # CNY and USD are reported separately
            print(f"{lb:26s} {n:4d} episodes  {s:7.2f} {unit}" + (f"  (cap {cap:.1f})" if cap else ""))
        print("CNY rows use Ark list prices (free quotas not deducted); USD rows are OpenRouter's billed cost.")
        return 0
    if not args.models:
        ap.error("--models is required")
    labels = args.models.split(",")
    for lb in labels:
        if lb not in MODELS:
            ap.error(f"unknown model {lb!r}; known: {', '.join(MODELS)}")
        c = MODELS[lb]
        missing = [k for k in ("base", "key", "model", "temp", "conc", "plan") if k not in c]
        if missing:
            ap.error(f"{lb}: config is missing {missing}")
        if c["base"] in (ARK, OPENROUTER) and not (c.get("cap") or c.get("cap_usd")):
            ap.error(f"{lb}: billed endpoint without a budget cap")
    if args.plan:
        try:
            args.plan = [(st_.split(":")[0], int(st_.split(":")[1])) for st_ in args.plan.split(",")]
        except (IndexError, ValueError):
            ap.error("--plan expects STAGE:N[,STAGE:N]")
        bad = [st_ for st_, _ in args.plan if st_ not in STAGES]
        if bad:
            ap.error(f"unknown stage(s) {bad}; known: {', '.join(STAGES)}")
    secret = Path(args.secret_file).read_text().strip() if args.split == "test" else None

    meta_path = Path(args.out) / "models.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    for lb in labels:
        c = MODELS[lb]
        meta[lb] = {"endpoint": c["base"], "model": c["model"], "temperature": c["temp"],
                    "tool_mode": "tools", "max_turns": args.max_turns, "max_tokens": args.max_tokens,
                    "plan": c["plan"], "cap_cny": c.get("cap"), "cap_usd": c.get("cap_usd"),
                    "cache_breakpoint": c.get("cache", False)}
    meta_path.write_text(json.dumps(meta, indent=2))

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
