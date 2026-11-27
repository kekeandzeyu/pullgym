"""Phase 4 tables: frontier matrix vs oracle, Tier 5 vs Tiers 1-4, cost, failure modes.

    uv run python scripts/phase4_analysis.py [--root runs/phase4] [--format md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from pullgym.report import report


def _failure_kind(trial: Path, result: dict) -> str:
    if result.get("api_error"):
        return "provider rejected output"
    lines = (trial / "trial.log").read_text().strip().splitlines() if (trial / "trial.log").is_file() else []
    last = lines[-1] if lines else ""
    turns = result.get("usage", {}).get("turns", 0)
    if "finished" in last:
        return "stopped without valid answer"
    return f"turn budget exhausted ({turns})" if turns else "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phase4")
    ap.add_argument("--format", default="text", choices=["text", "md"])
    args = ap.parse_args()
    root = Path(args.root)
    cal, neu = root / "dev_calibrated", root / "dev_neutral"
    both = [p for p in (cal, neu) if p.is_dir()]
    fmt = args.format

    def section(title: str, body: str) -> None:
        print(f"\n## {title}\n")
        print(body if fmt == "md" else f"```\n{body}\n```")

    section("Tier 5, calibrated prompt: models vs oracle (paired on shared seeds)",
            report([cal], group_by="model", truth_bins=1, baseline="oracle", fmt=fmt,
                   where={"tier": {"5"}, "variant": {"calibrated"}}))
    section("Tier 5 by family, calibrated",
            report([cal], group_by="family,model", truth_bins=1, fmt=fmt,
                   where={"tier": {"5"}, "variant": {"calibrated"}}))
    if neu.is_dir():
        section("Tier 5, calibrated vs neutral prompt",
                report(both, group_by="model,variant", truth_bins=1, fmt=fmt, where={"tier": {"5"}}))
    seeds = {f"{s}" for s in range(5)}
    rows_t14 = {"tier": {"1", "2", "3", "4"}, "variant": {"calibrated"}}
    section("Tiers 1-4, calibrated, seeds 0-4 (saturation check, paired with oracle)",
            report([cal], group_by="model", truth_bins=1, baseline="oracle", fmt=fmt,
                   where={**rows_t14, "task": {t for t in _tasks(cal, seeds)}}))

    # Failure modes and token usage per model (calibrated + neutral).
    fails: dict[str, Counter] = defaultdict(Counter)
    usage: dict[str, list[dict]] = defaultdict(list)
    for p in both:
        for res_path in p.glob("*/*/result.json"):
            r = json.loads(res_path.read_text())
            model = r.get("agent_info", {}).get("model_info", {}).get("name") or res_path.parent.parent.name
            usage[model].append(r.get("usage") or {})
            if not (r.get("rewards") or {}).get("valid"):
                fails[model][_failure_kind(res_path.parent, r)] += 1
            else:
                fails[model]["valid"] += 1
    lines = []
    for m in sorted(usage):
        u = usage[m]
        pt = st.median(x.get("prompt_tokens", 0) for x in u)
        ct = st.median(x.get("completion_tokens", 0) for x in u)
        cache = sum(x.get("cached_tokens", 0) for x in u) / max(sum(x.get("prompt_tokens", 0) for x in u), 1)
        turns = st.median(x.get("turns", 0) for x in u)
        f = ", ".join(f"{k}: {v}" for k, v in fails[m].most_common())
        lines.append(f"{m}: n={len(u)} median turns {turns:.0f}, prompt {pt / 1e3:.0f}k "
                     f"({cache:.0%} cached), completion {ct / 1e3:.1f}k | {f}")
    section("Outcomes and token usage per model", "\n".join(lines))
    return 0


def _tasks(job_root: Path, seeds: set[str]) -> set[str]:
    """Task names (as stored in details.json) whose dev seed index is in `seeds`."""
    out = set()
    for d in job_root.glob("*/*/verifier/details.json"):
        task = json.loads(d.read_text()).get("task", d.parent.parent.name)
        if str(int(task.rsplit("-", 1)[-1])) in seeds:
            out.add(task)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
