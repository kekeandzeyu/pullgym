#!/usr/bin/env python3
"""Phase 5 tables: base-mt vs binary-mt vs log_score-mt, paired by task.

Every row of `runs/phase5/eval/<set>_<variant>/draw<d>/<model>/...` is one
episode. Models are mapped to arms (`qwen3-8b-base` -> base, `*_binary_mt_s<k>`
-> binary, `*_log_score_mt_s<k>` -> log_score). For the paired tests, each arm's
score on a task is its mean over all training seeds and draws, so evaluation
sampling noise and seed noise are averaged before tasks are resampled. The
bootstrap (10k, fixed seed) runs over tasks.

    python scripts/phase5_analysis.py [--root runs/phase5/eval] [--format md]
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from pullgym.report import load_episodes
from pullgym.rl import OOD_V01_FAMILIES, OOD_V02_FAMILIES, TRAIN_FAMILIES
from pullgym.stats import paired_difference, summarize

ARM_RE = re.compile(r"_(binary|log_score)_mt_s(\d+)$")
GROUPS = {
    "ID (8 training families)": TRAIN_FAMILIES,
    "OOD v0.1 (4 families)": OOD_V01_FAMILIES,
    "OOD v0.2 Tier 5 (4 families)": OOD_V02_FAMILIES,
}
TIER1 = ("line_fit", "weighted_mean", "counting_rate")
PAIRS = [("log_score", "binary"), ("log_score", "base"), ("binary", "base")]


def arm_of(model: str) -> tuple[str, str]:
    m = ARM_RE.search(model)
    if m:
        return m.group(1), m.group(2)
    return ("base", "-") if "base" in model else (model, "-")


def _f(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:+.{nd}f}" if nd == 3 else f"{v:.{nd}f}"


def table(header, lines, fmt):
    if fmt == "md":
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(l) + " |" for l in lines]
    else:
        w = [max(len(str(x)) for x in col) for col in zip(header, *lines)]
        out = ["  ".join(str(x).ljust(k) for x, k in zip(r, w)) for r in [header, *lines]]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/phase5/eval")
    ap.add_argument("--format", default="text", choices=["text", "md"])
    args = ap.parse_args()
    root = Path(args.root)

    for set_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows, missing = load_episodes(set_dir)
        if not rows:
            continue
        for r in rows:
            r["arm"], r["train_seed"] = arm_of(r["model"])
        print(f"\n## {set_dir.name}  ({len(rows)} episodes, {missing} without verifier output)\n")
        for gname, fams in GROUPS.items():
            g = [r for r in rows if r["family"] in fams]
            if not g:
                continue
            print(f"### {gname}\n")
            # Pooled per arm, and per training seed (seed spread).
            lines = []
            by = defaultdict(list)
            for r in g:
                by[(r["arm"], "all")].append(r)
                by[(r["arm"], r["train_seed"])].append(r)
            for (arm, seed), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "all", kv[0][1])):
                if seed != "all" and arm == "base":
                    continue
                s = summarize(rs)
                lines.append([
                    arm if seed == "all" else f"  seed {seed}", str(s["n"]), _f(s["valid_rate"], True),
                    _f(s.get("reward_mean")), _f(st.median(r["reward"] for r in rs)),
                    _f(s.get("reward_valid_mean")), _f(s.get("cov68"), True), _f(s.get("cov95"), True),
                    _f(s.get("sharpness_gmean"), nd=2), _f(s.get("pinball_skill_clipped_mean")),
                ])
            print(table(["arm", "n", "valid", "rew_all", "rew_med", "rew_v", "cov68", "cov95",
                         "sharp_g", "pinball"], lines, args.format))
            # Paired by task: mean over seeds x draws per arm.
            per_task = defaultdict(lambda: defaultdict(list))
            for r in g:
                per_task[(r["family"], r["task"])][r["arm"]].append(r)
            lines = []
            for a, b in PAIRS:
                keys = [k for k, v in per_task.items() if a in v and b in v]
                if len(keys) < 2:
                    continue
                for metric, fn in (
                    ("rew_all", lambda rs: np.mean([r["reward"] for r in rs])),
                    ("valid", lambda rs: np.mean([r["valid"] for r in rs])),
                ):
                    d = paired_difference([fn(per_task[k][a]) for k in keys],
                                          [fn(per_task[k][b]) for k in keys])
                    pct = metric == "valid"
                    lines.append([f"{a} - {b}", metric, str(d["n"]), _f(d["mean"], pct),
                                  f"[{_f(d['ci_lo'], pct)}, {_f(d['ci_hi'], pct)}]"])
            if lines:
                print()
                print(table(["pair", "metric", "tasks", "mean diff", "95% CI"], lines, args.format))
            print()
        # Width-inflation audit on ID Tier 1.
        t1 = [r for r in rows if r["family"] in TIER1]
        if t1:
            lines = []
            for arm in sorted({r["arm"] for r in t1}):
                cells = [arm]
                for fam in TIER1:
                    s = summarize([r for r in t1 if r["arm"] == arm and r["family"] == fam])
                    cells.append(_f(s.get("sharpness_gmean"), nd=2))
                lines.append(cells)
            print("### Tier-1 geometric-mean sharpness (should stay near 1.00)\n")
            print(table(["arm", *TIER1], lines, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
