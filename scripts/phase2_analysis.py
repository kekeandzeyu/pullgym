"""Summarise Phase 2 (2A repair diagnostic, 2B run-fast transfer) with paired statistics.

    python scripts/phase2_analysis.py --phase2 runs/phase2 --r2 runs/eval_runs runs/eval_runs_extra
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pullgym.report import load_episodes, paired_differences, report
from pullgym.stats import summarize

MODELS = ("qwen3-8b-base", "qwen3-8b_binary_full_r2", "qwen3-8b_log_score_full_r2")


def _r2_job(r2_roots: list[Path], model: str, set_name: str) -> Path | None:
    """Round 2 job directory for a model and evaluation set (two naming styles exist)."""
    for root in r2_roots:
        for name in (f"{model}_{set_name}", f"{model.replace('-base', '_base')}_{set_name}"):
            if (root / name).is_dir():
                return root / name
    return None


def _assistant_text(trial: Path, turn: int = 1) -> str | None:
    try:
        msgs = json.loads((trial / "transcript.json").read_text())
    except (OSError, ValueError):
        return None
    texts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return texts[turn - 1] if len(texts) >= turn else None


def _relabel(rows: list[dict], label: str) -> list[dict]:
    """Copies of the rows with the model column replaced (inputs are not modified)."""
    return [dict(r, model=label) for r in rows]


def _fmt_diff(d: dict) -> str:
    ra, rv, c = d["reward_all"], d["reward_valid"], d["cov68"]
    return (
        f"n={d['n_all']:3d} d_valid={100 * d['valid']['diff']:+5.1f}pt "
        f"(+{d['valid']['a_only']}/-{d['valid']['b_only']})  "
        f"d_rew_all={ra['mean']:+.2f} [{ra['ci_lo']:+.2f},{ra['ci_hi']:+.2f}]  "
        f"n_v={d['n_valid']:3d} d_rew_v={rv['mean']:+.2f} [{rv.get('ci_lo', float('nan')):+.2f},"
        f"{rv.get('ci_hi', float('nan')):+.2f}]  d_cov68={100 * c['diff']:+.1f}±{100 * c['se']:.1f}pt "
        f"({c['a_only']}/{c['b_only']}, p={c['p']:.2g})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", default="runs/phase2")
    ap.add_argument("--r2", nargs="*", default=["runs/eval_runs", "runs/eval_runs_extra"])
    ap.add_argument("--sets", default="id_calibrated,ood_calibrated,ood_neutral")
    args = ap.parse_args()
    root = Path(args.phase2)
    r2_roots = [Path(p) for p in args.r2]
    sets = args.sets.split(",")

    print("# Phase 2A: same-prompt repair (turn 1 = Round 2 single-turn prompt)\n")
    for set_name in sets:
        print(f"## {set_name}\n")
        # 1. Reproduction of Round 2 by turn 1.
        for m in MODELS:
            k1 = root / "2A" / set_name / f"{m}_K1"
            r2 = _r2_job(r2_roots, m, set_name)
            if not k1.is_dir() or r2 is None:
                continue
            same = total = 0
            for t in k1.iterdir():
                a, b = _assistant_text(t), _assistant_text(r2 / t.name)
                if a is not None and b is not None:
                    total += 1
                    same += a.strip() == b.strip()
            rows = _relabel(load_episodes(k1)[0], "K1 now") + _relabel(load_episodes(r2)[0], "R2 then")
            d = paired_differences(rows, "K1 now", "R2 then")
            print(f"- reproduce {m}: identical turn-1 text {same}/{total}; K1 now - R2 then: {_fmt_diff(d)}")
        print()
        # 2. Effect of repair turns, per model.
        all_rows = []
        for m in MODELS:
            ks = sorted((root / "2A" / set_name).glob(f"{m}_K*"))
            if len(ks) < 2:
                continue
            rows = []
            for d_ in ks:
                rows += load_episodes(d_)[0]
            all_rows += rows
            k_hi = ks[-1].name.split("_K")[-1]
            d = paired_differences(rows, f"{m}|K{k_hi}", f"{m}|K1")
            print(f"- repair {m}: K{k_hi} - K1: {_fmt_diff(d)}")
        print()
        # 3. Model comparisons at K1 and at K_max.
        for k in ("K1", "K3"):
            sub = [r for r in all_rows if r["model"].endswith(f"|{k}")]
            if not sub:
                continue
            for m, base in (("qwen3-8b_log_score_full_r2", "qwen3-8b_binary_full_r2"),
                            ("qwen3-8b_log_score_full_r2", "qwen3-8b-base"),
                            ("qwen3-8b_binary_full_r2", "qwen3-8b-base")):
                d = paired_differences(sub, f"{m}|{k}", f"{base}|{k}")
                if d["n_all"]:
                    print(f"- {k}: {m.replace('qwen3-8b_', '')} - {base.replace('qwen3-8b_', '')}: {_fmt_diff(d)}")
        print()
        # 4. What failed on turn 1, and what repair fixed.
        for m in MODELS:
            p = root / "2A" / set_name / f"{m}_turns.jsonl"
            if not p.is_file():
                continue
            first, fixed = Counter(), Counter()
            for line in p.read_text().splitlines():
                e = json.loads(line)
                s1 = e["status"][0]
                if s1 == "valid":
                    continue
                kind = ("no code block" if s1 == "no code" else
                        "crash, no answer" if s1 == "missing" and e["exit_code"][0] not in (0, None) else
                        "ran, no answer" if s1 == "missing" else
                        "quantile order" if "strictly less" in s1 else
                        "schema/JSON" if s1.startswith("invalid") else s1)
                first[kind] += 1
                fixed[kind] += e["status"][-1] == "valid"
            if first:
                parts = ", ".join(f"{k}: {fixed[k]}/{v} fixed" for k, v in first.most_common())
                print(f"- turn-1 failures {m}: {parts}")
        print()

    print("# Phase 2B: run-fast code mode, max_turns=5 (no previews in turn 1)\n")
    for set_name in sets:
        jobs = [root / "2B" / set_name / m for m in MODELS if (root / "2B" / set_name / m).is_dir()]
        if not jobs:
            continue
        print(f"## {set_name}\n")
        print("```")
        print(report(jobs, group_by="model", truth_bins=1, baseline="qwen3-8b_binary_full_r2"))
        print("```")
        rows = []
        for j in jobs:
            rows += load_episodes(j)[0]
        # 2B vs 2A K3, same model and tasks
        for m in MODELS:
            k3 = root / "2A" / set_name / f"{m}_K3"
            if k3.is_dir():
                a = _relabel([r for r in rows if r["model"] == m], "2B")
                b = _relabel(load_episodes(k3)[0], "2A-K3")
                d = paired_differences(a + b, "2B", "2A-K3")
                print(f"- {m}: run-fast(5 turns) - 2A K3: {_fmt_diff(d)}")
        s = {m: summarize([r for r in rows if r["model"] == m]) for m in MODELS}
        print("- valid rates:", {m: f"{v['valid_rate']:.0%}" for m, v in s.items() if v.get("n")})
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
