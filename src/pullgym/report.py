"""Coverage and pull tables from Harbor job output.

Harbor writes one directory per trial:

    <trial>/config.json            trial config (task, trial name, job id)
    <trial>/result.json            agent name and model, under config.agent
    <trial>/trial.log
    <trial>/verifier/details.json  written by pullgym's scorer

Older Harbor versions put the agent block in config.json instead; both are read.

Episodes are grouped by any of: model, agent, family, variant, split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pullgym.stats import coverage_by_truth, summarize

COLUMNS = [
    ("n", "n", "{:.0f}"),
    ("valid", "valid_rate", "{:.0%}"),
    ("cov68", "cov68", "{:.1%}"),
    ("cov95", "cov95", "{:.1%}"),
    ("pull mean", "pull_mean", "{:+.2f}"),
    ("pull rms", "pull_rms", "{:.2f}"),
    ("pull width", "pull_width_robust", "{:.2f}"),
    ("sharpness", "sharpness_median", "{:.2f}"),
    ("reward", "reward_mean", "{:+.3f}"),
]


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def trial_agent(trial: Path) -> tuple[str, str]:
    """(agent name, model name) for a Harbor trial directory.

    A trial run without a model (such as Harbor's oracle) is labelled by its
    agent name in the model column.
    """
    result = _read_json(trial / "result.json")
    cfg = (result.get("config") or {}).get("agent") or {}
    if not cfg:
        cfg = _read_json(trial / "config.json").get("agent") or {}
    info = result.get("agent_info") or {}
    model_info = info.get("model_info") or {}
    agent = cfg.get("name") or info.get("name") or "unknown"
    model = cfg.get("model_name") or model_info.get("name") or agent
    return agent, model


def load_episodes(path: Path) -> tuple[list[dict], int]:
    """Returns (episode rows, number of trials that produced no verifier output)."""
    rows = []
    for details_path in sorted(path.rglob("verifier/details.json")):
        trial = details_path.parent.parent
        d = json.loads(details_path.read_text())
        agent, model = trial_agent(trial)
        for q, m in d["quantities"].items():
            rows.append(
                {
                    "model": model,
                    "agent": agent,
                    "family": d.get("family", "unknown"),
                    "variant": d.get("variant", "unknown"),
                    "split": d.get("split", "unknown"),
                    "quantity": q,
                    "valid": 1.0 if d.get("valid") else 0.0,
                    "truth": m["truth"],
                    **{
                        k: m[k]
                        for k in ("reward", "covered68", "covered95", "pull", "sharpness")
                    },
                }
            )
    trials = {p.parent for p in path.rglob("config.json") if (p.parent / "trial.log").exists()}
    missing = sum(1 for t in trials if not (t / "verifier" / "details.json").is_file())
    return rows, missing


def report(path: Path, group_by: str = "model,family,variant", truth_bins: int = 3) -> str:
    keys = [k.strip() for k in group_by.split(",") if k.strip()]
    rows, missing = load_episodes(path)
    if not rows:
        msg = f"No pullgym verifier output found under {path}."
        if missing:
            msg += (
                f" {missing} trial(s) ran but produced no verifier output; "
                "see exception.txt in each trial directory."
            )
        return msg
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)

    header = keys + [c[0] for c in COLUMNS]
    lines = []
    for g in sorted(groups):
        s = summarize(groups[g])
        cells = [str(v) for v in g]
        for _, key, fmt in COLUMNS:
            cells.append(fmt.format(s[key]) if key in s else "-")
        lines.append(cells)
    widths = [max(len(header[i]), *(len(ln[i]) for ln in lines)) for i in range(len(header))]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(c.ljust(w) for c, w in zip(ln, widths)) for ln in lines]
    if truth_bins > 1:
        out.append("")
        out.append(f"68% coverage in {truth_bins} equal-count bins of the true value:")
        for g in sorted(groups):
            bins = coverage_by_truth(groups[g], truth_bins)
            if bins:
                cells = "  ".join(f"[{b['lo']:.4g}, {b['hi']:.4g}] {b['cov68']:.0%} (n={b['n']})" for b in bins)
                out.append(f"  {' / '.join(map(str, g))}:  {cells}")
    out.append("")
    out.append(
        "Coverage, pull and sharpness use valid answers only; reward includes invalid "
        "answers. Calibrated: cov68 68%, cov95 95%, pull mean 0, pull width 1."
    )
    if missing:
        out.append(f"{missing} trial(s) produced no verifier output (agent or harness error).")
    return "\n".join(out)
