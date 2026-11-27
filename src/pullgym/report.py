"""Coverage and pull tables from Harbor job output.

Harbor writes one directory per trial:

    <trial>/config.json            trial config (task, trial name, job id)
    <trial>/result.json            agent name and model, under config.agent
    <trial>/trial.log
    <trial>/verifier/details.json  written by pullgym's scorer

Older Harbor versions put the agent block in config.json instead; both are read.

Episodes are grouped by any of: model, agent, family, tier, variant, split,
harness_version.

Paired inference: every episode is identified by its task key (family, split,
variant, task, quantity), so two models run on the same seeds can be compared
task by task. ``--paired`` restricts the main table to tasks every model
answered validly; ``--baseline M`` adds a table of paired differences of every
other model against M.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from pullgym.stats import clip_pinball, coverage_by_truth, mcnemar, paired_difference, summarize

FORMATS = ("text", "md", "latex", "json")
TASK_KEY = ("family", "split", "variant", "task", "quantity")


def _pct(v: float) -> str:
    return f"{v:.1%}"


def _ci(s: dict, key: str) -> str | None:
    lo, hi = s.get(f"{key}_ci_lo"), s.get(f"{key}_ci_hi")
    if lo is None or hi is None:
        return None
    return f"[{lo:.0%},{hi:.0%}]"


# (header, summary key, formatter). A formatter returning None prints "-".
COLUMNS = [
    ("n", "n", "{:.0f}".format),
    ("valid", "valid_rate", "{:.0%}".format),
    ("cov68", "cov68", _pct),
    ("cov68 CI", "cov68", None),  # Wilson 95%, filled in by _cells
    ("cov95", "cov95", _pct),
    ("pull mean", "pull_mean", "{:+.2f}".format),
    ("pull rms", "pull_rms", "{:.2f}".format),
    ("pull width", "pull_width_robust", "{:.2f}".format),
    ("sharp med", "sharpness_median", "{:.2f}".format),
    ("sharp gm", "sharpness_gmean", "{:.2f}".format),
    ("pinball", "pinball_skill_clipped_mean", "{:+.3f}".format),
    ("rew_v", "reward_valid_mean", "{:+.3f}".format),
    ("rew_all", "reward_mean", "{:+.3f}".format),
]

DIFF_HEADER = [
    "vs",
    "n_all",
    "d_valid",
    "d_rew_all [95% CI]",
    "n_valid",
    "d_rew_v [95% CI]",
    "d_cov68 ± se (+/-, p)",
    "d_cov95 ± se",
    "d_pinball",
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


def _tier(family: str, d: dict) -> int:
    from pullgym.families import FAMILIES

    if family in FAMILIES:
        return FAMILIES[family].tier
    return int(d.get("tier", 0) or 0)


def load_episodes(path: Path | list[Path]) -> tuple[list[dict], int]:
    """Returns (episode rows, number of trials that produced no verifier output).

    `path` may be one jobs directory or several (searched recursively).
    """
    paths = [Path(p) for p in path] if isinstance(path, (list, tuple)) else [Path(path)]
    rows = []
    details_files = sorted({p for root in paths for p in root.rglob("verifier/details.json")})
    for details_path in details_files:
        trial = details_path.parent.parent
        d = json.loads(details_path.read_text())
        agent, model = trial_agent(trial)
        harness = _read_json(trial / "result.json").get("harness_version") or "unknown"
        family = d.get("family", "unknown")
        for q, m in d["quantities"].items():
            row = {
                "model": model,
                "agent": agent,
                "family": family,
                "tier": _tier(family, d),
                "variant": d.get("variant", "unknown"),
                "split": d.get("split", "unknown"),
                "task": d.get("task", trial.name),
                "quantity": q,
                "harness_version": harness,
                "valid": 1.0 if d.get("valid") else 0.0,
                "truth": m["truth"],
                **{k: m[k] for k in ("reward", "covered68", "covered95", "pull", "sharpness")},
                # On invalid answers these describe the scored prior fallback,
                # which is what the *_all metrics in summarize() want.
                "pinball_skill": m.get("pinball_skill"),
                "ref_sigma": m.get("ref_sigma"),
            }
            answer = m.get("answer")
            if isinstance(answer, list) and len(answer) == 5 and m.get("ref_sigma"):
                row["q50"] = answer[2]
                row["pt_acc"] = 1.0 if abs(answer[2] - m["truth"]) <= m["ref_sigma"] else 0.0
            rows.append(row)
    trials = {
        p.parent
        for root in paths
        for p in root.rglob("config.json")
        if (p.parent / "trial.log").exists()
    }
    missing = sum(1 for t in trials if not (t / "verifier" / "details.json").is_file())
    return rows, missing


def task_key(row: dict) -> tuple:
    return tuple(row[k] for k in TASK_KEY)


def paired_subset(rows: list[dict]) -> list[dict]:
    """Rows whose task every loaded model answered validly."""
    models = {r["model"] for r in rows}
    valid_by_key: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if r["valid"] >= 1.0:
            valid_by_key[task_key(r)].add(r["model"])
    keep = {k for k, ms in valid_by_key.items() if ms == models}
    return [r for r in rows if task_key(r) in keep]


def paired_differences(rows: list[dict], model: str, baseline: str) -> dict:
    """Paired comparison of `model` against `baseline` on their shared tasks.

    rew_all uses every shared task (an invalid answer is scored as the prior
    fallback minus the penalty), so it has no survivorship bias. The valid-only
    numbers use tasks both models answered validly.
    """
    by_m = {task_key(r): r for r in rows if r["model"] == model}
    by_b = {task_key(r): r for r in rows if r["model"] == baseline}
    shared = sorted(set(by_m) & set(by_b))
    both_valid = [k for k in shared if by_m[k]["valid"] >= 1.0 and by_b[k]["valid"] >= 1.0]
    out = {
        "model": model,
        "baseline": baseline,
        "n_all": len(shared),
        "n_valid": len(both_valid),
        "valid": mcnemar([by_m[k]["valid"] for k in shared], [by_b[k]["valid"] for k in shared]),
        "reward_all": paired_difference(
            [by_m[k]["reward"] for k in shared], [by_b[k]["reward"] for k in shared]
        ),
        "reward_valid": paired_difference(
            [by_m[k]["reward"] for k in both_valid], [by_b[k]["reward"] for k in both_valid]
        ),
        "cov68": mcnemar(
            [by_m[k]["covered68"] for k in both_valid], [by_b[k]["covered68"] for k in both_valid]
        ),
        "cov95": mcnemar(
            [by_m[k]["covered95"] for k in both_valid], [by_b[k]["covered95"] for k in both_valid]
        ),
    }
    pk = [k for k in both_valid if by_m[k].get("pinball_skill") is not None
          and by_b[k].get("pinball_skill") is not None]
    if pk:
        out["pinball_valid"] = paired_difference(
            [clip_pinball(by_m[k]["pinball_skill"]) for k in pk],
            [clip_pinball(by_b[k]["pinball_skill"]) for k in pk],
        )
    return out


def _fmt_diff(d: dict, pct: bool = False) -> str:
    if not d.get("n") or not math.isfinite(d.get("mean", math.nan)):
        return "-"
    f = (lambda v: f"{100 * v:+.1f}pt") if pct else (lambda v: f"{v:+.2f}")
    if math.isfinite(d.get("ci_lo", math.nan)):
        return f"{f(d['mean'])} [{f(d['ci_lo'])},{f(d['ci_hi'])}]"
    return f(d["mean"])


def _fmt_mcnemar(d: dict, with_counts: bool = True) -> str:
    if not d.get("n"):
        return "-"
    s = f"{100 * d['diff']:+.1f}±{100 * d['se']:.1f}pt"
    if with_counts:
        s += f" ({d['a_only']}/{d['b_only']}, p={d['p']:.2g})"
    return s


def _diff_cells(d: dict) -> list[str]:
    return [
        d["baseline"],
        str(d["n_all"]),
        f"{100 * d['valid']['diff']:+.0f}pt" if d["n_all"] else "-",
        _fmt_diff(d["reward_all"]),
        str(d["n_valid"]),
        _fmt_diff(d["reward_valid"]),
        _fmt_mcnemar(d["cov68"]),
        _fmt_mcnemar(d["cov95"], with_counts=False),
        _fmt_diff(d["pinball_valid"]) if "pinball_valid" in d else "-",
    ]


def _cells(s: dict) -> list[str]:
    cells = []
    for header, key, fmt in COLUMNS:
        if header == "cov68 CI":
            cells.append(_ci(s, "cov68") or "-")
        elif key in s and s[key] is not None and not (
            isinstance(s[key], float) and math.isnan(s[key])
        ):
            cells.append(fmt(s[key]))
        else:
            cells.append("-")
    return cells


def _group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return groups


def _render(header: list[str], lines: list[list[str]], fmt: str) -> list[str]:
    if fmt == "md":
        out = ["| " + " | ".join(header) + " |"]
        out.append("|" + "|".join(" :--- " if i == 0 else " ---: " for i in range(len(header))) + "|")
        out += ["| " + " | ".join(c.replace("|", "\\|") for c in ln) + " |" for ln in lines]
        return out
    if fmt == "latex":
        esc = lambda t: (  # noqa: E731
            t.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("±", "$\\pm$")
        )
        out = ["\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\toprule"]
        out.append(" & ".join(esc(h) for h in header) + " \\\\")
        out.append("\\midrule")
        out += [" & ".join(esc(c) for c in ln) + " \\\\" for ln in lines]
        out += ["\\bottomrule", "\\end{tabular}"]
        return out
    widths = [max(len(header[i]), *(len(ln[i]) for ln in lines)) for i in range(len(header))]
    out = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(c.ljust(w) for c, w in zip(ln, widths)) for ln in lines]
    return out


def report(
    path: Path | list[Path],
    group_by: str = "model,family,variant",
    truth_bins: int = 3,
    *,
    paired: bool = False,
    baseline: str | None = None,
    fmt: str = "text",
    where: dict[str, set[str]] | None = None,
) -> str:
    """Leaderboard text for one or more jobs directories.

    `where` keeps only rows whose column values (compared as strings) are in the
    given sets, e.g. ``{"tier": {"5"}, "variant": {"calibrated"}}``.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; known: {', '.join(FORMATS)}")
    keys = [k.strip() for k in group_by.split(",") if k.strip()]
    rows, missing = load_episodes(path)
    if where:
        rows = [r for r in rows if all(str(r.get(k)) in vals for k, vals in where.items())]
    if not rows:
        where = ", ".join(map(str, path)) if isinstance(path, (list, tuple)) else path
        msg = f"No pullgym verifier output found under {where}."
        if missing:
            msg += (
                f" {missing} trial(s) ran but produced no verifier output; "
                "see exception.txt in each trial directory."
            )
        return msg
    unknown = [k for k in keys if k not in rows[0]]
    if unknown:
        raise ValueError(f"unknown grouping key(s): {', '.join(unknown)}")

    models = sorted({r["model"] for r in rows})
    if baseline is None and paired and len(models) == 2:
        baseline = models[0]
    if baseline is not None and baseline not in models:
        raise ValueError(f"baseline {baseline!r} not found; models: {', '.join(models)}")

    table_rows = paired_subset(rows) if paired else rows
    groups = _group(table_rows, keys)
    summaries = {g: summarize(groups[g]) for g in sorted(groups)}

    # Paired differences, grouped by every key except the model.
    diff_keys = [k for k in keys if k not in ("model", "agent")]
    diffs: list[tuple[tuple, dict]] = []
    if baseline is not None:
        for g, grp in sorted(_group(rows, diff_keys).items()):
            for m in models:
                if m != baseline:
                    d = paired_differences(grp, m, baseline)
                    if d["n_all"]:
                        diffs.append((g, d))

    versions = sorted({r["harness_version"] for r in rows})
    warnings = []
    if len(versions) > 1:
        warnings.append(
            "WARNING: trials span several harness versions ("
            + ", ".join(versions)
            + "; 'unknown' = recorded before 0.2.0.dev0). Do not mix them in one leaderboard."
        )
    if missing:
        warnings.append(f"{missing} trial(s) produced no verifier output (agent or harness error).")

    if fmt == "json":
        return json.dumps(
            {
                "group_by": keys,
                "paired": paired,
                "baseline": baseline,
                "groups": [dict(zip(keys, g), **s) for g, s in summaries.items()],
                "paired_differences": [dict(zip(diff_keys, g), **d) for g, d in diffs],
                "harness_versions": versions,
                "warnings": warnings,
            },
            indent=2,
            default=float,
        )

    header = keys + [c[0] for c in COLUMNS]
    lines = [[str(v) for v in g] + _cells(s) for g, s in summaries.items()]
    out = []
    if paired:
        out.append(f"Paired: {len(table_rows)} of {len(rows)} rows, on tasks every model answered validly.")
        out.append("")
    out += _render(header, lines, fmt)

    if diffs:
        out.append("")
        out.append(f"Paired differences (model - {baseline}) on shared tasks:")
        out.append("")
        d_header = diff_keys + ["model"] + DIFF_HEADER
        d_lines = [[str(v) for v in g] + [d["model"]] + _diff_cells(d) for g, d in diffs]
        out += _render(d_header, d_lines, fmt)

    if fmt == "latex":
        return "\n".join(out)

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
        "Coverage, pull, sharpness, pinball and rew_v use valid answers only; rew_all "
        "includes invalid answers (scored as the prior minus 1). cov68 CI is a Wilson "
        "95% interval. sharp gm is the geometric mean of sharpness; pinball is the mean "
        "pinball skill clipped at -20. Calibrated: cov68 68%, cov95 95%, pull mean 0, "
        "pull width 1, sharpness 1."
    )
    if diffs:
        out.append(
            "d_rew_all compares every shared task; the other differences use tasks both "
            "models answered validly. CIs are paired bootstrap 95%; (+/-) counts tasks "
            "covered by only the model / only the baseline, with the exact McNemar p."
        )
    out += warnings
    return "\n".join(out)
