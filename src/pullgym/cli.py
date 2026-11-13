"""Command-line interface: list, export, calibrate, probe, report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pullgym.families import FAMILIES, get_family


def _families(arg: str):
    return list(FAMILIES.values()) if arg == "all" else [get_family(n) for n in arg.split(",")]


def _fmt_summary(s: dict) -> str:
    if "cov68" not in s:
        return f"n={s['n']}  no valid answers"
    return (
        f"n={s['n']:<5d} valid={s['valid_rate']:.0%}  "
        f"cov68={s['cov68']:.1%}±{s['cov68_se']:.1%}  "
        f"cov95={s['cov95']:.1%}±{s['cov95_se']:.1%}  "
        f"pull mean={s['pull_mean']:+.2f} rms={s['pull_rms']:.2f} "
        f"robust={s['pull_width_robust']:.2f}  "
        f"sharp={s['sharpness_median']:.2f}  reward={s['reward_mean']:+.3f}"
    )


def cmd_list(args) -> int:
    for f in FAMILIES.values():
        qs = ", ".join(f"{q} [{s.unit}]" for q, s in f.quantities.items())
        print(f"tier {f.tier}  {f.name:<18s} {f.title}  ({qs})")
    return 0


def cmd_export(args) -> int:
    from pullgym.export import export_task, heldout_seeds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fam in _families(args.family):
        if args.split == "dev":
            seeds = list(range(args.start, args.start + args.n))
        else:
            if not args.secret_file:
                print("--secret-file is required for the test split", file=sys.stderr)
                return 2
            secret = Path(args.secret_file).read_text().strip()
            seeds = heldout_seeds(secret, fam.name, args.start + args.n)[args.start :]
        for seed in seeds:
            export_task(fam, seed, args.split, args.variant, out)
        print(f"{fam.name}: wrote {len(seeds)} tasks to {out}")
    return 0


def cmd_calibrate(args) -> int:
    from pullgym.stats import reference_episodes, summarize

    for fam in _families(args.family):
        rows = reference_episodes(fam, range(args.start, args.start + args.n))
        print(f"{fam.name:<18s} {_fmt_summary(summarize(rows))}")
    return 0


def cmd_probe(args) -> int:
    from pullgym.stats import probe

    for fam in _families(args.family):
        print(f"== {fam.name}")
        res = probe(fam, range(args.start, args.start + args.n))
        for name, s in sorted(res.items(), key=lambda kv: -kv[1]["reward_mean"]):
            print(f"  {name:<20s} reward={s['reward_mean']:+9.3f}  cov68={s['cov68']:.1%}")
    return 0


def cmd_report(args) -> int:
    from pullgym.report import report

    print(report(Path(args.path), group_by=args.by, truth_bins=args.truth_bins))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pullgym", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list experiment families").set_defaults(fn=cmd_list)

    e = sub.add_parser("export", help="write Harbor task folders")
    e.add_argument("--family", default="all", help="family name, comma list, or 'all'")
    e.add_argument("--split", choices=["dev", "test"], default="dev")
    e.add_argument("--n", type=int, default=10, help="tasks per family")
    e.add_argument("--start", type=int, default=0, help="index of the first task")
    e.add_argument("--variant", default="calibrated", choices=["calibrated", "neutral"])
    e.add_argument("--secret-file", help="private secret for the test split")
    e.add_argument("--out", default="tasks")
    e.set_defaults(fn=cmd_export)

    c = sub.add_parser("calibrate", help="check reference-analysis coverage on many seeds")
    c.add_argument("--family", default="all")
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--start", type=int, default=1_000_000)
    c.set_defaults(fn=cmd_calibrate)

    pr = sub.add_parser("probe", help="mean reward of degenerate answering strategies")
    pr.add_argument("--family", default="all")
    pr.add_argument("--n", type=int, default=1000)
    pr.add_argument("--start", type=int, default=2_000_000)
    pr.set_defaults(fn=cmd_probe)

    r = sub.add_parser("report", help="coverage and pull tables from Harbor job output")
    r.add_argument("path", help="a Harbor jobs directory (searched recursively)")
    r.add_argument(
        "--by", default="model,family,variant", help="comma list of grouping keys"
    )
    r.add_argument("--truth-bins", type=int, default=3, help="bins of the true value (1 = off)")
    r.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
