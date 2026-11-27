"""Command-line interface: list, export, calibrate, probe, run-fast, report."""

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
    from pullgym.stats import PROBE_DIAGNOSTICS, probe

    for fam in _families(args.family):
        print(f"== {fam.name}")
        res = probe(fam, range(args.start, args.start + args.n))
        for name, s in sorted(res.items(), key=lambda kv: -kv[1]["reward_mean"]):
            tag = "  (diagnostic)" if name in PROBE_DIAGNOSTICS else ""
            print(f"  {name:<20s} reward={s['reward_mean']:+9.3f}  cov68={s['cov68']:.1%}{tag}")
    return 0


def cmd_report(args) -> int:
    from pullgym.report import report

    try:
        text = report(
            [Path(p) for p in args.path],
            group_by=args.by,
            truth_bins=args.truth_bins,
            paired=args.paired,
            baseline=args.baseline,
            fmt=args.format,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(text)
    return 0


def cmd_run_fast(args) -> int:
    import json

    from pullgym.report import report
    from pullgym.runner import OpenAICompatibleAgent, OracleAgent, PriorAgent, run_suite

    secret = None
    if args.split == "test":
        if not args.secret_file:
            print("--secret-file is required for the test split", file=sys.stderr)
            return 2
        secret = Path(args.secret_file).read_text().strip()

    if args.agent == "oracle":
        agent = OracleAgent()
    elif args.agent == "prior":
        agent = PriorAgent()
    else:
        if not args.model:
            print("--model is required when --agent openai", file=sys.stderr)
            return 2
        extra_body = json.loads(args.extra_body) if args.extra_body else None
        agent = OpenAICompatibleAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tool_mode=args.tool_mode,
            http_timeout_sec=args.http_timeout,
            extra_body=extra_body,
        )

    def _progress(idx: int, total: int, oc) -> None:
        if oc.skipped:
            status = "skipped (resumed)"
        elif oc.error:
            status = f"ERROR: {oc.error}"
        else:
            r = oc.rewards or {}
            valid = int(r.get("valid", 0.0))
            cov68 = f" cov68={int(r['covered68'])}" if "covered68" in r else ""
            status = f"valid={valid}{cov68} reward={r.get('reward', float('nan')):+.3f} ({oc.elapsed_sec:.1f}s)"
        print(f"[{idx}/{total}] {oc.task_name:<32s} {status}", flush=True)

    job_dir, outcomes = run_suite(
        _families(args.family),
        split=args.split,
        n=args.n,
        start=args.start,
        variant=args.variant,
        secret=secret,
        agent=agent,
        out_dir=Path(args.out),
        job_id=args.job_id,
        concurrency=args.concurrency,
        tool_timeout_sec=args.tool_timeout,
        resume=args.resume,
        on_complete=_progress,
    )
    print(f"\nSaved {len(outcomes)} trial(s) to {job_dir}")
    if not args.no_report:
        print()
        print(report(job_dir, group_by="model,family,variant", truth_bins=1))
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
    r.add_argument("path", nargs="+", help="one or more jobs directories (searched recursively)")
    r.add_argument(
        "--by",
        default="model,family,variant",
        help="comma list of grouping keys: model, agent, family, tier, variant, split, "
        "harness_version",
    )
    r.add_argument("--truth-bins", type=int, default=3, help="bins of the true value (1 = off)")
    r.add_argument(
        "--paired",
        action="store_true",
        help="restrict the table to tasks every model answered validly",
    )
    r.add_argument(
        "--baseline",
        help="add paired differences (bootstrap CIs, McNemar) of every model against this one",
    )
    r.add_argument("--format", choices=["text", "md", "latex", "json"], default="text")
    r.set_defaults(fn=cmd_report)

    rf = sub.add_parser("run-fast", help="run episodes locally without Docker")
    rf.add_argument("--family", default="all", help="family name, comma list, or 'all'")
    rf.add_argument("--split", choices=["dev", "test"], default="dev")
    rf.add_argument("--n", type=int, default=10, help="episodes per family")
    rf.add_argument("--start", type=int, default=0, help="index of the first task")
    rf.add_argument("--variant", default="calibrated", choices=["calibrated", "neutral"])
    rf.add_argument("--secret-file", help="private secret for the test split")
    rf.add_argument("--agent", choices=["openai", "oracle", "prior"], default="openai")
    rf.add_argument("--model", help="model name for OpenAI-compatible endpoints")
    rf.add_argument("--base-url", help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL)")
    rf.add_argument("--api-key", help="API key (default: OPENAI_API_KEY)")
    rf.add_argument("--out", default="runs", help="output jobs directory")
    rf.add_argument("--job-id", help="job subdirectory name (auto-generated if omitted)")
    rf.add_argument("--concurrency", type=int, default=4, help="parallel episode workers")
    rf.add_argument("--max-turns", type=int, default=20, help="max LLM turns per episode")
    rf.add_argument("--tool-timeout", type=float, default=30.0, help="per-tool timeout in seconds")
    rf.add_argument("--http-timeout", type=float, default=180.0, help="per-HTTP-call timeout in seconds")
    rf.add_argument("--temperature", type=float, default=0.0)
    rf.add_argument("--max-tokens", type=int, default=8192)
    rf.add_argument("--tool-mode", choices=["tools", "code"], default="tools")
    rf.add_argument("--extra-body", help="extra JSON fields merged into the chat completion request")
    rf.add_argument("--resume", action="store_true", help="skip already-scored episodes in job dir")
    rf.add_argument("--no-report", action="store_true", help="do not print summary table at end")
    rf.set_defaults(fn=cmd_run_fast)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
