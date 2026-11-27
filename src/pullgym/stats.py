"""Aggregate per-episode metrics into the per-family numbers pullgym reports."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from pullgym.families import Family
from pullgym.families.base import _strictly_increasing, gaussian_quantiles
from pullgym.scoring import NOMINAL, score_quantity, sigma_from_quantiles


# Pinball loss grows linearly with the miss, so a single answer 1e6 sigma off
# dominates a plain mean. Aggregates clip pinball_skill at this floor (the
# reward's floor) before averaging; the raw mean is kept as pinball_skill_mean.
PINBALL_SKILL_FLOOR = -20.0


def clip_pinball(v: float) -> float:
    return max(PINBALL_SKILL_FLOOR, float(v))


def summarize(episodes: Iterable[dict]) -> dict[str, float]:
    """Aggregate per-episode rows.

    Each row has reward and valid, plus covered68, covered95, pull, sharpness
    and optionally pinball_skill and pt_acc. Calibration metrics (cov68, pull,
    sharpness, pinball_skill_mean, pt_acc_valid) use valid rows only. The
    ``*_all`` metrics and ``reward_mean`` use every row: on an invalid row the
    scorer has scored the prior fallback, so they are full-sample, fallback-
    imputed numbers free of survivorship bias.

    ``cov68_se`` / ``cov95_se`` are the standard errors a *calibrated* analysis
    would have at this n (for judging the reference). ``cov68_ci_lo/hi`` are
    Wilson 95% score intervals on the observed coverage (for judging agents).
    """
    eps = list(episodes)
    n_all = len(eps)
    valid = [e for e in eps if e.get("valid", 1.0) >= 1.0]
    n = len(valid)
    out: dict[str, float] = {
        "n": n_all,
        "n_valid": n,
        "valid_rate": n / n_all if n_all else math.nan,
    }
    if n_all:
        reward = np.array([e["reward"] for e in eps], dtype=float)
        out["reward_mean"] = float(reward.mean())
        out["reward_se"] = _se(reward)
        if all("pt_acc" in e for e in eps):
            out["pt_acc_all"] = float(np.mean([e["pt_acc"] for e in eps]))
        if all(e.get("pinball_skill") is not None for e in eps):
            out["pinball_skill_all_mean"] = float(
                np.mean([clip_pinball(e["pinball_skill"]) for e in eps])
            )
    if n == 0:
        return out
    c68 = np.array([e["covered68"] for e in valid], dtype=float)
    c95 = np.array([e["covered95"] for e in valid], dtype=float)
    pull = np.array([e["pull"] for e in valid], dtype=float)
    sharp = np.array([e["sharpness"] for e in valid], dtype=float)
    reward_v = np.array([e["reward"] for e in valid], dtype=float)
    p16, p84 = np.percentile(pull, [16, 84])
    lo68, hi68 = wilson_interval(int(round(c68.sum())), n)
    lo95, hi95 = wilson_interval(int(round(c95.sum())), n)
    pos = sharp[sharp > 0]
    out.update(
        cov68=float(c68.mean()),
        cov68_se=_binom_se(NOMINAL["covered68"], n),
        cov68_ci_lo=lo68,
        cov68_ci_hi=hi68,
        cov95=float(c95.mean()),
        cov95_se=_binom_se(NOMINAL["covered95"], n),
        cov95_ci_lo=lo95,
        cov95_ci_hi=hi95,
        pull_mean=float(pull.mean()),
        pull_mean_se=_se(pull),
        pull_rms=math.sqrt(float(np.mean(pull**2))),
        pull_width_robust=float((p84 - p16) / 2),
        sharpness_median=float(np.median(sharp)),
        # Geometric mean: unlike the median, it moves when a minority of
        # episodes is 10x too wide or too narrow (the width-inflation audit).
        sharpness_gmean=float(np.exp(np.mean(np.log(pos)))) if len(pos) else math.nan,
        reward_valid_mean=float(reward_v.mean()),
        reward_valid_se=_se(reward_v),
    )
    if all(e.get("pinball_skill") is not None for e in valid):
        pb = [e["pinball_skill"] for e in valid]
        out["pinball_skill_mean"] = float(np.mean(pb))
        out["pinball_skill_clipped_mean"] = float(np.mean([clip_pinball(v) for v in pb]))
    if all("pt_acc" in e for e in valid):
        out["pt_acc_valid"] = float(np.mean([e["pt_acc"] for e in valid]))
    return out


def _se(x: np.ndarray) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else math.nan


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default).

    Unlike the Wald interval it stays inside [0, 1] and has sensible width at
    0% or 100% observed coverage.
    """
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_difference(
    a: Iterable[float], b: Iterable[float], n_boot: int = 10_000, seed: int = 0
) -> dict[str, float]:
    """Mean of the paired differences a_i - b_i with SE and a bootstrap 95% CI.

    The bootstrap resamples pairs with a fixed seed, so reports are reproducible.
    """
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else math.nan, "se": _se(d)}
    if n >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.update(ci_lo=float(lo), ci_hi=float(hi))
    else:
        out.update(ci_lo=math.nan, ci_hi=math.nan)
    return out


def mcnemar(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    """Paired comparison of two binary outcomes (e.g. covered68) on the same tasks.

    Returns the difference in proportions a - b, its paired SE, the discordant
    counts (a only, b only) and the exact two-sided McNemar p-value.
    """
    from scipy.stats import binomtest

    a = np.asarray(list(a), dtype=float) >= 0.5
    b = np.asarray(list(b), dtype=float) >= 0.5
    n = len(a)
    n_a = int(np.sum(a & ~b))
    n_b = int(np.sum(~a & b))
    out = {"n": n, "a_only": n_a, "b_only": n_b}
    if n == 0:
        out.update(diff=math.nan, se=math.nan, p=math.nan)
        return out
    diff = (n_a - n_b) / n
    var = (n_a + n_b - (n_a - n_b) ** 2 / n) / n**2
    p = binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    out.update(diff=diff, se=math.sqrt(max(var, 0.0)), p=float(p))
    return out


def coverage_by_truth(episodes: Iterable[dict], n_bins: int = 3) -> list[dict]:
    """68% coverage in bins of the true value (equal-count bins).

    An analysis can be calibrated on average and still fail in one region, for
    example near a physical boundary. Needs a "truth" key on each episode.
    """
    eps = [e for e in episodes if e.get("valid", 1.0) >= 1.0 and "truth" in e]
    if len(eps) < n_bins:
        return []
    truths = np.array([e["truth"] for e in eps])
    edges = np.quantile(truths, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, truths, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = [e for e, i in zip(eps, idx) if i == b]
        if sel:
            out.append(
                {
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "n": len(sel),
                    "cov68": float(np.mean([e["covered68"] for e in sel])),
                    "cov95": float(np.mean([e["covered95"] for e in sel])),
                }
            )
    return out


def _binom_se(p: float, n: int) -> float:
    """Standard error of a coverage estimate if the analysis were calibrated."""
    return math.sqrt(p * (1 - p) / n)


def reference_episodes(family: Family, seeds: Iterable[int]) -> list[dict]:
    """Score the family's own reference analysis on many seeds."""
    rows = []
    for seed in seeds:
        ep = family.episode(seed)
        ref = family.reference_analysis(ep.files)
        for q, quants in ref.items():
            sigma = sigma_from_quantiles(quants)
            m = score_quantity(quants, ep.truth[q], sigma)
            m["valid"] = 1.0
            m["seed"] = seed
            m["truth"] = ep.truth[q]
            rows.append(m)
    return rows


def _scaled(quants, k: float) -> list[float]:
    """Scale every quantile's offset from the median by k (keeps the shape)."""
    q50 = quants[2]
    return _strictly_increasing([float(q50 + k * (q - q50)) for q in quants])


# Each strategy maps (reference quantiles, reference sigma, prior quantiles) to
# an answer. Width and shift probes act on the reference's own quantiles, so a
# skewed or heavy-tailed reference is compared with versions of itself.
PROBE_STRATEGIES = {
    "reference": lambda q, s, prior: list(q),
    "gaussian approx": lambda q, s, prior: gaussian_quantiles(q[2], s),
    "width x0.5": lambda q, s, prior: _scaled(q, 0.5),
    "width x0.1": lambda q, s, prior: _scaled(q, 0.1),
    "width x2": lambda q, s, prior: _scaled(q, 2.0),
    "width x10": lambda q, s, prior: _scaled(q, 10.0),
    "shifted 1 sigma": lambda q, s, prior: _strictly_increasing([float(v + s) for v in q]),
    "prior (ignore data)": lambda q, s, prior: prior,
}
# Not a degenerate strategy: it measures how much a symmetric Gaussian summary
# of the reference loses, and may tie with the reference on Gaussian families.
PROBE_DIAGNOSTICS = frozenset({"gaussian approx"})


def probe(family: Family, seeds: Iterable[int]) -> dict[str, dict[str, float]]:
    """Mean reward of degenerate answering strategies, relative to the reference.

    A sound environment gives the reference a higher mean reward than every
    strategy outside PROBE_DIAGNOSTICS.
    """
    prior = family.prior_quantiles()
    rows: dict[str, list[dict]] = {k: [] for k in PROBE_STRATEGIES}
    for seed in seeds:
        ep = family.episode(seed)
        ref = family.reference_analysis(ep.files)
        for q, quants in ref.items():
            s = sigma_from_quantiles(quants)
            for name, strat in PROBE_STRATEGIES.items():
                m = score_quantity(strat(quants, s, prior[q]), ep.truth[q], s)
                m["valid"] = 1.0
                rows[name].append(m)
    return {k: summarize(v) for k, v in rows.items()}
