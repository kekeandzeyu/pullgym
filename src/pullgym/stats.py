"""Aggregate per-episode metrics into the per-family numbers pullgym reports."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from pullgym.families import Family
from pullgym.families.base import gaussian_quantiles
from pullgym.scoring import NOMINAL, score_quantity, sigma_from_quantiles


def summarize(episodes: Iterable[dict]) -> dict[str, float]:
    """episodes: dicts with covered68, covered95, pull, sharpness, reward, valid."""
    eps = list(episodes)
    n_all = len(eps)
    valid = [e for e in eps if e.get("valid", 1.0) >= 1.0]
    n = len(valid)
    out: dict[str, float] = {"n": n_all, "valid_rate": n / n_all if n_all else math.nan}
    if n == 0:
        return out
    c68 = np.array([e["covered68"] for e in valid])
    c95 = np.array([e["covered95"] for e in valid])
    pull = np.array([e["pull"] for e in valid])
    sharp = np.array([e["sharpness"] for e in valid])
    reward = np.array([e["reward"] for e in eps])
    p16, p84 = np.percentile(pull, [16, 84])
    out.update(
        cov68=c68.mean(),
        cov68_se=_binom_se(NOMINAL["covered68"], n),
        cov95=c95.mean(),
        cov95_se=_binom_se(NOMINAL["covered95"], n),
        pull_mean=pull.mean(),
        pull_mean_se=pull.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan,
        pull_rms=math.sqrt(float(np.mean(pull**2))),
        pull_width_robust=(p84 - p16) / 2,
        sharpness_median=float(np.median(sharp)),
        reward_mean=reward.mean(),
        reward_se=reward.std(ddof=1) / math.sqrt(n_all) if n_all > 1 else math.nan,
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


PROBE_STRATEGIES = {
    "reference": lambda mu, s, prior: gaussian_quantiles(mu, s),
    "width x0.5": lambda mu, s, prior: gaussian_quantiles(mu, 0.5 * s),
    "width x0.1": lambda mu, s, prior: gaussian_quantiles(mu, 0.1 * s),
    "width x2": lambda mu, s, prior: gaussian_quantiles(mu, 2 * s),
    "width x10": lambda mu, s, prior: gaussian_quantiles(mu, 10 * s),
    "shifted 1 sigma": lambda mu, s, prior: gaussian_quantiles(mu + s, s),
    "prior (ignore data)": lambda mu, s, prior: prior,
}


def probe(family: Family, seeds: Iterable[int]) -> dict[str, dict[str, float]]:
    """Mean reward of degenerate answering strategies, relative to the reference.

    A sound environment gives the reference the highest mean reward.
    """
    prior = family.prior_quantiles()
    rows: dict[str, list[dict]] = {k: [] for k in PROBE_STRATEGIES}
    for seed in seeds:
        ep = family.episode(seed)
        ref = family.reference_analysis(ep.files)
        for q, quants in ref.items():
            mu, s = quants[2], sigma_from_quantiles(quants)
            for name, strat in PROBE_STRATEGIES.items():
                m = score_quantity(strat(mu, s, prior[q]), ep.truth[q], s)
                m["valid"] = 1.0
                rows[name].append(m)
    return {k: summarize(v) for k, v in rows.items()}
