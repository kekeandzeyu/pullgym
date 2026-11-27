"""Tier 5: unannounced glitches in an automated measurement series.

A fraction of the drops is corrupted by trigger glitches that scatter far more
than the stated per-drop uncertainty. The apparatus note does not mention them;
only the residuals reveal them. A weighted mean with its formal error is badly
overconfident, and hard sigma-clipping ignores the uncertainty of which points
are glitches. The honest analysis models the contamination.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from pullgym.families.base import Family, Quantity, grid_quantiles, read_csv, write_csv

G_LO, G_HI = 9.78, 9.83  # m/s^2: stated range for local g
EPS_MAX = 0.25  # glitch fraction is drawn uniformly from [0, EPS_MAX]
K_LO, K_HI = 3.5, 15.0  # glitch scatter / stated sigma, log-uniform
_EPS = np.linspace(0.0, EPS_MAX, 26)
_LOGK = np.linspace(np.log(K_LO), np.log(K_HI), 12)
_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


class Outliers(Family):
    name = "outliers"
    tier = 5
    title = "Local gravity from automated free-fall drops"
    quantities = {"g": Quantity("m/s^2", "the local gravitational acceleration")}
    task = (
        "An automated free-fall gravimeter dropped a test mass repeatedly and fitted "
        "each drop's trajectory to obtain a value of g. Determine the local "
        "gravitational acceleration in m/s^2."
    )

    def sample_theta(self, rng):
        return {
            "g": float(rng.uniform(G_LO, G_HI)),
            "sigma0": float(rng.uniform(0.003, 0.008)),
            "n": int(rng.integers(18, 29)),
            "eps": float(rng.uniform(0.0, EPS_MAX)),
            "k": float(np.exp(rng.uniform(np.log(K_LO), np.log(K_HI)))),
        }

    def truth(self, theta):
        return {"g": theta["g"]}

    def simulate(self, theta, rng):
        n = theta["n"]
        sigma = theta["sigma0"] * rng.uniform(0.7, 1.3, n)
        glitch = rng.random(n) < theta["eps"]
        scale = np.where(glitch, theta["k"] * sigma, sigma)
        g = theta["g"] + rng.normal(0.0, scale)
        return {
            "drops.csv": write_csv(
                {"drop": np.arange(1, n + 1), "g_m_s2": g, "sigma_m_s2": sigma},
                {"drop": "d", "g_m_s2": ".6f", "sigma_m_s2": ".6f"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Free-fall gravimeter run\n\n"
            "- `drops.csv`: one row per drop: the drop number, the value of g from the\n"
            "  fit to that drop's interferometer fringe timings (`g_m_s2`), and the 1-sigma\n"
            "  uncertainty of that fit from timing jitter (`sigma_m_s2`).\n"
            "- Earth tides, the height of the reference point and the vertical gravity\n"
            "  gradient have already been corrected for in every row.\n"
            "- Rows are the automated logger's export of the whole session, in the order\n"
            "  the drops were made.\n"
            f"- The regional gravity survey puts g at this site between {G_LO:.2f} and\n"
            f"  {G_HI:.2f} m/s^2, with no value in that range favoured.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["drops.csv"])
        y, s = d["g_m_s2"], d["sigma_m_s2"]
        # Grid wide enough to hold the posterior even if most points are glitches.
        med = float(np.median(y))
        w = 1.0 / s**2
        spread = max(15.0 * float(np.sqrt(1.0 / w.sum())), 8.0 * 1.4826 * float(
            np.median(np.abs(y - med))) / np.sqrt(len(y)), 5.0 * float(s.max()))
        core = np.linspace(max(G_LO, med - spread), min(G_HI, med + spread), 401)
        g = np.unique(np.concatenate([np.linspace(G_LO, G_HI, 41), core]))
        # Mixture likelihood over (g, eps, k), with the generator's priors.
        r2 = ((y[None, :] - g[:, None]) / s[None, :]) ** 2  # (G, N)
        log_good = -0.5 * r2 - np.log(s) - _LOG_SQRT_2PI  # (G, N)
        k = np.exp(_LOGK)
        log_bad = (
            -0.5 * r2[:, None, :] / k[None, :, None] ** 2
            - np.log(k)[None, :, None]
            - np.log(s)[None, None, :]
            - _LOG_SQRT_2PI
        )  # (G, K, N)
        with np.errstate(divide="ignore"):
            l1e, le = np.log1p(-_EPS), np.log(_EPS)
        per_point = np.logaddexp(
            l1e[None, :, None, None] + log_good[:, None, None, :],
            le[None, :, None, None] + log_bad[:, None, :, :],
        )  # (G, E, K, N)
        logl = per_point.sum(axis=3)  # (G, E, K)
        # Trapezoid weights for the uniform eps prior; uniform in log k.
        we = np.full(_EPS.size, 1.0)
        we[[0, -1]] = 0.5
        log_post = logsumexp(logl + np.log(we)[None, :, None], axis=(1, 2))
        return {"g": grid_quantiles(g, log_post)}
