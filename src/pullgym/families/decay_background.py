"""Tier 2: muon-lifetime-style decay curve over a flat background (binned Poisson)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv

WINDOW_US = 300.0
N_BINS = 100


def _signal_fractions(lo, hi, tau):
    """Fraction of decays (truncated to the window) that land in each bin."""
    norm = -np.expm1(-WINDOW_US / tau)
    return (np.exp(-lo / tau) - np.exp(-hi / tau)) / norm


class DecayBackground(Family):
    name = "decay_background"
    tier = 2
    title = "Decay curve over a flat background"
    quantities = {"lifetime": Quantity("us", "the mean lifetime tau of the decaying state")}
    task = (
        "A detector records the time between a start signal and a later stop signal. "
        "Genuine decays follow an exponential law; accidental coincidences add a "
        "background that is flat in time. Determine the mean lifetime of the "
        "decaying state in microseconds."
    )

    def sample_theta(self, rng):
        return {
            "lifetime": float(rng.uniform(10.0, 50.0)),  # us
            "n_signal": float(rng.uniform(1500.0, 6000.0)),  # expected decays in window
            "n_background": float(rng.uniform(1000.0, 8000.0)),  # expected flat counts
        }

    def truth(self, theta):
        return {"lifetime": theta["lifetime"]}

    def simulate(self, theta, rng):
        edges = np.linspace(0.0, WINDOW_US, N_BINS + 1)
        lo, hi = edges[:-1], edges[1:]
        mu = theta["n_signal"] * _signal_fractions(lo, hi, theta["lifetime"])
        mu = mu + theta["n_background"] * (hi - lo) / WINDOW_US
        counts = rng.poisson(mu)
        text = write_csv(
            {"t_low_us": lo, "t_high_us": hi, "counts": counts},
            {"t_low_us": ".1f", "t_high_us": ".1f", "counts": "d"},
        )
        return {"histogram.csv": text}

    def apparatus_note(self, theta):
        return (
            "# Delayed-coincidence timer\n\n"
            f"- `histogram.csv`: counts of stop signals in {N_BINS} time bins between\n"
            f"  0 and {WINDOW_US:.0f} us after the start signal. Columns are the bin edges\n"
            "  `t_low_us`, `t_high_us` (us) and the raw `counts`.\n"
            "- The timer's resolution is far smaller than the bin width.\n"
            "- Accidental coincidences give a background that is flat in time over the\n"
            "  whole window. Its rate is not known in advance.\n"
            "- The counts are raw event counts with no corrections applied.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["histogram.csv"])
        lo, hi, n = d["t_low_us"], d["t_high_us"], d["counts"]
        width = (hi - lo) / WINDOW_US

        def model(p):
            s, tau, b = p
            return s * _signal_fractions(lo, hi, tau) + b * width

        def nll(p):
            mu = model(p)
            return float(np.sum(mu - n * np.log(mu)))

        tail = hi > 0.7 * WINDOW_US
        b0 = max(n[tail].mean() / width[tail].mean(), 1.0)
        s0 = max(n.sum() - b0, 1.0)
        mean_t = np.sum(n * (lo + hi) / 2) / n.sum()
        tau0 = float(np.clip(mean_t, 2.0, 200.0))
        bounds = [(1.0, None), (0.5, 1000.0), (1e-3, None)]
        best = None
        for tau_start in (tau0, 0.5 * tau0, 2.0 * tau0, 30.0):
            r = minimize(nll, [s0, tau_start, b0], method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        # Expected (Fisher) information at the fit, derivatives by central difference.
        jac = np.empty((len(n), 3))
        for k in range(3):
            h = 1e-5 * max(abs(p[k]), 1.0)
            up, dn = p.copy(), p.copy()
            up[k] += h
            dn[k] -= h
            jac[:, k] = (model(up) - model(dn)) / (2 * h)
        info = jac.T @ (jac / model(p)[:, None])
        cov = np.linalg.inv(info)
        return {"lifetime": gaussian_quantiles(p[1], float(np.sqrt(cov[1, 1])))}
