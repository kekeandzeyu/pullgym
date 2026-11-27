"""Tier 5: a switched measurement on a baseline that drifts, unannounced.

Sample and reference runs are interleaved, but not in a balanced pattern, and
the baseline drifts smoothly during the hour. The apparatus note does not
mention drift; a goodness-of-fit check of the constant-baseline model reveals
it. The naive difference of means is biased and its error far too small. Any
reasonable smooth detrending (a quadratic, a cubic, a spline) recovers an
honest answer.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pullgym.families.base import Family, Quantity, read_csv, write_csv
from pullgym.scoring import LEVELS

T_MAX = 60.0  # min
N_RUNS = 28
SHIFT_LO, SHIFT_HI = -5.0, 5.0  # Hz, stated range


def _legendre(t):
    x = 2.0 * t / T_MAX - 1.0
    return x, 0.5 * (3.0 * x**2 - 1.0)


class UnmodeledDrift(Family):
    name = "unmodeled_drift"
    tier = 5
    title = "Frequency shift from interleaved sample and reference runs"
    quantities = {
        "shift": Quantity("Hz", "the change in resonance frequency caused by the sample")
    }
    task = (
        "A resonator's frequency was read in a sequence of runs, some with the sample "
        "inserted and some without (reference runs). Determine the frequency shift "
        "caused by the sample, in Hz (sample minus reference)."
    )

    def sample_theta(self, rng):
        sigma0 = float(rng.uniform(0.2, 0.6))
        sign = lambda: 1.0 if rng.random() < 0.5 else -1.0  # noqa: E731
        return {
            "shift": float(rng.uniform(SHIFT_LO, SHIFT_HI)),
            "c0": float(rng.uniform(-50.0, 50.0)),
            # Drift sizes in units of the per-run noise, so the drift always
            # dominates the error of a naive analysis but never swamps the data.
            "c1": sign() * float(rng.uniform(2.0, 12.0)) * sigma0,
            "c2": sign() * float(rng.uniform(3.0, 12.0)) * sigma0,
            "sigma0": sigma0,
            # How unevenly the two modes are spread over the hour.
            "a1": sign() * float(rng.uniform(0.15, 0.35)),
            "a2": sign() * float(rng.uniform(0.15, 0.35)),
        }

    def truth(self, theta):
        return {"shift": theta["shift"]}

    def simulate(self, theta, rng):
        t = np.sort(rng.uniform(0.0, T_MAX, N_RUNS))
        p1, p2 = _legendre(t)
        # Runs are scheduled in short blocks; the chance that a block is a
        # sample block varies over the hour (operator availability), so the
        # modes are not balanced in time.
        mode = np.zeros(N_RUNS, dtype=int)
        i = 0
        while i < N_RUNS:
            length = int(rng.integers(2, 6))
            p = np.clip(0.5 + theta["a1"] * p1[i] + theta["a2"] * p2[i], 0.1, 0.9)
            mode[i : i + length] = 1 if rng.random() < p else 0
            i += length
        if mode.sum() < 6 or (1 - mode).sum() < 6:  # keep both modes measurable
            order = np.argsort(rng.random(N_RUNS))
            mode[order[:6]] = 1
            mode[order[6:12]] = 0
        sigma = theta["sigma0"] * rng.uniform(0.8, 1.25, N_RUNS)
        reading = (
            theta["c0"]
            + theta["shift"] * mode
            + theta["c1"] * p1
            + theta["c2"] * p2
            + rng.normal(0.0, sigma)
        )
        return {
            "runs.csv": write_csv(
                {
                    "run": np.arange(1, N_RUNS + 1),
                    "time_min": t,
                    "sample_in": mode,
                    "frequency_offset_Hz": reading,
                    "sigma_Hz": sigma,
                },
                {"run": "d", "time_min": ".2f", "sample_in": "d",
                 "frequency_offset_Hz": ".3f", "sigma_Hz": ".3f"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Resonator frequency-shift measurement\n\n"
            "- `runs.csv`: one row per run, in time order. `time_min` is the start time\n"
            "  in minutes, `sample_in` is 1 if the sample was inserted and 0 for a\n"
            "  reference run, `frequency_offset_Hz` is the measured resonance frequency\n"
            "  minus the nominal frequency, and `sigma_Hz` is its 1-sigma statistical\n"
            "  uncertainty from the frequency counter's integration.\n"
            "- Inserting the sample is the only intended difference between the two kinds\n"
            "  of run.\n"
            f"- The shift is expected to be between {SHIFT_LO:.0f} and {SHIFT_HI:+.0f} Hz.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["runs.csv"])
        t, mode = d["time_min"], d["sample_in"]
        y, s = d["frequency_offset_Hz"], d["sigma_Hz"]
        p1, p2 = _legendre(t)
        # Linear-Gaussian model with flat priors: the posterior on the shift is
        # Gaussian with the weighted-least-squares mean and covariance.
        a = np.column_stack([np.ones_like(t), mode, p1, p2])
        w = 1.0 / s**2
        cov = np.linalg.inv(a.T @ (a * w[:, None]))
        beta = cov @ (a.T @ (w * y))
        mu, sd = float(beta[1]), float(np.sqrt(cov[1, 1]))
        return {"shift": [float(mu + sd * z) for z in norm.ppf(LEVELS)]}
