"""Tier 3: correcting a high count rate for detector dead time of uncertain size.

The statistical error is tiny; the dead-time uncertainty dominates and enters
non-linearly, so the honest interval is asymmetric.
"""

from __future__ import annotations

import numpy as np

from pullgym.families.base import Family, Quantity, read_csv, write_csv
from pullgym.scoring import LEVELS

TAU_US, TAU_SIGMA_US = 150.0, 15.0  # stated dead time and its 1-sigma uncertainty
RUN_S = 10.0


class DeadTime(Family):
    name = "dead_time"
    tier = 3
    title = "True rate behind a busy counter"
    quantities = {"true_rate": Quantity("1/s", "the true rate of particles reaching the detector")}
    task = (
        "A counter recorded particles from an intense source in a series of runs. At "
        "this rate the counter misses events while it is busy. Determine the true rate "
        "at which particles reach the detector, in particles per second."
    )

    def sample_theta(self, rng):
        return {
            "true_rate": float(rng.uniform(500.0, 4000.0)),
            "tau_us": float(rng.normal(TAU_US, TAU_SIGMA_US)),
            "runs": int(rng.integers(5, 21)),
        }

    def truth(self, theta):
        return {"true_rate": theta["true_rate"]}

    def simulate(self, theta, rng):
        n, tau = theta["true_rate"], theta["tau_us"] * 1e-6
        # Non-paralysable counter: a renewal process with mean interval tau + 1/n.
        mean_gap, var_gap = tau + 1.0 / n, 1.0 / n**2
        mean = RUN_S / mean_gap
        sd = np.sqrt(RUN_S * var_gap / mean_gap**3)
        counts = np.round(rng.normal(mean, sd, theta["runs"])).astype(int)
        return {
            "runs.csv": write_csv(
                {"run": np.arange(1, theta["runs"] + 1), "recorded_counts": counts},
                {"run": "d", "recorded_counts": "d"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Counter at high rate\n\n"
            f"- `runs.csv`: recorded counts in each run. Every run lasted {RUN_S:.0f} s of\n"
            "  real time.\n"
            "- The counter is non-paralysable: after each recorded event it is dead for a\n"
            "  fixed time and ignores anything that arrives meanwhile.\n"
            f"- Manufacturer's dead time: {TAU_US:.0f} us with a 1-sigma uncertainty of\n"
            f"  {TAU_SIGMA_US:.0f} us (Gaussian). It has not been measured for this unit.\n"
            "- The source intensity was constant during the runs. Background is negligible.\n"
        )

    def reference_analysis(self, files):
        counts = read_csv(files["runs.csv"])["recorded_counts"]
        m_hat = counts.mean() / RUN_S
        m_sd = counts.std(ddof=1) / np.sqrt(len(counts)) / RUN_S
        # Propagate both uncertainties through n = m / (1 - m tau) by simulation.
        rng = np.random.default_rng(12345)
        tau = rng.normal(TAU_US, TAU_SIGMA_US, 200_000) * 1e-6
        # Few runs: the rate's scatter is itself estimated, so use Student-t.
        m = m_hat + m_sd * rng.standard_t(len(counts) - 1, tau.size)
        n = m / (1.0 - m * tau)
        return {"true_rate": [float(v) for v in np.quantile(n, LEVELS)]}
