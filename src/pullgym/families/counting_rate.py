"""Tier 1: a Poisson counting rate from repeated fixed-time counts."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from pullgym.families.base import Family, Quantity, read_csv, write_csv
from pullgym.scoring import LEVELS

INTERVAL_S = 10.0


class CountingRate(Family):
    name = "counting_rate"
    tier = 1
    title = "Count rate of a radioactive source"
    quantities = {"rate": Quantity("1/s", "the mean count rate of the source")}
    task = (
        "A Geiger counter recorded the number of counts from a long-lived radioactive "
        "source in a series of equal, back-to-back time intervals. Determine the mean "
        "count rate in counts per second."
    )

    def sample_theta(self, rng):
        return {"rate": float(rng.uniform(2.0, 40.0)), "n": int(rng.integers(20, 61))}

    def truth(self, theta):
        return {"rate": theta["rate"]}

    def simulate(self, theta, rng):
        counts = rng.poisson(theta["rate"] * INTERVAL_S, theta["n"])
        return {
            "counts.csv": write_csv(
                {"interval": np.arange(1, theta["n"] + 1), "counts": counts},
                {"interval": "d", "counts": "d"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Geiger counter, fixed-time counting\n\n"
            f"- `counts.csv`: number of counts in each interval. Every interval lasted\n"
            f"  exactly {INTERVAL_S:.0f} s, and the intervals follow each other without gaps.\n"
            "- Background was measured separately and is below 0.01 counts/s: negligible.\n"
            "- The counter's dead time is negligible at these rates.\n"
            "- The source's half-life is many years, so its rate is constant here.\n"
        )

    def reference_analysis(self, files):
        counts = read_csv(files["counts.csv"])["counts"]
        total, exposure = float(counts.sum()), INTERVAL_S * len(counts)
        # Posterior for a Poisson rate under a flat prior: Gamma(N + 1, 1/T).
        q = gamma.ppf(LEVELS, total + 1.0, scale=1.0 / exposure)
        return {"rate": [float(v) for v in q]}
