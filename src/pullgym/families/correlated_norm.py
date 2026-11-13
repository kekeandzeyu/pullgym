"""Tier 3: runs grouped by day share a calibration error within each day.

Ignoring the correlation gives error bars that are far too small; adding the
calibration uncertainty once, as if all days shared it, gives ones too large.
"""

from __future__ import annotations

import numpy as np

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv

CAL_REL = 0.005  # relative 1-sigma calibration uncertainty per day, stated in the note


class CorrelatedNorm(Family):
    name = "correlated_norm"
    tier = 3
    title = "Liquid density over several days"
    quantities = {"density": Quantity("kg/m^3", "the density of the liquid")}
    task = (
        "The density of a liquid was measured with a pycnometer in several runs spread "
        "over a few days. Determine the density of the liquid in kg/m^3."
    )

    def sample_theta(self, rng):
        days = int(rng.integers(2, 5))
        theta = {"density": float(rng.uniform(780.0, 1300.0)), "days": days}
        for d in range(days):
            theta[f"cal_{d}"] = float(rng.normal(1.0, CAL_REL))
            theta[f"runs_{d}"] = int(rng.integers(3, 9))
        return theta

    def truth(self, theta):
        return {"density": theta["density"]}

    def simulate(self, theta, rng):
        day, value, sigma = [], [], []
        for d in range(theta["days"]):
            k = theta[f"runs_{d}"]
            rel = rng.uniform(0.001, 0.004, k)
            true_sigma = theta["density"] * rel
            vals = theta[f"cal_{d}"] * theta["density"] + rng.normal(0.0, true_sigma)
            day += [d + 1] * k
            value += list(vals)
            sigma += list(true_sigma)
        return {
            "runs.csv": write_csv(
                {"day": np.array(day), "density_kg_m3": np.array(value), "sigma_kg_m3": np.array(sigma)},
                {"day": "d", "density_kg_m3": ".3f", "sigma_kg_m3": ".3f"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Pycnometer density runs\n\n"
            "- `runs.csv`: for each run, the `day` it was taken, the measured density\n"
            "  `density_kg_m3` and its 1-sigma statistical uncertainty `sigma_kg_m3`\n"
            "  (weighing and temperature scatter), in kg/m^3.\n"
            "- The pycnometer volume was recalibrated with water at the start of each day.\n"
            f"  Each day's calibration has a relative 1-sigma uncertainty of {CAL_REL:.1%}.\n"
            "  It scales every run of that day by the same factor, and the calibrations on\n"
            "  different days are independent of each other.\n"
            "- The statistical uncertainties are reliable and independent between runs.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["runs.csv"])
        day, y, s = d["day"].astype(int), d["density_kg_m3"], d["sigma_kg_m3"]
        same_day = (day[:, None] == day[None, :]).astype(float)
        est = float(np.average(y, weights=1 / s**2))
        for _ in range(3):  # the calibration term scales with the estimate itself
            V = np.diag(s**2) + same_day * (CAL_REL * est) ** 2
            Vi = np.linalg.inv(V)
            one = np.ones_like(y)
            var = 1.0 / float(one @ Vi @ one)
            est = var * float(one @ Vi @ y)
        return {"density": gaussian_quantiles(est, float(np.sqrt(var)))}
