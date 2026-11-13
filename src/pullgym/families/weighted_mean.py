"""Tier 1: combine independent measurements with known, very different errors."""

from __future__ import annotations

import numpy as np

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv


class WeightedMean(Family):
    name = "weighted_mean"
    tier = 1
    title = "Speed of sound from several groups"
    quantities = {"speed": Quantity("m/s", "the speed of sound in the lab air")}
    task = (
        "Several student groups measured the speed of sound in the same room on the "
        "same afternoon, each with a different method and its own uncertainty. "
        "Combine their results and determine the speed of sound in m/s."
    )

    def sample_theta(self, rng):
        return {"speed": float(rng.uniform(335.0, 350.0)), "k": int(rng.integers(4, 13))}

    def truth(self, theta):
        return {"speed": theta["speed"]}

    def simulate(self, theta, rng):
        k = theta["k"]
        sigma = np.exp(rng.uniform(np.log(0.3), np.log(8.0), k))
        values = theta["speed"] + rng.normal(0.0, sigma)
        return {
            "groups.csv": write_csv(
                {"group": np.arange(1, k + 1), "speed_m_s": values, "sigma_m_s": sigma},
                {"group": "d", "speed_m_s": ".2f", "sigma_m_s": ".2f"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Speed of sound: results from each group\n\n"
            "- `groups.csv`: each group's result `speed_m_s` and its 1-sigma uncertainty\n"
            "  `sigma_m_s`, both in m/s.\n"
            "- All groups measured the same air at the same temperature and humidity.\n"
            "- Each uncertainty was carefully evaluated by the instructor and is a reliable\n"
            "  Gaussian 1-sigma estimate. The groups' errors are independent.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["groups.csv"])
        w = 1.0 / d["sigma_m_s"] ** 2
        mean = float(np.sum(w * d["speed_m_s"]) / np.sum(w))
        return {"speed": gaussian_quantiles(mean, float(1.0 / np.sqrt(np.sum(w))))}
