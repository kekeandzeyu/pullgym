"""Tier 1: constant-velocity cart, straight-line fit with known Gaussian errors."""

from __future__ import annotations

import numpy as np

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv


class LineFit(Family):
    name = "line_fit"
    tier = 1
    title = "Cart on an air track"
    quantities = {"velocity": Quantity("cm/s", "the cart's constant velocity")}
    task = (
        "A cart glides along an air track at constant velocity. Its position was "
        "recorded at a series of times. Determine the cart's velocity in cm/s."
    )

    def sample_theta(self, rng):
        return {
            "x0": float(rng.uniform(-20.0, 20.0)),  # cm
            "velocity": float(rng.uniform(5.0, 40.0)),  # cm/s
            "n": int(rng.integers(6, 21)),
            "sigma_lo": float(rng.uniform(0.2, 1.0)),  # cm
        }

    def truth(self, theta):
        return {"velocity": theta["velocity"]}

    def simulate(self, theta, rng):
        n = theta["n"]
        t = np.sort(rng.uniform(0.0, 4.0, n))
        sigma = theta["sigma_lo"] * rng.uniform(1.0, 3.0, n)
        x = theta["x0"] + theta["velocity"] * t + rng.normal(0.0, sigma)
        text = write_csv(
            {"t_s": t, "x_cm": x, "sigma_x_cm": sigma},
            {"t_s": ".4f", "x_cm": ".3f", "sigma_x_cm": ".3f"},
        )
        return {"positions.csv": text}

    def apparatus_note(self, theta):
        return (
            "# Air track: cart position versus time\n\n"
            "- `positions.csv`: time `t_s` (s), position `x_cm` (cm), and the 1-sigma\n"
            "  position uncertainty `sigma_x_cm` (cm) for each reading.\n"
            "- Times come from a photogate clock; their uncertainty is negligible.\n"
            "- Position uncertainties were determined independently for each reading and\n"
            "  are Gaussian and uncorrelated. They are known, not estimated.\n"
            "- Friction is negligible over the recorded range.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["positions.csv"])
        t, x, s = d["t_s"], d["x_cm"], d["sigma_x_cm"]
        w = 1.0 / s**2
        A = np.vstack([np.ones_like(t), t]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        beta = cov @ (A.T @ (w * x))
        return {"velocity": gaussian_quantiles(beta[1], float(np.sqrt(cov[1, 1])))}
