"""Tier 3: a straight-line fit where the x values have errors as large as y's."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv


class ErrorsXY(Family):
    name = "errors_xy"
    tier = 3
    title = "Spring constant from force and extension"
    quantities = {"spring_constant": Quantity("N/m", "the spring constant k")}
    task = (
        "A spring was loaded with a series of forces and its extension measured each "
        "time. Both the force and the extension readings have uncertainties. Determine "
        "the spring constant in N/m."
    )

    def sample_theta(self, rng):
        return {
            "spring_constant": float(rng.uniform(5.0, 50.0)),
            "preload": float(rng.uniform(0.0, 0.5)),  # N
            "n": int(rng.integers(8, 21)),
            "sigma_x": float(rng.uniform(0.002, 0.006)),  # m
            "sigma_f": float(rng.uniform(0.01, 0.05)),  # N
        }

    def truth(self, theta):
        return {"spring_constant": theta["spring_constant"]}

    def simulate(self, theta, rng):
        n = theta["n"]
        x_true = np.sort(rng.uniform(0.02, 0.20, n))
        sx = theta["sigma_x"] * rng.uniform(0.8, 1.2, n)
        sf = theta["sigma_f"] * rng.uniform(0.8, 1.2, n)
        x = x_true + rng.normal(0.0, sx)
        f = theta["preload"] + theta["spring_constant"] * x_true + rng.normal(0.0, sf)
        return {
            "spring.csv": write_csv(
                {"extension_m": x, "sigma_extension_m": sx, "force_N": f, "sigma_force_N": sf},
                {
                    "extension_m": ".5f",
                    "sigma_extension_m": ".5f",
                    "force_N": ".4f",
                    "sigma_force_N": ".4f",
                },
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Spring loading\n\n"
            "- `spring.csv`: extension `extension_m` (m) with its 1-sigma uncertainty\n"
            "  `sigma_extension_m`, and applied force `force_N` (N) with its 1-sigma\n"
            "  uncertainty `sigma_force_N`, for each loading.\n"
            "- The extension was read off a ruler by eye; the force from a calibrated\n"
            "  load cell. All uncertainties are reliable, Gaussian and independent.\n"
            "- The spring obeys Hooke's law over this range, but it was slightly preloaded,\n"
            "  so the force is not zero at zero measured extension.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["spring.csv"])
        x, sx, f, sf = d["extension_m"], d["sigma_extension_m"], d["force_N"], d["sigma_force_N"]

        # Effective-variance chi-square: for a straight line this is the exact
        # profile of the errors-in-both-variables likelihood over the true x.
        def chi2(p):
            a, k = p
            return float(np.sum((f - a - k * x) ** 2 / (sf**2 + (k * sx) ** 2)))

        k0 = np.polyfit(x, f, 1)
        p = minimize(chi2, [k0[1], k0[0]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12}).x
        # Covariance from the curvature: cov = 2 H^-1.
        h = np.array([1e-4 * max(abs(p[0]), 1e-2), 1e-4 * max(abs(p[1]), 1.0)])
        H = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                ei, ej = np.eye(2)[i] * h[i], np.eye(2)[j] * h[j]
                H[i, j] = (
                    chi2(p + ei + ej) - chi2(p + ei - ej) - chi2(p - ei + ej) + chi2(p - ei - ej)
                ) / (4 * h[i] * h[j])
        cov = 2.0 * np.linalg.inv(H)
        return {"spring_constant": gaussian_quantiles(p[1], float(np.sqrt(cov[1, 1])))}
