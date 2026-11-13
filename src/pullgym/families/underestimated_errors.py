"""Tier 3: stated error bars that are too small by an unknown common factor.

Only the scatter of the points about the fit reveals it. The honest analysis
rescales by the goodness of fit and uses Student-t quantiles.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t

from pullgym.families.base import Family, Quantity, read_csv, weighted_line_fit, write_csv
from pullgym.scoring import LEVELS


class UnderestimatedErrors(Family):
    name = "underestimated_errors"
    tier = 3
    title = "Resistance from a V-I curve"
    quantities = {"resistance": Quantity("ohm", "the resistance of the resistor")}
    task = (
        "A student measured the voltage across a resistor at a series of currents and "
        "wrote down an uncertainty for each voltage reading. Determine the resistance "
        "in ohms."
    )

    def sample_theta(self, rng):
        return {
            "resistance": float(rng.uniform(50.0, 500.0)),
            "offset": float(rng.uniform(-0.05, 0.05)),  # V, meter zero offset
            "n": int(rng.integers(8, 21)),
            "sigma_scale": float(rng.uniform(1.0, 3.0)),  # true errors / stated errors
        }

    def truth(self, theta):
        return {"resistance": theta["resistance"]}

    def simulate(self, theta, rng):
        n = theta["n"]
        current = np.sort(rng.uniform(1.0, 20.0, n))  # mA
        stated = rng.uniform(0.01, 0.05, n)  # V
        voltage = (
            theta["offset"]
            + theta["resistance"] * current * 1e-3
            + rng.normal(0.0, stated * theta["sigma_scale"])
        )
        return {
            "vi.csv": write_csv(
                {"current_mA": current, "voltage_V": voltage, "sigma_V": stated},
                {"current_mA": ".3f", "voltage_V": ".4f", "sigma_V": ".3f"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Resistor V-I measurement\n\n"
            "- `vi.csv`: current `current_mA` (mA), voltage `voltage_V` (V), and the\n"
            "  student's uncertainty `sigma_V` (V) for each voltage reading.\n"
            "- The current readings are precise; their uncertainty is negligible.\n"
            "- The `sigma_V` values were estimated by the student from how much the last\n"
            "  digits of the display flickered. They were not checked independently.\n"
            "- The voltmeter may have a small zero offset.\n"
            "- The resistor is ohmic and its temperature was stable.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["vi.csv"])
        x, y, s = d["current_mA"] * 1e-3, d["voltage_V"], d["sigma_V"]
        beta, cov, chi2 = weighted_line_fit(x, y, s)
        dof = len(x) - 2
        # Errors are known only up to a common factor: estimate it from the fit.
        sigma = float(np.sqrt(cov[1, 1] * chi2 / dof))
        q = beta[1] + sigma * student_t.ppf(LEVELS, dof)
        return {"resistance": [float(v) for v in q]}
