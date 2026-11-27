"""Tier 5: a resistance from a meter whose errors are given as a datasheet.

The datasheet follows GUM conventions that a careful metrologist must decode:
the gain accuracy is an expanded uncertainty (k = 2, so the standard
uncertainty is half of it), the zero offset is a rectangular limit (standard
uncertainty a / sqrt(3)), and the readout scatter must be estimated from a
handful of readings (Student-t-like tails). The readings are in mV, the current
in uA, and the answer is requested in ohms.

Settings are drawn so that each of the three terms dominates the error budget
on a substantial fraction of episodes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammainc, logsumexp, roots_hermitenorm, roots_legendre

from pullgym.families.base import Family, Quantity, grid_quantiles, read_csv, write_csv

GAIN_K2 = 0.030  # expanded (k = 2) relative gain accuracy
OFFSET_LIMIT_MV = 1.8  # rectangular zero-offset limit
NOISE_LO_MV, NOISE_HI_MV = 0.4, 4.0  # stated range of the rms readout noise
R_LO, R_HI = 50.0, 2000.0  # ohm, stated range for the resistance
_GH_X, _GH_W = roots_hermitenorm(21)  # standard-normal nodes for the gain
_GL_X, _GL_W = roots_legendre(21)  # uniform nodes for the offset


class DatasheetMetrology(Family):
    name = "datasheet_metrology"
    tier = 5
    title = "Resistance from a datasheet-specified voltmeter"
    quantities = {"resistance": Quantity("ohm", "the resistance of the sensor element")}
    task = (
        "A technician passed a calibrated constant current through a sensor element "
        "and read the voltage across it several times with a digital voltmeter. "
        "Determine the resistance of the element in ohms."
    )

    def sample_theta(self, rng):
        low_range = rng.random() < 0.5
        v_target = rng.uniform(15.0, 45.0) if low_range else rng.uniform(120.0, 250.0)  # mV
        r = float(np.exp(rng.uniform(np.log(R_LO * 1.2), np.log(R_HI / 1.2))))
        current_ua = float(np.round(1000.0 * v_target / r, 1))
        return {
            "resistance": r,
            "current_uA": max(current_ua, 1.0),
            "gain": float(1.0 + 0.5 * GAIN_K2 * rng.standard_normal()),
            "offset_mV": float(rng.uniform(-OFFSET_LIMIT_MV, OFFSET_LIMIT_MV)),
            "noise_mV": float(np.exp(rng.uniform(np.log(NOISE_LO_MV), np.log(NOISE_HI_MV)))),
            "n": int(rng.integers(5, 9)),
        }

    def truth(self, theta):
        return {"resistance": theta["resistance"]}

    def simulate(self, theta, rng):
        v_true = theta["current_uA"] * theta["resistance"] / 1000.0  # mV
        mean = theta["gain"] * v_true + theta["offset_mV"]
        v = mean + rng.normal(0.0, theta["noise_mV"], theta["n"])
        return {
            "readings.csv": write_csv(
                {"reading": np.arange(1, theta["n"] + 1), "voltage_mV": v},
                {"reading": "d", "voltage_mV": ".3f"},
            ),
            "setup.csv": write_csv(
                {"excitation_current_uA": np.array([theta["current_uA"]])},
                {"excitation_current_uA": ".1f"},
            ),
        }

    def apparatus_note(self, theta):
        return (
            "# Sensor resistance check\n\n"
            "- `setup.csv`: the excitation current, in microamperes, from a calibrated\n"
            "  current source. Its uncertainty is negligible.\n"
            "- `readings.csv`: repeated readings of the voltage across the element, in\n"
            "  millivolts, all taken at that current with the same meter settings.\n"
            "- The resistance is somewhere between "
            f"{R_LO:.0f} and {R_HI:.0f} ohm (it has not been measured\n"
            "  before).\n\n"
            "## Voltmeter datasheet (DC voltage, 300 mV range)\n\n"
            "| Specification | Value |\n"
            "| --- | --- |\n"
            f"| Gain accuracy | +/-{100 * GAIN_K2:.1f}% of reading (expanded uncertainty, coverage factor k = 2) |\n"
            f"| Zero offset | within +/-{OFFSET_LIMIT_MV:.1f} mV (limits) |\n"
            f"| Readout noise | {NOISE_LO_MV:.1f} to {NOISE_HI_MV:.1f} mV rms, depending on the environment |\n\n"
            "The gain error and the zero offset are fixed properties of this meter: they\n"
            "are the same for every reading. Neither has been calibrated for this unit.\n"
        )

    def reference_analysis(self, files):
        y = read_csv(files["readings.csv"])["voltage_mV"]
        current = float(read_csv(files["setup.csv"])["excitation_current_uA"][0])
        n = len(y)
        ybar, ss = float(y.mean()), float(np.sum((y - y.mean()) ** 2))
        # Posterior on R with the stated priors: gain ~ N(1, (k2/2)^2), offset
        # uniform in +-limit, log-uniform noise in [lo, hi], R uniform. The
        # noise integrates in closed form:
        #   int sigma^-(n+1) exp(-Q / 2 sigma^2) dsigma
        #     ~ Q^(-n/2) [P(n/2, Q / 2 lo^2) - P(n/2, Q / 2 hi^2)].
        gain = 1.0 + 0.5 * GAIN_K2 * _GH_X
        offset = OFFSET_LIMIT_MV * _GL_X
        r_hat = 1000.0 * ybar / current
        rel = np.sqrt(GAIN_K2**2 + (2.5 * OFFSET_LIMIT_MV / max(abs(ybar), 1e-9)) ** 2
                      + (3 * NOISE_HI_MV) ** 2 / n / max(ybar, 1e-9) ** 2)
        r = np.linspace(max(R_LO, r_hat * (1 - 3 * rel)), min(R_HI, r_hat * (1 + 3 * rel)), 701)
        mu = gain[None, :, None] * current * r[:, None, None] / 1000.0 + offset[None, None, :]
        q = ss + n * (ybar - mu) ** 2  # (R, gain, offset)
        a = 0.5 * n
        band = gammainc(a, q / (2 * NOISE_LO_MV**2)) - gammainc(a, q / (2 * NOISE_HI_MV**2))
        with np.errstate(divide="ignore"):
            logl = -a * np.log(q) + np.log(np.maximum(band, 0.0))
        logw = np.log(_GH_W)[None, :, None] + np.log(_GL_W)[None, None, :]
        return {"resistance": grid_quantiles(r, logsumexp(logl + logw, axis=(1, 2)))}
