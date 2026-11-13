"""Tier 3: a voltage measured with a meter whose gain has a certified uncertainty.

The statistical error of the mean is tiny. The honest error bar is dominated by
the gain calibration, which is stated only in the apparatus note.
"""

from __future__ import annotations

import numpy as np

from pullgym.families.base import Family, Quantity, gaussian_quantiles, read_csv, write_csv

GAIN_NOMINAL = 1.0
GAIN_SIGMA = 0.003  # certified 1-sigma relative gain uncertainty


class VoltmeterGain(Family):
    name = "voltmeter_gain"
    tier = 3
    title = "Reference voltage on a calibrated meter"
    quantities = {"voltage": Quantity("V", "the true output voltage of the source")}
    task = (
        "The output of a stable DC voltage source was read many times with a digital "
        "voltmeter, and the meter's zero was checked with its input shorted. "
        "Determine the true output voltage of the source in volts."
    )

    def sample_theta(self, rng):
        return {
            "voltage": float(rng.uniform(2.0, 20.0)),
            "gain": float(rng.normal(GAIN_NOMINAL, GAIN_SIGMA)),
            "offset": float(rng.uniform(-0.05, 0.05)),
            "noise": float(rng.uniform(0.005, 0.03)),
            "n": int(rng.integers(100, 301)),
            "n_zero": int(rng.integers(30, 81)),
        }

    def truth(self, theta):
        return {"voltage": theta["voltage"]}

    def simulate(self, theta, rng):
        g, off, s = theta["gain"], theta["offset"], theta["noise"]
        reads = g * theta["voltage"] + off + rng.normal(0.0, s, theta["n"])
        zeros = off + rng.normal(0.0, s, theta["n_zero"])
        fmt = {"reading_V": ".5f"}
        return {
            "readings.csv": write_csv({"reading_V": reads}, fmt),
            "zero_readings.csv": write_csv({"reading_V": zeros}, fmt),
        }

    def apparatus_note(self, theta):
        return (
            "# Digital voltmeter, 20 V range\n\n"
            "- `readings.csv`: repeated readings of the source (V), taken over a few\n"
            "  minutes. The source and the meter were stable over that time.\n"
            "- `zero_readings.csv`: readings with the meter input shorted (V), taken in the\n"
            "  same session, to measure the meter's zero offset.\n"
            "- The reading noise is Gaussian and the same for both files. It is not\n"
            "  specified by the manufacturer.\n"
            "- Calibration certificate for this meter on the 20 V range: gain\n"
            f"  {GAIN_NOMINAL:.4f} with a standard uncertainty of {GAIN_SIGMA:.4f} (k = 1),\n"
            "  i.e. a displayed reading equals gain x input + zero offset.\n"
        )

    def reference_analysis(self, files):
        r = read_csv(files["readings.csv"])["reading_V"]
        z = read_csv(files["zero_readings.csv"])["reading_V"]
        net = r.mean() - z.mean()
        v = net / GAIN_NOMINAL
        stat2 = (r.var(ddof=1) / len(r) + z.var(ddof=1) / len(z)) / GAIN_NOMINAL**2
        sys2 = (v * GAIN_SIGMA / GAIN_NOMINAL) ** 2
        return {"voltage": gaussian_quantiles(v, float(np.sqrt(stat2 + sys2)))}
