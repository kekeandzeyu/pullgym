"""Tier 4: search for a weak line that is often too faint to see.

In many episodes the data cannot distinguish the signal from zero. The honest
answer is then an interval that starts at zero and works as an upper limit, not
a Gaussian "measurement" that extends to negative values.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from pullgym.families.base import Family, Quantity, grid_quantiles, read_csv, write_csv

E_LINE, RESOLUTION = 1000.0, 3.0  # keV; both stated in the note
E_LO, E_HI, N_BINS = 950.0, 1050.0, 100
S_MAX = 150.0  # counts; stated prior range
B_MIN, B_MAX = 50.0, 300.0  # background counts per bin; stated prior range


def _shape(lo, hi):
    return ndtr((hi - E_LINE) / RESOLUTION) - ndtr((lo - E_LINE) / RESOLUTION)


class NoSignal(Family):
    name = "no_signal"
    tier = 4
    title = "Search for a faint gamma-ray line"
    quantities = {
        "signal_counts": Quantity(
            "counts", "the expected number of counts in the line (its strength)"
        )
    }
    task = (
        f"A low-background spectrometer looked for a faint gamma-ray line at {E_LINE:.0f} "
        "keV that a rare decay would produce. The line may be too weak to see. "
        "Determine the strength of the line, as the expected number of counts it "
        "contributes to this spectrum."
    )

    def sample_theta(self, rng):
        return {
            "signal_counts": float(rng.uniform(0.0, S_MAX)),
            "background": float(rng.uniform(B_MIN, B_MAX)),
        }

    def truth(self, theta):
        return {"signal_counts": theta["signal_counts"]}

    def simulate(self, theta, rng):
        edges = np.linspace(E_LO, E_HI, N_BINS + 1)
        lo, hi = edges[:-1], edges[1:]
        mu = theta["signal_counts"] * _shape(lo, hi) + theta["background"]
        return {
            "spectrum.csv": write_csv(
                {"e_low_keV": lo, "e_high_keV": hi, "counts": rng.poisson(mu)},
                {"e_low_keV": ".1f", "e_high_keV": ".1f", "counts": "d"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Rare-decay line search\n\n"
            f"- `spectrum.csv`: raw counts in {N_BINS} bins of 1 keV from {E_LO:.0f} to\n"
            f"  {E_HI:.0f} keV, with bin edges `e_low_keV` and `e_high_keV`.\n"
            f"- If present, the line is at exactly {E_LINE:.0f} keV. The detector's line\n"
            f"  shape there is Gaussian with a standard deviation of {RESOLUTION:.1f} keV;\n"
            "  both are known precisely from calibration sources.\n"
            "- The background is flat over this window. Its level is not known in advance,\n"
            f"  but past runs put it between {B_MIN:.0f} and {B_MAX:.0f} counts per bin.\n"
            f"- Theory allows any line strength between 0 and {S_MAX:.0f} expected counts in\n"
            "  this spectrum, with no value in that range favoured. It cannot be negative.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["spectrum.csv"])
        lo, hi, n = d["e_low_keV"], d["e_high_keV"], d["counts"]
        f = _shape(lo, hi)
        # Exact posterior on a grid, flat priors as stated. The background is
        # well determined by the 100 bins, so its grid can stay narrow.
        b_hat = float(n.mean())
        b_sd = np.sqrt(b_hat / len(n)) + 1e-9
        lo_b = max(B_MIN, b_hat - 8 * b_sd - S_MAX / N_BINS)
        b = np.linspace(lo_b, min(B_MAX, b_hat + 8 * b_sd), 121)
        s = np.linspace(0.0, S_MAX, 751)
        sf = s[:, None] * f[None, :]
        logl = np.empty((s.size, b.size))
        for j, bj in enumerate(b):
            mu = sf + bj
            logl[:, j] = np.sum(n * np.log(mu) - mu, axis=1)
        return {"signal_counts": grid_quantiles(s, logsumexp(logl, axis=1))}
