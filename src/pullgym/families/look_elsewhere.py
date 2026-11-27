"""Tier 5: a resonance search where the line's energy is not known in advance.

Pure background routinely produces a 3-sigma bump somewhere among the ~40
resolution elements of the window. An analysis that fits the largest bump and
quotes its local significance mistakes a fluctuation for a signal (the
look-elsewhere effect). The honest analysis marginalises over the line energy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from pullgym.families.base import Family, Quantity, grid_quantiles, read_csv, write_csv

E_LO, E_HI, N_BINS = 400.0, 760.0, 120  # keV, 3 keV bins
RESOLUTION = 3.0  # keV, Gaussian line sigma
E0_LO, E0_HI = 420.0, 740.0  # stated range for the line energy
B_LO, B_HI = 40.0, 150.0  # stated range for the flat background, counts per bin
S_MEAN = 45.0  # stated: the line strength is exponential with this mean
_EDGES = np.linspace(E_LO, E_HI, N_BINS + 1)
_E0 = np.linspace(E0_LO, E0_HI, 321)  # 1 keV steps
_HALF = 6  # bins on each side of the line in the local stencil (+-18 keV = 6 sigma)


def _shape(e0, lo, hi):
    return ndtr((hi - e0) / RESOLUTION) - ndtr((lo - e0) / RESOLUTION)


# Local stencil: for each candidate energy, the 13 bins around it and the
# fraction of the line falling in each. Outside the stencil the line puts
# less than 1e-9 of its counts, and inside the window it is fully contained.
_CENTRE = np.clip(np.searchsorted(_EDGES, _E0) - 1, _HALF, N_BINS - 1 - _HALF)
_IDX = _CENTRE[:, None] + np.arange(-_HALF, _HALF + 1)[None, :]  # (E0, 13)
_F = _shape(_E0[:, None], _EDGES[:-1][_IDX], _EDGES[1:][_IDX])  # (E0, 13)


class LookElsewhere(Family):
    name = "look_elsewhere"
    tier = 5
    title = "Search for a resonance of unknown energy"
    quantities = {
        "signal_counts": Quantity(
            "counts", "the expected number of counts in the resonance line (its strength)"
        )
    }
    task = (
        "A gamma-ray spectrometer searched for a narrow resonance predicted to lie "
        f"somewhere between {E0_LO:.0f} and {E0_HI:.0f} keV. Its energy is not known in "
        "advance and the line may be weak. Determine the strength of the line, as the "
        "expected number of counts it contributes to this spectrum."
    )

    def sample_theta(self, rng):
        return {
            "signal_counts": float(rng.exponential(S_MEAN)),
            "e0": float(rng.uniform(E0_LO, E0_HI)),
            "background": float(rng.uniform(B_LO, B_HI)),
        }

    def truth(self, theta):
        return {"signal_counts": theta["signal_counts"]}

    def simulate(self, theta, rng):
        lo, hi = _EDGES[:-1], _EDGES[1:]
        mu = theta["background"] + theta["signal_counts"] * _shape(theta["e0"], lo, hi)
        return {
            "spectrum.csv": write_csv(
                {"e_low_keV": lo, "e_high_keV": hi, "counts": rng.poisson(mu)},
                {"e_low_keV": ".1f", "e_high_keV": ".1f", "counts": "d"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Resonance search\n\n"
            f"- `spectrum.csv`: raw counts in {N_BINS} bins of 3 keV from {E_LO:.0f} to\n"
            f"  {E_HI:.0f} keV, with bin edges `e_low_keV` and `e_high_keV`.\n"
            "- The resonance, if it is there, is a single line whose energy theory places\n"
            f"  anywhere between {E0_LO:.0f} and {E0_HI:.0f} keV, with no energy in that range\n"
            "  favoured.\n"
            f"- The detector's line shape is Gaussian with a standard deviation of\n"
            f"  {RESOLUTION:.1f} keV at all energies in this window, known from calibration.\n"
            "- The background is flat over the window. Its level is not known in advance,\n"
            f"  but past runs put it between {B_LO:.0f} and {B_HI:.0f} counts per bin.\n"
            "- Theory predicts the line strength (expected counts in this spectrum) only\n"
            f"  roughly: an exponential distribution with mean {S_MEAN:.0f} counts. It cannot be\n"
            "  negative.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["spectrum.csv"])
        n = d["counts"]
        n_tot, m = float(n.sum()), len(n)
        # The line is fully contained, so log L(s, E0, b) separates into
        #   sum_k n_k log b - M b - s  +  sum_{k near E0} n_k log(1 + (s/b) f_k(E0)).
        # The last term depends on (s, b) only through u = s / b: marginalise E0
        # once on a grid of u, then interpolate onto (s, b).
        b_hat = n_tot / m
        b_sd = np.sqrt(b_hat / m)
        b = np.linspace(max(B_LO, b_hat - 8 * b_sd - 12 * S_MEAN / m), min(B_HI, b_hat + 8 * b_sd), 41)
        s = 12 * S_MEAN * np.linspace(0.0, 1.0, 601) ** 2
        u = (s.max() / b.min()) * np.linspace(0.0, 1.0, 801) ** 2
        n_loc = n[_IDX]  # (E0, 13)
        loc = np.einsum("ek,uek->ue", n_loc, np.log1p(u[:, None, None] * _F[None, :, :]))
        h = logsumexp(loc, axis=1)  # uniform E0 prior (constant dropped)
        uu = s[:, None] / b[None, :]
        log_post = (
            np.interp(uu, u, h)
            + n_tot * np.log(b)[None, :]
            - m * b[None, :]
            - s[:, None] * (1.0 + 1.0 / S_MEAN)
        )
        return {"signal_counts": grid_quantiles(s, logsumexp(log_post, axis=1))}
