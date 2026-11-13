"""Tier 2: a Gaussian peak on a sloped background in an energy spectrum."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from pullgym.families.base import (
    Family,
    Quantity,
    fit_binned_poisson,
    gaussian_quantiles,
    read_csv,
    write_csv,
)

E_LO, E_HI, N_BINS = 400.0, 800.0, 200
E_MID = 0.5 * (E_LO + E_HI)


def _model(p, lo, hi):
    s, e0, w, c0, c1 = p
    peak = s * (ndtr((hi - e0) / w) - ndtr((lo - e0) / w))
    centre = 0.5 * (lo + hi)
    bg = (c0 + c1 * (centre - E_MID)) * (hi - lo)  # exact for a linear density
    return peak + bg


class PeakBackground(Family):
    name = "peak_background"
    tier = 2
    title = "Gamma-ray line on a sloped background"
    quantities = {"energy": Quantity("keV", "the energy of the gamma-ray line (peak centre)")}
    task = (
        "A gamma-ray spectrometer recorded an energy spectrum containing a single "
        "gamma-ray line on top of a smoothly varying background. Determine the energy "
        "of the line in keV."
    )

    def sample_theta(self, rng):
        total_bg = rng.uniform(3000.0, 20000.0)
        ratio = rng.uniform(0.5, 2.0)  # background density, right edge / left edge
        c0 = total_bg / (E_HI - E_LO)  # mean density, counts/keV
        c1 = 2.0 * c0 * (ratio - 1.0) / (ratio + 1.0) / (E_HI - E_LO)
        return {
            "energy": float(rng.uniform(480.0, 720.0)),
            "n_signal": float(rng.uniform(600.0, 4000.0)),
            "width": float(rng.uniform(4.0, 12.0)),
            "c0": float(c0),
            "c1": float(c1),
        }

    def truth(self, theta):
        return {"energy": theta["energy"]}

    def simulate(self, theta, rng):
        edges = np.linspace(E_LO, E_HI, N_BINS + 1)
        lo, hi = edges[:-1], edges[1:]
        p = [theta["n_signal"], theta["energy"], theta["width"], theta["c0"], theta["c1"]]
        counts = rng.poisson(_model(p, lo, hi))
        return {
            "spectrum.csv": write_csv(
                {"e_low_keV": lo, "e_high_keV": hi, "counts": counts},
                {"e_low_keV": ".1f", "e_high_keV": ".1f", "counts": "d"},
            )
        }

    def apparatus_note(self, theta):
        return (
            "# Gamma-ray spectrometer\n\n"
            f"- `spectrum.csv`: raw counts in {N_BINS} energy bins from {E_LO:.0f} to\n"
            f"  {E_HI:.0f} keV, with bin edges `e_low_keV` and `e_high_keV`.\n"
            "- The energy calibration was checked the same day; its uncertainty is\n"
            "  negligible compared with what this spectrum can resolve.\n"
            "- The detector's line shape is Gaussian. Its width at this energy is not\n"
            "  known precisely.\n"
            "- Over this range the background is smooth and close to linear in energy.\n"
        )

    def reference_analysis(self, files):
        d = read_csv(files["spectrum.csv"])
        lo, hi, n = d["e_low_keV"], d["e_high_keV"], d["counts"]
        width = hi - lo
        centre = 0.5 * (lo + hi)
        # Starting values: linear background from the outer fifths, peak from the excess.
        side = (centre < E_LO + 0.2 * (E_HI - E_LO)) | (centre > E_HI - 0.2 * (E_HI - E_LO))
        c1_0, c0_0 = np.polyfit(centre[side] - E_MID, n[side] / width[side], 1)
        c0_0 = max(c0_0, 1.0)
        excess = n - (c0_0 + c1_0 * (centre - E_MID)) * width
        smooth = np.convolve(excess, np.ones(5) / 5, mode="same")
        e0_0 = float(centre[np.argmax(smooth)])
        s0 = max(float(excess.sum()), 50.0)
        starts = [[s0, e0_0, w0, c0_0, c1_0] for w0 in (5.0, 9.0)]
        bounds = [(1.0, None), (E_LO, E_HI), (0.5, 50.0), (1e-3, None), (None, None)]
        scales = [max(s0, 100.0), 10.0, 5.0, c0_0, max(abs(c0_0) / (E_HI - E_LO), 1e-3)]
        p, cov = fit_binned_poisson(lambda q: _model(q, lo, hi), n, starts, bounds, scales)
        return {"energy": gaussian_quantiles(p[1], float(np.sqrt(cov[1, 1])))}
