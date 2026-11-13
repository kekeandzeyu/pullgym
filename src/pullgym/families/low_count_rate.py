"""Tier 2: a weak source over background, at counts where Gaussian errors fail."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from pullgym.families.base import Family, Quantity, grid_quantiles

S_MAX = 2.0  # counts/min, stated in the note
B_MIN, B_MAX = 0.2, 1.5  # counts/min, stated in the note


class LowCountRate(Family):
    name = "low_count_rate"
    tier = 2
    title = "Weak source over background"
    quantities = {
        "source_rate": Quantity("1/min", "the count rate due to the source alone")
    }
    task = (
        "A detector counted for a while with a weak source in place, and for a while "
        "with the source removed to measure the background. Determine the count rate "
        "due to the source alone, in counts per minute."
    )

    def sample_theta(self, rng):
        return {
            "source_rate": float(rng.uniform(0.0, S_MAX)),
            "background_rate": float(rng.uniform(B_MIN, B_MAX)),
            "t_on": float(np.round(rng.uniform(3.0, 10.0), 1)),
            "t_off": float(np.round(rng.uniform(10.0, 40.0), 1)),
        }

    def truth(self, theta):
        return {"source_rate": theta["source_rate"]}

    def simulate(self, theta, rng):
        s, b = theta["source_rate"], theta["background_rate"]
        n_on = rng.poisson((s + b) * theta["t_on"])
        n_off = rng.poisson(b * theta["t_off"])
        text = (
            "run,duration_min,counts\n"
            f"source_in,{theta['t_on']:.1f},{n_on}\n"
            f"background,{theta['t_off']:.1f},{n_off}\n"
        )
        return {"runs.csv": text}

    def apparatus_note(self, theta):
        return (
            "# Low-activity counting\n\n"
            "- `runs.csv`: two runs. `source_in` is with the source in place; `background`\n"
            "  is with it removed. `duration_min` is the live time in minutes and `counts`\n"
            "  the raw number of counts.\n"
            "- The background rate is the same in both runs.\n"
            f"- From how the source was prepared, its count rate in this detector is\n"
            f"  somewhere between 0 and {S_MAX:.0f} counts/min, with no value in that range\n"
            "  favoured over another.\n"
            f"- Past runs put the background between {B_MIN} and {B_MAX} counts/min, again\n"
            "  with no value favoured.\n"
        )

    def reference_analysis(self, files):
        lines = [ln.split(",") for ln in files["runs.csv"].strip().splitlines()[1:]]
        runs = {name: (float(t), float(c)) for name, t, c in lines}
        t_on, n_on = runs["source_in"]
        t_off, n_off = runs["background"]
        # Exact posterior on a grid, with the stated flat priors.
        s = np.linspace(0.0, S_MAX, 1601)
        b = np.linspace(B_MIN, B_MAX, 801)
        S, B = np.meshgrid(s, b, indexing="ij")
        mu_on, mu_off = (S + B) * t_on, B * t_off
        logl = n_on * np.log(mu_on) - mu_on + n_off * np.log(mu_off) - mu_off
        return {"source_rate": grid_quantiles(s, logsumexp(logl, axis=1))}

