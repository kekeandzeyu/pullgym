"""The interface every experiment family implements."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from pullgym.scoring import LEVELS

Quantiles = list[float]  # values at scoring.LEVELS


@dataclass(frozen=True)
class Quantity:
    unit: str
    description: str


@dataclass
class Episode:
    family: str
    seed: int
    theta: dict[str, float]
    truth: dict[str, float]
    files: dict[str, str]  # file name -> text, written to /app/data
    note: str  # the apparatus note, written to /app/data/apparatus_note.md
    extra: dict = field(default_factory=dict)


class Family(ABC):
    """One kind of simulated experiment.

    A family must never put the hidden parameters into the agent-visible
    files or the apparatus note, except as the simulated measurement itself.
    """

    name: str
    tier: int
    title: str
    quantities: dict[str, Quantity]
    task: str  # one paragraph: what the agent is asked to determine

    @abstractmethod
    def sample_theta(self, rng: np.random.Generator) -> dict[str, float]:
        """Draw the hidden parameters from the family's prior."""

    @abstractmethod
    def truth(self, theta: dict[str, float]) -> dict[str, float]:
        """The requested quantities, as functions of the hidden parameters."""

    @abstractmethod
    def simulate(self, theta: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
        """Simulate the experiment. Returns data files as {name: text}."""

    @abstractmethod
    def apparatus_note(self, theta: dict[str, float]) -> str:
        """The lab logbook entry the agent reads. Must not leak theta."""

    @abstractmethod
    def reference_analysis(self, files: dict[str, str]) -> dict[str, Quantiles]:
        """A careful human analysis of the data. Must reach nominal coverage."""

    # ------------------------------------------------------------------

    def episode(self, seed: int) -> Episode:
        # Independent streams for the hidden parameters and for the noise.
        theta = self.sample_theta(np.random.default_rng([seed, 0]))
        files = self.simulate(theta, np.random.default_rng([seed, 1]))
        return Episode(
            family=self.name,
            seed=seed,
            theta=theta,
            truth=self.truth(theta),
            files=files,
            note=self.apparatus_note(theta),
        )

    def prior_quantiles(self, n: int = 20000) -> dict[str, Quantiles]:
        """Quantiles of each requested quantity under the prior.

        Used as the fallback answer when an agent's answer is invalid: it is
        what an analyst who never looked at the data could honestly report.
        """
        rng = np.random.default_rng([2**31 - 1, 7])
        draws = [self.truth(self.sample_theta(rng)) for _ in range(n)]
        out = {}
        for q in self.quantities:
            vals = np.quantile([d[q] for d in draws], LEVELS)
            out[q] = _strictly_increasing([float(v) for v in vals])
        return out


def gaussian_quantiles(mean: float, sigma: float) -> Quantiles:
    return [float(mean + sigma * norm.ppf(t)) for t in LEVELS]


def read_csv(text: str) -> dict[str, np.ndarray]:
    """Parse a simple comma-separated file with a header row, ignoring # lines."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    header = [h.strip() for h in lines[0].split(",")]
    arr = np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=",", ndmin=2)
    return {h: arr[:, i] for i, h in enumerate(header)}


def write_csv(columns: dict[str, np.ndarray], fmt: dict[str, str], comment: str = "") -> str:
    names = list(columns)
    out = io.StringIO()
    for line in comment.splitlines():
        out.write(f"# {line}\n")
    out.write(",".join(names) + "\n")
    for row in zip(*(columns[n] for n in names)):
        out.write(",".join(format(v, fmt[n]) for v, n in zip(row, names)) + "\n")
    return out.getvalue()


def _strictly_increasing(vals: list[float]) -> list[float]:
    for i in range(1, len(vals)):
        if vals[i] <= vals[i - 1]:
            vals[i] = float(np.nextafter(vals[i - 1], np.inf))
    return vals
