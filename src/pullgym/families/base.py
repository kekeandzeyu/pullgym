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


# ----------------------------------------------------------------------
# Shared analysis helpers for reference analyses.


def grid_quantiles(x: np.ndarray, log_post: np.ndarray) -> Quantiles:
    """Quantiles at LEVELS of a 1-D posterior tabulated on an increasing grid."""
    p = np.exp(log_post - np.max(log_post))
    # Trapezoid CDF, then linear interpolation of its inverse.
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
    cdf /= cdf[-1]
    keep = np.concatenate([[True], np.diff(cdf) > 0])
    vals = np.interp(LEVELS, cdf[keep], x[keep])
    return _strictly_increasing([float(v) for v in vals])


def weighted_line_fit(x, y, sigma):
    """Weighted least squares y = a + b x. Returns (a, b), covariance, chi2."""
    w = 1.0 / np.asarray(sigma) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ beta) ** 2))
    return beta, cov, chi2


def fit_binned_poisson(model, n, starts, bounds, scales=None):
    """Binned Poisson maximum likelihood with an expected-information covariance.

    model(p) returns expected counts per bin. Tries each start in `starts`.
    The optimiser works in coordinates p / scales, so that parameters of very
    different sizes (thousands of counts next to a slope of 0.01) move together.
    """
    from scipy.optimize import minimize

    n = np.asarray(n, dtype=float)
    scales = np.asarray(
        scales if scales is not None else np.maximum(np.abs(starts[0]), 1e-3), dtype=float
    )
    sbounds = [
        (None if lo is None else lo / sc, None if hi is None else hi / sc)
        for (lo, hi), sc in zip(bounds, scales)
    ]

    def nll(u):
        mu = model(u * scales)
        if np.any(mu <= 0) or not np.all(np.isfinite(mu)):
            return 1e300
        return float(np.sum(mu - n * np.log(mu)))

    best = None
    for p0 in starts:
        u0 = np.asarray(p0, dtype=float) / scales
        r = minimize(nll, u0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        u1 = np.clip(r.x, [lo if lo is not None else -np.inf for lo, _ in sbounds],
                     [hi if hi is not None else np.inf for _, hi in sbounds])
        r = minimize(nll, u1, method="L-BFGS-B", bounds=sbounds)
        if best is None or r.fun < best.fun:
            best = r
    p = np.array(best.x, dtype=float) * scales
    jac = np.empty((len(n), len(p)))
    for k in range(len(p)):
        h = 1e-5 * max(abs(p[k]), scales[k])
        up, dn = p.copy(), p.copy()
        up[k] += h
        dn[k] -= h
        jac[:, k] = (model(up) - model(dn)) / (2 * h)
    info = jac.T @ (jac / model(p)[:, None])
    return p, np.linalg.inv(info)
