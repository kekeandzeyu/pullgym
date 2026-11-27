"""Design checks for the Tier-5 families (implicit metrology, unprompted diagnostics).

Each family must (1) give no hint of its trap in the apparatus note when the
trap is meant to be found unprompted, (2) make a common shortcut fail clearly,
and (3) let a reasonable analysis that does not know the generator's hidden
hyperparameters come close to the reference. (3) is what keeps an exact
Bayesian reference from carrying a secret advantage.
"""

import re
import time

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr
from scipy.stats import norm
from scipy.stats import t as student_t

from pullgym.families import get_family
from pullgym.families.base import gaussian_quantiles, grid_quantiles, read_csv
from pullgym.scoring import LEVELS, score_quantity, sigma_from_quantiles
from pullgym.stats import summarize

TIER5 = ["outliers", "unmodeled_drift", "look_elsewhere", "datasheet_metrology"]
SPOILERS = {
    "outliers": r"glitch|outlier|spike|reject|filter|clip|anomal|robust",
    "unmodeled_drift": r"drift|trend|baseline|warm|temperature|stabil|time.depend",
}


@pytest.mark.parametrize("name", list(SPOILERS))
def test_no_unprompted_spoilers_in_note(name):
    fam = get_family(name)
    for seed in range(5):
        ep = fam.episode(seed)
        text = (ep.note + fam.task + fam.title).lower()
        assert not re.search(SPOILERS[name], text), re.search(SPOILERS[name], text)


@pytest.mark.parametrize("name", TIER5)
def test_reference_is_fast(name):
    fam = get_family(name)
    eps = [fam.episode(s) for s in range(20)]
    fam.reference_analysis(eps[0].files)  # warm-up
    t0 = time.perf_counter()
    for ep in eps:
        fam.reference_analysis(ep.files)
    assert (time.perf_counter() - t0) / len(eps) < 0.05  # 30 ms target, slack for CI


# ----------------------------------------------------------------------
# Shortcuts and agnostic analyses, one pair per family.


def _outliers_wmean(files):
    d = read_csv(files["drops.csv"])
    y, s = d["g_m_s2"], d["sigma_m_s2"]
    w = 1 / s**2
    return gaussian_quantiles(float((w * y).sum() / w.sum()), float(w.sum() ** -0.5))


def _outliers_agnostic(files):
    """Mixture with deliberately different hyperpriors: eps in [0, 0.5], k in [2, 50]."""
    d = read_csv(files["drops.csv"])
    y, s = d["g_m_s2"], d["sigma_m_s2"]
    med = np.median(y)
    g = np.linspace(med - 0.03, med + 0.03, 1201)
    eps = np.linspace(0, 0.5, 26)
    k = np.exp(np.linspace(np.log(2), np.log(50), 15))
    r2 = ((y[None] - g[:, None]) / s) ** 2
    lg = -0.5 * r2 - np.log(s)
    lb = -0.5 * r2[:, None, :] / k[None, :, None] ** 2 - np.log(k)[None, :, None] - np.log(s)
    with np.errstate(divide="ignore"):
        pp = np.logaddexp(np.log1p(-eps)[None, :, None, None] + lg[:, None, None, :],
                          np.log(eps)[None, :, None, None] + lb[:, None, :, :])
    return grid_quantiles(g, logsumexp(pp.sum(3), axis=(1, 2)))


def _drift_fit(files, degree):
    d = read_csv(files["runs.csv"])
    x = d["time_min"] / 30 - 1
    cols = [np.ones_like(x), d["sample_in"]] + [x**k for k in range(1, degree + 1)]
    a, w, y = np.column_stack(cols), 1 / d["sigma_Hz"] ** 2, d["frequency_offset_Hz"]
    cov = np.linalg.inv(a.T @ (a * w[:, None]))
    b = cov @ (a.T @ (w * y))
    return [float(b[1] + np.sqrt(cov[1, 1]) * z) for z in norm.ppf(LEVELS)]


_LE_EDGES = np.linspace(400.0, 760.0, 121)


def _le_argmax(files):
    """Fit the largest bump in the window and quote a Gaussian error."""
    n = read_csv(files["spectrum.csv"])["counts"]
    lo, hi = _LE_EDGES[:-1], _LE_EDGES[1:]
    b0, best = n.mean(), None
    for e0 in np.arange(420.0, 741.0):
        f = ndtr((hi - e0) / 3.0) - ndtr((lo - e0) / 3.0)
        shat = ((n - b0) * f).sum() / (f * f).sum()
        if best is None or shat > best[0]:
            best = (shat, f)
    shat, f = best
    return gaussian_quantiles(float(shat), float(np.sqrt(b0 / (f * f).sum())))


def _le_agnostic(files):
    """Marginalise the energy with a flat prior on s instead of the stated exponential."""
    n = read_csv(files["spectrum.csv"])["counts"]
    lo, hi = _LE_EDGES[:-1], _LE_EDGES[1:]
    s = np.linspace(0, 1500, 751)
    b = np.linspace(n.mean() * 0.9, n.mean() * 1.1, 31)
    acc = []
    for e0 in np.arange(420.0, 741.0, 2.0):
        f = ndtr((hi - e0) / 3.0) - ndtr((lo - e0) / 3.0)
        k = f > 1e-9
        mu = b[None, :, None] + s[:, None, None] * f[None, None, k]
        ll = (n[k] * np.log(mu) - mu).sum(2) - b[None, :] * (~k).sum() + n[~k].sum() * np.log(b)[None, :]
        acc.append(logsumexp(ll, axis=1))
    return grid_quantiles(s, logsumexp(np.array(acc), axis=0))


def _dm(files, gain_sd, off_sd):
    y = read_csv(files["readings.csv"])["voltage_mV"]
    i = float(read_csv(files["setup.csv"])["excitation_current_uA"][0])
    n = len(y)
    u_a = y.std(ddof=1) / np.sqrt(n)
    u = np.sqrt((gain_sd * y.mean()) ** 2 + off_sd**2 + u_a**2)
    nu = max(u**4 / (u_a**4 / (n - 1)), 1.0)  # Welch-Satterthwaite
    return [float(1000 * (y.mean() + u * z) / i) for z in student_t.ppf(LEVELS, nu)]


CASES = {
    # family: (shortcut, agnostic analysis, min reward loss of the shortcut,
    #          max reward loss of the agnostic analysis)
    "outliers": (_outliers_wmean, _outliers_agnostic, 1.5, 0.15),
    # Agnostic: a cubic baseline although the drift is quadratic.
    "unmodeled_drift": (lambda f: _drift_fit(f, 0), lambda f: _drift_fit(f, 3), 5.0, 0.15),
    # Every parameter of this family is stated in the note. The "agnostic"
    # analysis ignores the stated exponential prior on the line strength, so
    # it is a misspecification check and gets a looser bound.
    "look_elsewhere": (_le_argmax, _le_agnostic, 0.3, 0.25),
    "datasheet_metrology": (
        lambda f: _dm(f, 0.0, 0.0), lambda f: _dm(f, 0.015, 1.8 / np.sqrt(3)), 2.0, 0.15
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", TIER5)
def test_shortcut_fails_and_agnostic_analysis_is_close(name):
    fam = get_family(name)
    shortcut, agnostic, min_loss, max_agnostic_loss = CASES[name]
    rows = {"reference": [], "shortcut": [], "agnostic": []}
    q = next(iter(fam.quantities))
    for seed in range(10_000, 10_200):
        ep = fam.episode(seed)
        ref = fam.reference_analysis(ep.files)[q]
        s = sigma_from_quantiles(ref)
        for key, ans in (("reference", ref), ("shortcut", shortcut(ep.files)), ("agnostic", agnostic(ep.files))):
            m = score_quantity(ans, ep.truth[q], s)
            m["valid"] = 1.0
            rows[key].append(m)
    r = {k: summarize(v) for k, v in rows.items()}
    assert r["reference"]["reward_mean"] - r["shortcut"]["reward_mean"] > min_loss
    assert r["shortcut"]["cov68"] < r["reference"]["cov68"] - 0.15
    assert r["reference"]["reward_mean"] - r["agnostic"]["reward_mean"] < max_agnostic_loss
    assert abs(r["agnostic"]["cov68"] - 0.68) < 0.08
