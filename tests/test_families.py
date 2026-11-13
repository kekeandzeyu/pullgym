import pytest

from pullgym.families import FAMILIES
from pullgym.stats import probe, reference_episodes, summarize

FAMS = list(FAMILIES.values())


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_episode_is_deterministic(fam):
    a, b = fam.episode(123), fam.episode(123)
    assert a.files == b.files and a.truth == b.truth and a.note == b.note
    assert fam.episode(124).files != a.files


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_reference_answer_shape(fam):
    ep = fam.episode(5)
    ref = fam.reference_analysis(ep.files)
    assert set(ref) == set(fam.quantities) == set(ep.truth)
    for q in ref.values():
        assert len(q) == 5 and all(a < b for a, b in zip(q, q[1:]))


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_truth_not_in_visible_text(fam):
    ep = fam.episode(7)
    visible = ep.note + "".join(ep.files.values())
    for v in ep.truth.values():
        assert f"{v:.8g}" not in visible


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_reference_analysis_is_calibrated(fam):
    """The generator, verifier and reference must agree before any model is run.

    400 seeds: 68% coverage has SE 2.3 points, so the band is about 3 SE.
    `pullgym calibrate` runs the same check with more seeds.
    """
    s = summarize(reference_episodes(fam, range(10_000, 10_400)))
    assert abs(s["cov68"] - 0.68) < 0.07
    assert abs(s["cov95"] - 0.95) < 0.035
    assert abs(s["pull_mean"]) < 0.15
    assert 0.85 < s["pull_width_robust"] < 1.15


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_reference_beats_degenerate_strategies(fam):
    res = probe(fam, range(20_000, 20_200))
    best = max(res, key=lambda k: res[k]["reward_mean"])
    assert best == "reference"
