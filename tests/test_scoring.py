import json
import math

import numpy as np
import pytest

from pullgym import scoring
from pullgym.families.base import gaussian_quantiles

GOOD = {"q025": 1.0, "q16": 2.0, "q50": 3.0, "q84": 4.0, "q975": 5.0}


def _write(tmp_path, obj, raw=None):
    p = tmp_path / "answer.json"
    p.write_text(raw if raw is not None else json.dumps(obj))
    return str(p)


def test_valid_answer_parses(tmp_path):
    ans = scoring.load_answer(_write(tmp_path, {"x": GOOD}), ["x"])
    assert ans == {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}


@pytest.mark.parametrize(
    "obj",
    [
        [],
        {"y": GOOD},
        {"x": {**GOOD, "q16": 3.0}},  # not strictly increasing
        {"x": {**GOOD, "q84": 1.5}},
        {"x": {**GOOD, "q50": "3"}},
        {"x": {**GOOD, "q50": True}},
        {"x": {**GOOD, "q50": 1e31}},
        {"x": {k: v for k, v in GOOD.items() if k != "q975"}},
        {"x": [1, 2, 3, 4, 5]},
    ],
)
def test_invalid_answers_rejected(tmp_path, obj):
    with pytest.raises(scoring.AnswerError):
        scoring.load_answer(_write(tmp_path, obj), ["x"])


@pytest.mark.parametrize("raw", ["", "{", '{"x": {"q025": NaN}}', '{"x": Infinity}', "[" * 100000])
def test_malformed_files_rejected(tmp_path, raw):
    with pytest.raises(scoring.AnswerError):
        scoring.load_answer(_write(tmp_path, None, raw=raw), ["x"])


def test_missing_and_symlinked_files_rejected(tmp_path):
    with pytest.raises(scoring.AnswerError):
        scoring.load_answer(str(tmp_path / "nope.json"), ["x"])
    target = _write(tmp_path, {"x": GOOD})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(scoring.AnswerError):
        scoring.load_answer(str(link), ["x"])


def test_oversized_file_rejected(tmp_path):
    raw = json.dumps({"x": GOOD, "pad": "a" * (scoring.MAX_ANSWER_BYTES + 1)})
    with pytest.raises(scoring.AnswerError):
        scoring.load_answer(_write(tmp_path, None, raw=raw), ["x"])


def test_gaussian_reference_loss_constant():
    rng = np.random.default_rng(0)
    q = gaussian_quantiles(0.0, 1.0)
    losses = [scoring.pinball(q, y) for y in rng.normal(size=200_000)]
    assert np.mean(losses) == pytest.approx(scoring.GAUSSIAN_REFERENCE_LOSS, rel=0.01)


def test_honest_quantiles_beat_miscalibrated_ones():
    """Propriety, checked by simulation: the true distribution scores best."""
    rng = np.random.default_rng(1)
    ys = rng.normal(size=100_000)
    mean_reward = {}
    for scale in (0.5, 0.8, 1.0, 1.25, 2.0):
        q = gaussian_quantiles(0.0, scale)
        mean_reward[scale] = np.mean([scoring.score_quantity(q, y, 1.0)["reward"] for y in ys])
    assert max(mean_reward, key=mean_reward.get) == 1.0
    assert mean_reward[1.0] == pytest.approx(0.0, abs=0.02)


def test_pull_and_coverage():
    q = gaussian_quantiles(10.0, 2.0)
    m = scoring.score_quantity(q, 13.0, 2.0)
    assert m["pull"] == pytest.approx(-1.5, rel=1e-3)
    assert m["covered68"] == 0.0 and m["covered95"] == 1.0
    assert m["sharpness"] == pytest.approx(1.0, rel=1e-3)


def _truth_doc():
    return {
        "family": "f",
        "quantities": {
            "x": {
                "truth": 3.2,
                "ref_sigma": 1.0,
                "fallback": [-100.0, -50.0, 0.0, 50.0, 100.0],
            }
        },
    }


def test_invalid_answer_scores_below_fallback(tmp_path):
    doc = _truth_doc()
    fb = dict(zip(scoring.KEYS, doc["quantities"]["x"]["fallback"]))
    as_valid = scoring.score_episode(_write(tmp_path, {"x": fb}), doc)
    invalid = scoring.score_episode(str(tmp_path / "missing.json"), doc)
    assert invalid["rewards"]["valid"] == 0.0
    assert invalid["rewards"]["reward"] == pytest.approx(
        as_valid["rewards"]["reward"] - scoring.INVALID_PENALTY
    )


def test_reward_json_is_finite_numbers(tmp_path):
    extreme = {"x": {"q025": -1e30, "q16": -1e-300, "q50": 0.0, "q84": 1e-300, "q975": 1e30}}
    out = tmp_path / "out"
    scoring.main(["--answer", _write(tmp_path, extreme), "--truth", _dump(tmp_path), "--out", str(out)])
    rewards = json.loads((out / "reward.json").read_text())
    assert all(isinstance(v, float) and math.isfinite(v) for v in rewards.values())
    assert json.loads((out / "details.json").read_text())["valid"] is True


def _dump(tmp_path):
    p = tmp_path / "truth.json"
    p.write_text(json.dumps(_truth_doc()))
    return str(p)


def test_gaussian_answer_scores_as_that_gaussian():
    from scipy.stats import norm

    q = gaussian_quantiles(2.0, 0.5)
    for y in (-1.0, 1.4, 2.0, 2.3, 4.5):
        assert scoring.log_density(q, y) == pytest.approx(norm.logpdf(y, 2.0, 0.5), abs=1e-5)


def test_asymmetric_answer_is_a_normalised_density():
    from scipy.integrate import quad

    q = [0.1, 0.4, 1.0, 2.5, 6.0]
    total, _ = quad(lambda y: math.exp(scoring.log_density(q, y)), -50, 100, points=q, limit=200)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_overconfidence_costs_more_than_underconfidence():
    rng = np.random.default_rng(2)
    ys = rng.normal(size=50_000)
    narrow = np.mean([scoring.score_quantity(gaussian_quantiles(0, 0.1), y, 1.0)["reward"] for y in ys])
    wide = np.mean([scoring.score_quantity(gaussian_quantiles(0, 10), y, 1.0)["reward"] for y in ys])
    assert narrow < 3 * wide < 0
    assert scoring.REWARD_FLOOR <= narrow


def test_reward_is_clipped():
    q = gaussian_quantiles(0.0, 1e-6)
    assert scoring.score_quantity(q, 1.0, 1.0)["reward"] == scoring.REWARD_FLOOR
    assert scoring.score_quantity(q, 0.0, 1.0)["reward"] == scoring.REWARD_CAP
