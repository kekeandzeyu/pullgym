"""Answer validation and per-episode scoring.

This module is deliberately standard-library only and imports nothing else
from pullgym: the exporter copies it verbatim into each task's verifier image,
where it runs as a script (see ``main``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
KEYS = ("q025", "q16", "q50", "q84", "q975")
MAX_ANSWER_BYTES = 1_000_000
PULL_CAP = 1e6
MAX_ABS = 1e30  # keeps every derived number finite, so reward.json stays valid

# Expected mean pinball loss, in units of sigma, of the exact quantiles of a
# Gaussian with that sigma: mean over LEVELS of phi(z_level).
_PHI = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)  # noqa: E731
_Z = {0.025: -1.959964, 0.16: -0.994458, 0.5: 0.0, 0.84: 0.994458, 0.975: 1.959964}
GAUSSIAN_REFERENCE_LOSS = sum(_PHI(_Z[t]) for t in LEVELS) / len(LEVELS)
# q16 and q84 sit at -/+ Z68 sigma for a Gaussian; dividing by it recovers sigma.
Z68 = _Z[0.84]
NOMINAL = {"covered68": 0.68, "covered95": 0.95}


def sigma_from_quantiles(q: list[float]) -> float:
    """The Gaussian sigma implied by the q16-q84 interval."""
    return (q[3] - q[1]) / (2.0 * Z68)

# Applied on top of the fallback (prior) answer when an answer is invalid, so
# that a valid "I know nothing" answer is always strictly better than none.
INVALID_PENALTY = 1.0


class AnswerError(ValueError):
    """The answer file cannot be scored."""


def parse_quantiles(obj: Any, name: str) -> list[float]:
    if not isinstance(obj, dict):
        raise AnswerError(f"{name}: expected an object with keys {', '.join(KEYS)}")
    out = []
    for key in KEYS:
        if key not in obj:
            raise AnswerError(f"{name}: missing {key}")
        v = obj[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnswerError(f"{name}.{key}: not a number")
        v = float(v)
        if not math.isfinite(v):
            raise AnswerError(f"{name}.{key}: not finite")
        if abs(v) > MAX_ABS:
            raise AnswerError(f"{name}.{key}: magnitude above {MAX_ABS:g}")
        out.append(v)
    for a, b, ka, kb in zip(out, out[1:], KEYS, KEYS[1:]):
        if not a < b:
            raise AnswerError(f"{name}: {ka} must be strictly less than {kb}")
    return out


def load_answer(path: str, quantities: list[str]) -> dict[str, list[float]]:
    """Read and validate an answer file. Raises AnswerError."""
    if not os.path.isfile(path) or os.path.islink(path):
        raise AnswerError(f"no answer file at {path}")
    if os.path.getsize(path) > MAX_ANSWER_BYTES:
        raise AnswerError("answer file is too large")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f, parse_constant=_reject_constant)
    except AnswerError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as e:
        raise AnswerError(f"answer file is not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise AnswerError("answer file must contain a JSON object")
    return {q: parse_quantiles(data.get(q), q) for q in quantities}


def _reject_constant(token: str) -> float:
    raise AnswerError(f"{token} is not allowed in the answer")


def pinball(q: list[float], y: float) -> float:
    """Mean quantile (pinball) loss of the quantiles q at LEVELS for outcome y."""
    total = 0.0
    for tau, qt in zip(LEVELS, q):
        total += ((1.0 if y < qt else 0.0) - tau) * (qt - y)
    return total / len(LEVELS)


def score_quantity(q: list[float], truth: float, ref_sigma: float) -> dict[str, float]:
    """Per-episode metrics for one quantity.

    reward: 1 - L / L0, where L is the pinball loss in units of the reference
    analysis's sigma and L0 the expected loss of an exactly calibrated Gaussian
    answer as sharp as the reference. A linear function of a proper score, so
    still proper. The reference analysis scores about 0 on average; a perfect
    point mass on the truth would score 1.
    """
    loss = pinball(q, truth) / ref_sigma
    q025, q16, q50, q84, q975 = q
    half = ((q84 - q50) if truth >= q50 else (q50 - q16)) / Z68
    pull = (q50 - truth) / half
    pull = max(-PULL_CAP, min(PULL_CAP, pull))
    return {
        "reward": 1.0 - loss / GAUSSIAN_REFERENCE_LOSS,
        "loss": loss,
        "covered68": 1.0 if q16 <= truth <= q84 else 0.0,
        "covered95": 1.0 if q025 <= truth <= q975 else 0.0,
        "pull": pull,
        "sharpness": sigma_from_quantiles(q) / ref_sigma,
    }


def score_episode(answer_path: str, truth_doc: dict) -> dict:
    """Score an answer file against a truth document written by the exporter.

    Returns {"rewards": {...numeric...}, "details": {...}}.
    """
    quantities = truth_doc["quantities"]
    names = list(quantities)
    details: dict[str, Any] = {k: v for k, v in truth_doc.items() if k != "quantities"}
    try:
        answer = load_answer(answer_path, names)
        valid, error = True, None
    except AnswerError as e:
        answer = {n: quantities[n]["fallback"] for n in names}
        valid, error = False, str(e)

    per_q = {}
    for n in names:
        spec = quantities[n]
        per_q[n] = score_quantity(answer[n], spec["truth"], spec["ref_sigma"])
        if not valid:
            per_q[n]["reward"] -= INVALID_PENALTY
        per_q[n]["answer"] = answer[n]
        per_q[n]["truth"] = spec["truth"]
        per_q[n]["ref_sigma"] = spec["ref_sigma"]

    rewards: dict[str, float] = {
        "reward": sum(per_q[n]["reward"] for n in names) / len(names),
        "valid": 1.0 if valid else 0.0,
    }
    # Calibration metrics describe the agent's answer, so they are omitted
    # when the answer was invalid and the fallback was scored instead.
    if len(names) == 1 and valid:
        m = per_q[names[0]]
        for k in ("covered68", "covered95", "pull", "sharpness"):
            rewards[k] = m[k]
    details.update(valid=valid, error=error, quantities=per_q, reward=rewards["reward"])
    return {"rewards": rewards, "details": details}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a pullgym answer file.")
    p.add_argument("--answer", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True, help="directory for reward.json and details.json")
    args = p.parse_args(argv)
    with open(args.truth, encoding="utf-8") as f:
        truth_doc = json.load(f)
    result = score_episode(args.answer, truth_doc)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "reward.json"), "w", encoding="utf-8") as f:
        json.dump(result["rewards"], f)
    with open(os.path.join(args.out, "details.json"), "w", encoding="utf-8") as f:
        json.dump(result["details"], f, indent=2)
    print(json.dumps(result["rewards"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
