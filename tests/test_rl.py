import json

import pytest

from pullgym import scoring
from pullgym.families import FAMILIES
from pullgym.report import report
from pullgym.rl import (
    OOD_TEST_FAMILIES,
    TRAIN_FAMILIES,
    build_rl_records,
    evaluate_completions_to_trials,
    execute_completion,
    extract_python_code,
    make_binary_reward_fn,
    make_log_score_reward_fn,
)


def test_train_and_ood_splits_partition_all_families():
    assert set(TRAIN_FAMILIES).isdisjoint(set(OOD_TEST_FAMILIES))
    assert set(TRAIN_FAMILIES) | set(OOD_TEST_FAMILIES) == set(FAMILIES)
    assert len(TRAIN_FAMILIES) == 8
    assert len(OOD_TEST_FAMILIES) == 4


def test_build_rl_records_has_no_truth_leak():
    records = build_rl_records(
        ("line_fit", "voltmeter_gain", "dead_time", "no_signal"),
        n_per_family=2,
        seed_start=100_000,
        shuffle=True,
    )
    assert len(records) == 8
    for r in records:
        assert len(r["prompt"]) == 2
        user_msg = r["prompt"][1]["content"]
        assert "apparatus_note.md" in user_msg
        assert "Data file previews" in user_msg
        truth_doc = json.loads(r["truth_json"])
        for spec in truth_doc["quantities"].values():
            assert f"{spec['truth']:.8g}" not in user_msg


def test_extract_python_code_ignores_think_blocks():
    comp = (
        "<think>Maybe try:\n```python\nprint('wrong draft')\n```\n</think>\n"
        "Here is the final script:\n"
        "```python\nprint('final')\n```"
    )
    assert extract_python_code(comp) == "print('final')"
    # Also works on TRL conversational format.
    assert extract_python_code([{"role": "assistant", "content": comp}]) == "print('final')"
    assert extract_python_code("no code here") is None


def _line_fit_script(scale: float = 1.0) -> str:
    return (
        "<think>Compute weighted least squares for velocity.</think>\n"
        "```python\n"
        "import json, numpy as np\n"
        "d = np.loadtxt('data/positions.csv', delimiter=',', skiprows=1)\n"
        "t, x, s = d[:, 0], d[:, 1], d[:, 2]\n"
        "w = 1.0 / s**2\n"
        "A = np.vstack([np.ones_like(t), t]).T\n"
        "cov = np.linalg.inv(A.T @ (A * w[:, None]))\n"
        "beta = cov @ (A.T @ (w * x))\n"
        f"v, sv = float(beta[1]), float(np.sqrt(cov[1, 1])) * {scale}\n"
        "ans = {'velocity': {'q025': v-1.959964*sv, 'q16': v-0.994458*sv, "
        "'q50': v, 'q84': v+0.994458*sv, 'q975': v+1.959964*sv}}\n"
        "open('artifacts/answer.json', 'w').write(json.dumps(ans))\n"
        "```"
    )


def test_binary_vs_log_score_reward_behavior():
    records = build_rl_records(("line_fit",), n_per_family=10, seed_start=100_000, shuffle=False)
    files = [r["files_json"] for r in records]
    truths = [r["truth_json"] for r in records]

    honest_comps = [_line_fit_script(1.0) for _ in records]
    overconf_comps = [_line_fit_script(0.05) for _ in records]
    broken_comps = ["```python\nraise RuntimeError('boom')\n```" for _ in records]

    bin_fn = make_binary_reward_fn(k=1.0, invalid_penalty=-1.0, max_workers=4)
    log_fn = make_log_score_reward_fn(floor=-10.0, invalid_penalty=-12.0, max_workers=4)

    # Binary reward CANNOT distinguish calibrated (1x) from wildly overconfident (0.05x)!
    bin_honest = bin_fn(honest_comps, files_json=files, truth_json=truths)
    bin_overconf = bin_fn(overconf_comps, files_json=files, truth_json=truths)
    bin_broken = bin_fn(broken_comps, files_json=files, truth_json=truths)
    assert bin_honest == bin_overconf
    assert all(r == -1.0 for r in bin_broken)

    # Proper log-score reward strongly prefers calibrated (1x) over overconfident (0.05x),
    # and penalizes crashes strictly below the reward floor (-12 < -10).
    log_honest = log_fn(honest_comps, files_json=files, truth_json=truths)
    log_overconf = log_fn(overconf_comps, files_json=files, truth_json=truths)
    log_broken = log_fn(broken_comps, files_json=files, truth_json=truths)
    assert sum(log_honest) / len(log_honest) > sum(log_overconf) / len(log_overconf) + 3.0
    assert all(r == -12.0 for r in log_broken)
    assert all(r >= -10.0 for r in log_overconf)


def test_execute_completion_handles_timeout():
    rec = build_rl_records(("line_fit",), n_per_family=1, seed_start=100_000, shuffle=False)[0]
    oc = execute_completion(
        "```python\nimport time; time.sleep(5)\n```",
        rec["files_json"],
        rec["truth_json"],
        timeout_sec=0.2,
    )
    assert oc.valid is False
    assert "timed out" in (oc.error or "")


def test_evaluate_completions_to_trials_and_report(tmp_path):
    records = build_rl_records(
        ("line_fit",), n_per_family=3, seed_start=0, split="dev", shuffle=False
    )
    completions = [_line_fit_script(1.0), _line_fit_script(1.0), "no code"]
    job_dir = evaluate_completions_to_trials(
        records,
        completions,
        tmp_path / "rl_eval",
        job_id="arm-b-eval",
        model_name="qwen3-4b-logscore",
        max_workers=2,
    )
    text = report(job_dir, group_by="model,family,variant", truth_bins=1)
    assert "qwen3-4b-logscore" in text
    assert "line_fit" in text
    assert "67%" in text  # 2 of 3 valid
