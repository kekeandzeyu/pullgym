import pytest
import json

from pullgym import scoring
from pullgym.export import export_task
from pullgym.families import get_family
from pullgym.report import report


def _fake_trial(jobs, name, model, task_root, answer):
    trial = jobs / "job1" / name
    (trial / "verifier").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"agent": {"name": "agentx", "model_name": model}}))
    (trial / "trial.log").write_text("")
    ans = trial / "answer.json"
    ans.write_text(json.dumps(answer))
    truth = json.loads((task_root / "tests" / "truth.json").read_text())
    res = scoring.score_episode(str(ans), truth)
    (trial / "verifier" / "details.json").write_text(json.dumps(res["details"]))
    (trial / "verifier" / "reward.json").write_text(json.dumps(res["rewards"]))
    return res


def test_report_groups_by_model(tmp_path):
    fam = get_family("line_fit")
    jobs = tmp_path / "jobs"
    for seed in range(4):
        root = export_task(fam, seed, "dev", "calibrated", tmp_path / "tasks")
        ref = fam.reference_analysis(fam.episode(seed).files)["velocity"]
        _fake_trial(jobs, f"t{seed}a", "model-a", root, {"velocity": dict(zip(scoring.KEYS, ref))})
        bad = _fake_trial(jobs, f"t{seed}b", "model-b", root, {"velocity": "oops"})
        assert "covered68" not in bad["rewards"]
    # A trial that crashed before the verifier ran.
    crashed = jobs / "job1" / "crashed"
    crashed.mkdir()
    (crashed / "config.json").write_text("{}")
    (crashed / "trial.log").write_text("")

    text = report(jobs)
    lines = {ln.split()[0]: ln for ln in text.splitlines() if ln.startswith("model-")}
    assert "100%" in lines["model-a"] and "line_fit" in lines["model-a"]
    assert " 0% " in lines["model-b"]
    assert "1 trial(s) produced no verifier output" in text


def _trial_dir(tmp_path, name="t"):
    trial = tmp_path / "jobs" / "job1" / name
    trial.mkdir(parents=True)
    (trial / "trial.log").write_text("")
    return trial


def test_agent_and_model_from_harbor_result_json(tmp_path):
    # Harbor 0.25: config.json has no agent block; result.json carries it.
    from pullgym.report import trial_agent

    trial = _trial_dir(tmp_path)
    (trial / "config.json").write_text(json.dumps({"task": {}, "trial_name": "t", "job_id": "j"}))
    (trial / "result.json").write_text(
        json.dumps(
            {
                "config": {"agent": {"name": "claude-code", "model_name": "anthropic/m1"}},
                "agent_info": {"name": "claude-code", "model_info": {"name": "m1"}},
            }
        )
    )
    assert trial_agent(trial) == ("claude-code", "anthropic/m1")


def test_oracle_trial_is_labelled_by_agent(tmp_path):
    from pullgym.report import trial_agent

    trial = _trial_dir(tmp_path)
    (trial / "config.json").write_text("{}")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "config": {"agent": {"name": "oracle", "model_name": None}},
                "agent_info": {"name": "oracle", "model_info": None},
            }
        )
    )
    assert trial_agent(trial) == ("oracle", "oracle")


def test_report_counts_failures_when_nothing_scored(tmp_path):
    for i in range(3):
        trial = _trial_dir(tmp_path, f"t{i}")
        (trial / "config.json").write_text("{}")
        (trial / "exception.txt").write_text("RuntimeError: docker compose failed")
    text = report(tmp_path / "jobs")
    assert "No pullgym verifier output" in text
    assert "3 trial(s) ran but produced no verifier output" in text


def test_coverage_by_truth_finds_a_bad_region():
    from pullgym.stats import coverage_by_truth

    eps = [{"truth": t, "covered68": 1.0 if t > 1 else 0.0, "covered95": 1.0} for t in range(9)]
    bins = coverage_by_truth(eps, 3)
    assert [b["n"] for b in bins] == [3, 3, 3]
    assert bins[0]["cov68"] < 1.0 and bins[2]["cov68"] == 1.0


def _two_model_jobs(tmp_path, n=6):
    """model-a answers with the reference; model-b is 2x too narrow and invalid on seed 0."""
    fam = get_family("line_fit")
    jobs = tmp_path / "jobs"
    for seed in range(n):
        root = export_task(fam, seed, "dev", "calibrated", tmp_path / "tasks")
        ref = fam.reference_analysis(fam.episode(seed).files)["velocity"]
        narrow = [ref[2] + 0.5 * (q - ref[2]) for q in ref]
        _fake_trial(jobs, f"a{seed}", "model-a", root, {"velocity": dict(zip(scoring.KEYS, ref))})
        b_answer = {"velocity": "oops"} if seed == 0 else {"velocity": dict(zip(scoring.KEYS, narrow))}
        _fake_trial(jobs, f"b{seed}", "model-b", root, b_answer)
    return jobs


def test_report_rows_have_tier_task_and_pinball(tmp_path):
    from pullgym.report import load_episodes

    rows, _ = load_episodes(_two_model_jobs(tmp_path, n=2))
    r = rows[0]
    assert r["tier"] == 1 and r["task"].startswith("line_fit-dev-")
    assert r["pinball_skill"] is not None and "pt_acc" in r
    assert r["harness_version"] == "unknown"  # fixture writes no result.json


def test_summarize_valid_vs_all_and_wilson():
    from pullgym.stats import summarize, wilson_interval

    eps = [
        {"valid": 1.0, "reward": 1.0, "covered68": 1.0, "covered95": 1.0, "pull": 0.0, "sharpness": 1.0},
        {"valid": 1.0, "reward": 0.0, "covered68": 1.0, "covered95": 1.0, "pull": 0.5, "sharpness": 4.0},
        {"valid": 0.0, "reward": -4.0, "covered68": 0.0, "covered95": 1.0, "pull": 9.0, "sharpness": 50.0},
    ]
    s = summarize(eps)
    assert s["n"] == 3 and s["n_valid"] == 2
    assert s["reward_valid_mean"] == pytest.approx(0.5)
    assert s["reward_mean"] == pytest.approx(-1.0)
    assert s["cov68"] == 1.0
    assert s["sharpness_gmean"] == pytest.approx(2.0)
    # cli.py reads these
    assert "cov68_se" in s and "cov95_se" in s
    # Wilson at the extremes stays inside [0, 1] with non-zero width.
    lo, hi = wilson_interval(0, 15)
    assert lo == 0.0 and 0.15 < hi < 0.25
    lo, hi = wilson_interval(15, 15)
    assert hi == 1.0 and 0.75 < lo < 0.85
    assert s["cov68_ci_hi"] == 1.0 and s["cov68_ci_lo"] < 1.0


def test_paired_statistics():
    from pullgym.stats import mcnemar, paired_difference

    d = paired_difference([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0])
    assert d["mean"] == pytest.approx(1.0) and d["se"] == pytest.approx(0.0)
    assert d["ci_lo"] == pytest.approx(1.0) and d["ci_hi"] == pytest.approx(1.0)
    m = mcnemar([1, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1])
    assert (m["a_only"], m["b_only"]) == (2, 1)
    assert m["diff"] == pytest.approx(1 / 6)
    assert 0.0 < m["p"] <= 1.0


def test_report_paired_and_baseline(tmp_path):
    jobs = _two_model_jobs(tmp_path)
    text = report(jobs, group_by="model", truth_bins=1, paired=True)
    assert "Paired: 10 of 12 rows" in text
    # With exactly two models, --paired picks the first as baseline.
    assert "Paired differences (model - model-a)" in text
    diff_line = [ln for ln in text.splitlines() if ln.startswith("model-b ")][-1]
    cells = diff_line.split()
    assert cells[1] == "model-a" and cells[2] == "6" and cells[3] == "-17pt"
    # Narrow intervals lose reward against the reference on every valid pair.
    assert cells[6] == "5" and cells[7].startswith("-")  # "[CI]" cells split in two

    js = json.loads(report(jobs, group_by="model,tier", baseline="model-a", fmt="json"))
    assert js["baseline"] == "model-a"
    assert {g["model"] for g in js["groups"]} == {"model-a", "model-b"}
    d = js["paired_differences"][0]
    assert d["tier"] == 1 and d["n_valid"] == 5 and d["reward_valid"]["mean"] < 0


def test_report_formats_and_harness_warning(tmp_path):
    jobs = _two_model_jobs(tmp_path, n=3)
    md = report(jobs, group_by="model", truth_bins=1, fmt="md")
    assert md.startswith("| model | n | valid |") and "| model-a | 3 | 100% |" in md
    tex = report(jobs, group_by="model,family", truth_bins=1, fmt="latex", baseline="model-a")
    assert "\\begin{tabular}" in tex and "line\\_fit" in tex and "\\%" in tex
    assert "WARNING" not in report(jobs, truth_bins=1)
    trial = next(jobs.rglob("a0"))
    (trial / "result.json").write_text(json.dumps({"harness_version": "0.2.0"}))
    assert "WARNING: trials span several harness versions" in report(jobs, truth_bins=1)
    with pytest.raises(ValueError):
        report(jobs, baseline="no-such-model")
