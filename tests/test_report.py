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


def test_coverage_by_truth_finds_a_bad_region():
    from pullgym.stats import coverage_by_truth

    eps = [{"truth": t, "covered68": 1.0 if t > 1 else 0.0, "covered95": 1.0} for t in range(9)]
    bins = coverage_by_truth(eps, 3)
    assert [b["n"] for b in bins] == [3, 3, 3]
    assert bins[0]["cov68"] < 1.0 and bins[2]["cov68"] == 1.0
