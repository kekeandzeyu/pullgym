import json
import subprocess
import sys
import tomllib

from pullgym.export import export_task, task_id, heldout_seeds
from pullgym.families import get_family


def test_export_layout_and_oracle(tmp_path):
    fam = get_family("voltmeter_gain")
    root = export_task(fam, 3, "dev", "calibrated", tmp_path)
    for rel in [
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/data/apparatus_note.md",
        "environment/data/readings.csv",
        "tests/Dockerfile",
        "tests/test.sh",
        "tests/scoring.py",
        "tests/truth.json",
        "solution/solve.sh",
    ]:
        assert (root / rel).is_file(), rel

    cfg = tomllib.loads((root / "task.toml").read_text())
    assert cfg["verifier"]["environment_mode"] == "separate"
    assert cfg["metadata"]["family"] == "voltmeter_gain"

    # Nothing in the agent's build context reveals the truth.
    truth = json.loads((root / "tests/truth.json").read_text())
    v = truth["quantities"]["voltage"]["truth"]
    for f in (root / "environment").rglob("*"):
        if f.is_file():
            assert f"{v:.5f}" not in f.read_text()

    # The oracle's answer scores like the reference, using the copied scorer.
    solve = (root / "solution/solve.sh").read_text()
    answer_json = solve.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    ans = tmp_path / "answer.json"
    ans.write_text(answer_json)
    out = tmp_path / "verifier"
    subprocess.run(
        [sys.executable, "-I", str(root / "tests/scoring.py"), "--answer", str(ans),
         "--truth", str(root / "tests/truth.json"), "--out", str(out)],
        check=True, capture_output=True,
    )
    rewards = json.loads((out / "reward.json").read_text())
    assert rewards["valid"] == 1.0
    assert abs(rewards["sharpness"] - 1.0) < 1e-6


def test_test_split_hides_seed(tmp_path):
    fam = get_family("line_fit")
    seed = heldout_seeds("s3cret", fam.name, 1)[0]
    name = task_id(fam, "test", seed)
    assert str(seed) not in name
    root = export_task(fam, seed, "test", "neutral", tmp_path)
    assert "seed" not in tomllib.loads((root / "task.toml").read_text())["metadata"]
    assert "seed" not in json.loads((root / "tests/truth.json").read_text())
    assert heldout_seeds("s3cret", fam.name, 3) != heldout_seeds("other", fam.name, 3)
