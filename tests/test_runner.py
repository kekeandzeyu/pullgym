import json
import os
import subprocess
import sys

import pytest

from pullgym import cli, scoring
from pullgym.export import export_task
from pullgym.families import get_family
from pullgym.report import report, trial_agent
from pullgym.runner import (
    AgentError,
    AgentRunResult,
    OpenAICompatibleAgent,
    OracleAgent,
    PriorAgent,
    run_episode,
    run_suite,
)


@pytest.mark.parametrize(
    "family_name",
    ["line_fit", "decay_background", "voltmeter_gain", "correlated_norm", "dead_time", "no_signal"],
)
def test_oracle_exact_parity_with_harbor(tmp_path, family_name):
    fam = get_family(family_name)
    seed = 7
    harbor_root = export_task(fam, seed, "dev", "calibrated", tmp_path / "harbor_tasks")

    # Run Harbor's standalone verifier script on Harbor's oracle answer.
    solve = (harbor_root / "solution" / "solve.sh").read_text()
    answer_json = solve.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]
    ans_path = tmp_path / "harbor_answer.json"
    ans_path.write_text(answer_json)
    harbor_out = tmp_path / "harbor_verifier"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(harbor_root / "tests" / "scoring.py"),
            "--answer",
            str(ans_path),
            "--truth",
            str(harbor_root / "tests" / "truth.json"),
            "--out",
            str(harbor_out),
        ],
        check=True,
        capture_output=True,
    )
    harbor_rewards = json.loads((harbor_out / "reward.json").read_text())
    harbor_details = json.loads((harbor_out / "details.json").read_text())

    # Run pullgym run-fast with OracleAgent on the same family and seed.
    job_dir, outcomes = run_suite(
        [fam],
        split="dev",
        n=1,
        start=seed,
        variant="calibrated",
        agent=OracleAgent(),
        out_dir=tmp_path / "fast_runs",
        job_id="oracle-parity",
        concurrency=1,
    )
    assert len(outcomes) == 1
    oc = outcomes[0]
    assert oc.error is None
    fast_rewards = json.loads((oc.trial_dir / "verifier" / "reward.json").read_text())
    fast_details = json.loads((oc.trial_dir / "verifier" / "details.json").read_text())

    assert fast_rewards == pytest.approx(harbor_rewards, rel=1e-12, abs=1e-12)
    assert fast_details["valid"] == harbor_details["valid"] is True
    assert fast_details["family"] == harbor_details["family"] == family_name
    assert fast_details["variant"] == harbor_details["variant"] == "calibrated"
    assert trial_agent(oc.trial_dir) == ("oracle", "oracle")

    rep = report(job_dir, group_by="model,family,variant", truth_bins=1)
    assert "oracle" in rep and family_name in rep and "100%" in rep


def test_workspace_isolation_readonly_env_and_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-secret")
    fam = get_family("voltmeter_gain")
    seen = {}

    class InspectAgent:
        agent_name = "inspector"
        model_name = "inspector-v1"

        def solve(self, prompt, workspace, family):
            # 1. No truth.json anywhere inside the episode root.
            all_names = [p.name for p in workspace.root.rglob("*")]
            seen["has_truth_file"] = any("truth" in n for n in all_names)

            # 2. Data directory and files are read-only.
            res_write = workspace.run_python(
                "from pathlib import Path\n"
                "try:\n"
                "    Path('data/apparatus_note.md').write_text('hacked')\n"
                "    print('WRITABLE')\n"
                "except PermissionError:\n"
                "    print('READONLY')\n"
            )
            seen["readonly_out"] = res_write["stdout"].strip()

            # 3. Sensitive API keys are stripped from subprocess environment.
            res_env = workspace.run_python(
                "import os\n"
                "print(os.environ.get('OPENAI_API_KEY'), os.environ.get('DEEPSEEK_API_KEY'))\n"
            )
            seen["env_out"] = res_env["stdout"].strip()

            # 4. Hanging subprocess is killed on timeout.
            res_timeout = workspace.run_python("import time; time.sleep(5)", timeout=0.2)
            seen["timed_out"] = res_timeout["timed_out"]
            return AgentRunResult(turns=1, log_lines=["inspected"])

    oc = run_episode(fam, 3, "dev", "calibrated", InspectAgent(), tmp_path / "job")
    assert oc.error is None
    assert seen["has_truth_file"] is False
    if os.geteuid() != 0:
        assert seen["readonly_out"] == "READONLY"
    assert seen["env_out"] == "None None"
    assert seen["timed_out"] is True
    # Missing answer.json scores as invalid (fallback - penalty).
    assert oc.rewards["valid"] == 0.0


def test_openai_agent_tools_loop_and_report(tmp_path):
    fam = get_family("line_fit")
    calls = []

    def fake_chat(payload):
        calls.append(payload)
        turn = len(calls)
        if turn == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Inspecting data directory first.",
                            "tool_calls": [
                                {
                                    "id": "tc_1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_bash",
                                        "arguments": json.dumps({"command": "ls data"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        if turn == 2:
            code = (
                "import json\n"
                "import numpy as np\n"
                "data = np.loadtxt('data/positions.csv', delimiter=',', skiprows=1)\n"
                "t, x, s = data[:, 0], data[:, 1], data[:, 2]\n"
                "w = 1.0 / s**2\n"
                "A = np.vstack([np.ones_like(t), t]).T\n"
                "cov = np.linalg.inv(A.T @ (A * w[:, None]))\n"
                "beta = cov @ (A.T @ (w * x))\n"
                "v, sv = float(beta[1]), float(np.sqrt(cov[1, 1]))\n"
                "ans = {'velocity': {'q025': v-1.96*sv, 'q16': v-sv, 'q50': v, 'q84': v+sv, 'q975': v+1.96*sv}}\n"
                "with open('artifacts/answer.json', 'w') as f:\n"
                "    json.dump(ans, f)\n"
                "print('wrote answer')\n"
            )
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "Inspecting data directory first.",
                            "tool_calls": [
                                {
                                    "id": "tc_2",
                                    "type": "function",
                                    "function": {
                                        "name": "run_python",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 80},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": "Done."}}],
            "usage": {"prompt_tokens": 250, "completion_tokens": 5},
        }

    agent = OpenAICompatibleAgent(model="mock/test-model", chat_fn=fake_chat, max_turns=5)
    job_dir, outcomes = run_suite(
        [fam],
        split="dev",
        n=1,
        start=2,
        variant="neutral",
        agent=agent,
        out_dir=tmp_path / "runs",
        job_id="mock-tools",
        concurrency=1,
    )
    oc = outcomes[0]
    assert oc.error is None
    assert oc.rewards["valid"] == 1.0
    assert abs(oc.rewards["sharpness"] - 1.0) < 0.05
    # Verify reasoning_content is echoed back in multi-turn tool-call history (required by deepseek-reasoner).
    turn2_asst = [m for m in calls[1]["messages"] if m["role"] == "assistant"][0]
    assert turn2_asst.get("reasoning_content") == "Inspecting data directory first."

    result_doc = json.loads((oc.trial_dir / "result.json").read_text())
    assert result_doc["usage"] == {"turns": 3, "prompt_tokens": 550, "completion_tokens": 105}
    assert trial_agent(oc.trial_dir) == ("pullgym-fast", "mock/test-model")
    assert (oc.trial_dir / "transcript.json").is_file()

    rep = report(job_dir, group_by="model,family,variant", truth_bins=1)
    assert "mock/test-model" in rep and "neutral" in rep and "100%" in rep


def test_openai_agent_code_mode_and_error_reporting(tmp_path):
    fam = get_family("weighted_mean")

    def code_chat(payload):
        turn = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        if turn == 0:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Let's compute the weighted mean:\n```python\n"
                                "import json, numpy as np\n"
                                "d = np.loadtxt('data/groups.csv', delimiter=',', skiprows=1)\n"
                                "y, s = d[:, 1], d[:, 2]\n"
                                "w = 1.0 / s**2\n"
                                "mu = float(np.sum(w*y) / np.sum(w))\n"
                                "sig = float(1.0 / np.sqrt(np.sum(w)))\n"
                                "ans = {'speed': {'q025': mu-1.96*sig, 'q16': mu-sig, 'q50': mu, 'q84': mu+sig, 'q975': mu+1.96*sig}}\n"
                                "open('artifacts/answer.json', 'w').write(json.dumps(ans))\n"
                                "```\n"
                            ),
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Finished."}}]}

    agent = OpenAICompatibleAgent(
        model="mock/code-model", chat_fn=code_chat, tool_mode="code", max_turns=3
    )
    oc = run_episode(fam, 0, "dev", "calibrated", agent, tmp_path / "code_job")
    assert oc.error is None
    assert oc.rewards["valid"] == 1.0

    # Unrecoverable API error writes exception.txt and is counted by pullgym report.
    def broken_chat(payload):
        raise AgentError("HTTP 500 upstream down")

    bad_agent = OpenAICompatibleAgent(model="mock/broken", chat_fn=broken_chat)
    err_job = tmp_path / "err_job"
    oc_err = run_episode(fam, 0, "dev", "calibrated", bad_agent, err_job)
    assert oc_err.error is not None
    assert (oc_err.trial_dir / "exception.txt").is_file()
    assert not (oc_err.trial_dir / "verifier" / "details.json").exists()
    rep = report(err_job)
    assert "1 trial(s) ran but produced no verifier output" in rep


def test_resume_skips_completed_and_retries_failed(tmp_path):
    fam = get_family("counting_rate")
    job_dir = tmp_path / "runs" / "resume-job"

    # First run: seed 0 succeeds, seed 1 fails.
    class FlakyAgent:
        agent_name = "flaky"
        model_name = "flaky-1"
        calls = 0

        def solve(self, prompt, workspace, family):
            self.calls += 1
            if self.calls == 2:
                raise AgentError("transient failure")
            return OracleAgent().solve(prompt, workspace, family)

    flaky = FlakyAgent()
    _, first = run_suite(
        [fam],
        n=2,
        start=0,
        agent=flaky,
        out_dir=tmp_path / "runs",
        job_id="resume-job",
        concurrency=1,
    )
    assert first[0].error is None and first[1].error is not None
    assert (first[1].trial_dir / "exception.txt").is_file()

    # Second run with resume=True: skips seed 0, retries seed 1 and clears exception.txt.
    _, second = run_suite(
        [fam],
        n=2,
        start=0,
        agent=OracleAgent(),
        out_dir=tmp_path / "runs",
        job_id="resume-job",
        concurrency=1,
        resume=True,
    )
    by_seed = {o.seed: o for o in second}
    assert by_seed[0].skipped is True
    assert by_seed[1].skipped is False and by_seed[1].error is None
    assert not (by_seed[1].trial_dir / "exception.txt").exists()


def test_cli_run_fast_oracle_and_prior(tmp_path, capsys):
    rc1 = cli.main(
        [
            "run-fast",
            "--family",
            "line_fit,voltmeter_gain",
            "--n",
            "2",
            "--agent",
            "oracle",
            "--out",
            str(tmp_path / "runs"),
            "--job-id",
            "cli-oracle",
            "--concurrency",
            "2",
        ]
    )
    assert rc1 == 0
    out1 = capsys.readouterr().out
    assert "Saved 4 trial(s)" in out1 and "oracle" in out1

    rc2 = cli.main(
        [
            "run-fast",
            "--family",
            "line_fit",
            "--n",
            "2",
            "--agent",
            "prior",
            "--out",
            str(tmp_path / "runs"),
            "--job-id",
            "cli-prior",
            "--no-report",
        ]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "Saved 2 trial(s)" in out2
    assert isinstance(PriorAgent(), PriorAgent)


def test_http_client_retries_incomplete_read_and_retry_after(monkeypatch):
    import http.client
    import io
    import urllib.error
    import urllib.request

    from pullgym.runner import openai_http_chat

    sleeps = []
    monkeypatch.setattr("pullgym.runner.time.sleep", lambda s: sleeps.append(s))

    attempts = [0]

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        attempts[0] += 1
        if attempts[0] == 1:
            raise http.client.IncompleteRead(b"")
        if attempts[0] == 2:
            raise urllib.error.HTTPError(
                req.full_url, 429, "Too Many Requests", {"Retry-After": "3"}, io.BytesIO(b"rate limited")
            )
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    chat = openai_http_chat("https://example.invalid/v1", api_key="k", max_retries=4)
    resp = chat({"model": "m", "messages": []})
    assert resp["choices"][0]["message"]["content"] == "ok"
    assert attempts[0] == 3
    assert len(sleeps) == 2 and sleeps[1] >= 4.0


def test_final_turn_reminder_injected_before_cutoff(tmp_path):
    fam = get_family("line_fit")
    seen_tool_contents = []

    def fake_chat(payload):
        tool_msgs = [m for m in payload["messages"] if m["role"] == "tool"]
        if tool_msgs:
            seen_tool_contents.append(json.loads(tool_msgs[-1]["content"]))
        if len(tool_msgs) == 0:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tc_1",
                                    "type": "function",
                                    "function": {"name": "run_bash", "arguments": json.dumps({"command": "ls data"})},
                                }
                            ],
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"role": "assistant", "content": "Stopping."}}]}

    agent = OpenAICompatibleAgent(model="mock/budget", chat_fn=fake_chat, max_turns=2)
    run_episode(fam, 0, "dev", "calibrated", agent, tmp_path / "budget_job")
    assert len(seen_tool_contents) == 1
    assert "Final turn next" in seen_tool_contents[0].get("note", "")


def test_agent_python_has_advertised_libraries(tmp_path):
    # The prompt promises numpy, scipy, pandas and matplotlib; run-fast executes
    # agent code with this interpreter, so they must be core dependencies.
    from pullgym.runner import EpisodeWorkspace

    ws = EpisodeWorkspace(root=tmp_path, data_dir=tmp_path, answer_path=tmp_path / "a.json")
    out = ws.run_python("import numpy, scipy, pandas, matplotlib\nprint('ok')")
    assert out["exit_code"] == 0, out["stderr"]
    assert out["stdout"].strip() == "ok"


def test_answer_status_is_schema_only(tmp_path):
    from pullgym.runner import answer_status

    p = tmp_path / "answer.json"
    assert answer_status(p, ["x"]) == "missing"
    p.write_text("{not json")
    assert answer_status(p, ["x"]).startswith("invalid: answer file is not valid JSON")
    p.write_text(json.dumps({"x": {"q025": 0, "q16": 0, "q50": 1, "q84": 2, "q975": 3}}))
    assert answer_status(p, ["x"]) == "invalid: x: q025 must be strictly less than q16"
    p.write_text(json.dumps({"x": {"q025": 0, "q16": 1, "q50": 2, "q84": 3, "q975": 4}}))
    assert answer_status(p, ["x"]) == "valid"


def _code_answer(qs):
    keys = ("q025", "q16", "q50", "q84", "q975")
    ans = {"velocity": dict(zip(keys, qs))}
    return (
        "```python\nimport json, os\nos.makedirs('artifacts', exist_ok=True)\n"
        f"json.dump({ans!r}, open('artifacts/answer.json', 'w'))\n```"
    )


def test_tools_mode_code_block_with_invalid_answer_does_not_end_episode(tmp_path):
    """A tools-mode model writing an invalid answer via a code block can fix it."""
    fam = get_family("line_fit")
    ep = fam.episode(0)
    ref = fam.reference_analysis(ep.files)["velocity"]
    feedbacks = []

    def fake_chat(payload):
        users = [m for m in payload["messages"] if m["role"] == "user"]
        if len(users) > 1:
            feedbacks.append(users[-1]["content"])
        n_asst = sum(1 for m in payload["messages"] if m["role"] == "assistant")
        if n_asst == 0:  # ties at the boundary: q025 == q16
            bad = [ref[0], ref[0], ref[2], ref[3], ref[4]]
            return {"choices": [{"message": {"role": "assistant", "content": _code_answer(bad)}}]}
        if n_asst == 1:
            return {"choices": [{"message": {"role": "assistant", "content": _code_answer(ref)}}]}
        raise AssertionError("episode should end once the answer is valid")

    agent = OpenAICompatibleAgent(model="mock/fix", chat_fn=fake_chat, max_turns=6)
    oc = run_episode(fam, 0, "dev", "calibrated", agent, tmp_path / "fix_job")
    assert oc.error is None
    assert oc.rewards["valid"] == 1.0
    assert len(feedbacks) == 1
    assert "turn: 1/6" in feedbacks[0]
    assert "answer_status (`artifacts/answer.json`): invalid: velocity: q025 must be strictly less than q16" in feedbacks[0]
    log = (oc.trial_dir / "trial.log").read_text()
    assert "answer_status=invalid" in log and "answer_status=valid" in log
    result = json.loads((oc.trial_dir / "result.json").read_text())
    from pullgym import __version__

    assert result["harness_version"] == __version__


def test_tool_observations_carry_turn_and_status_and_early_warning(tmp_path):
    fam = get_family("line_fit")
    seen = []

    def fake_chat(payload):
        tool_msgs = [m for m in payload["messages"] if m["role"] == "tool"]
        if tool_msgs:
            seen.append(json.loads(tool_msgs[-1]["content"]))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"tc_{len(tool_msgs)}",
                                "type": "function",
                                "function": {"name": "run_bash", "arguments": json.dumps({"command": "true"})},
                            }
                        ],
                    }
                }
            ]
        }

    agent = OpenAICompatibleAgent(model="mock/stall", chat_fn=fake_chat, max_turns=6)
    oc = run_episode(fam, 0, "dev", "calibrated", agent, tmp_path / "stall_job")
    assert oc.rewards["valid"] == 0.0
    assert [s["turn"] for s in seen] == ["1/6", "2/6", "3/6", "4/6", "5/6"]
    assert all(s["answer_status"] == "missing" for s in seen)
    notes = {s["turn"]: s.get("note", "") for s in seen}
    assert "3 turns left" in notes["3/6"]
    assert "Final turn next" in notes["5/6"]
    assert notes["1/6"] == notes["2/6"] == notes["4/6"] == ""
