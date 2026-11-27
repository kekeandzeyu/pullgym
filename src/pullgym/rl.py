"""RLVR dataset generation, code-execution sandbox, and GRPO reward functions.

Designed for single-GPU Colab Pro training with Unsloth + TRL ``GRPOTrainer``.
Each episode is formulated as a single-turn code-synthesis problem:

1. The prompt includes the task description, full ``apparatus_note.md``, and
   headers + first rows of every CSV file in ``data/``.
2. The model reasons and outputs a self-contained Python script inside a
   ````python ... ```` block that reads ``data/...`` and writes
   ``artifacts/answer.json``.
3. The GRPO reward function executes each completion in parallel inside an
   isolated read-only workspace and scores ``artifacts/answer.json`` against
   hidden ground truth (never exposed in the prompt).

Two controlled reward arms are provided for ablation:
- Arm A (``make_binary_reward_fn``): standard binary point-accuracy RLVR
  checking ``|q50 - truth| <= k * ref_sigma``.
- Arm B (``make_log_score_reward_fn``): strictly proper distributional RLVR
  using ``scoring.log_score_reward``.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pullgym import __version__, scoring
from pullgym.export import VARIANTS, build_truth_doc, instruction, task_id
from pullgym.families import Family, get_family
from pullgym.runner import (
    EpisodeWorkspace,
    FAST_ANSWER_PATH,
    FAST_DATA_DIR,
    _set_readonly,
    _set_writable,
)

# 8 training families covering clean stats, nuisance parameters, and 2 systematics;
# 4 hard systematic / boundary families held out strictly for OOD evaluation.
TRAIN_FAMILIES: tuple[str, ...] = (
    "line_fit",
    "weighted_mean",
    "counting_rate",
    "decay_background",
    "peak_background",
    "low_count_rate",
    "voltmeter_gain",
    "underestimated_errors",
)
OOD_TEST_FAMILIES: tuple[str, ...] = (
    "errors_xy",
    "correlated_norm",
    "dead_time",
    "no_signal",
)

# Disjoint from dev (0..), test_families (10k/20k), calibrate (1M), probe (2M).
DEFAULT_TRAIN_SEED_START = 100_000

RL_SYSTEM_PROMPT = (
    "You are an experimental physicist and statistician. "
    "Read the apparatus note and data previews carefully, identify every statistical "
    "and systematic source of uncertainty (calibration certificates, correlated errors, "
    "nuisance parameters, nonlinear error propagation, and physical boundaries), and "
    "write a single self-contained Python 3 script inside a ```python ... ``` code block.\n"
    "Keep comments brief and write working Python code directly without repeating explanations.\n"
    "Your script runs with the working directory containing `data/` and `artifacts/`. "
    "It must read the full data files from `data/` (using numpy/scipy/pandas), compute "
    "calibrated posterior/uncertainty quantiles, and write the final JSON object to "
    "`artifacts/answer.json` with strictly increasing finite numbers for "
    "`q025`, `q16`, `q50`, `q84`, `q975`."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def preview_file(name: str, text: str, max_rows: int = 5) -> str:
    """Format a compact header + first-row preview of a data file for the RL prompt."""
    lines = text.splitlines()
    n_lines = len(lines)
    shown = lines[: max_rows + 1]
    body = "\n".join(shown)
    suffix = f"\n... ({n_lines - len(shown)} more lines)" if n_lines > len(shown) else ""
    return f"### `data/{name}` ({n_lines} lines)\n```csv\n{body}{suffix}\n```"


def build_rl_prompt(
    family: Family,
    ep,
    *,
    variant: str = "calibrated",
    preview_rows: int = 5,
) -> list[dict[str, str]]:
    """Build conversational messages for single-turn code-synthesis RL."""
    base = instruction(
        family, variant, data_dir=FAST_DATA_DIR, answer_path=FAST_ANSWER_PATH
    )
    previews = "\n\n".join(
        preview_file(fname, text, max_rows=preview_rows)
        for fname, text in sorted(ep.files.items())
    )
    user_content = (
        f"{base}\n"
        f"## Apparatus note (`data/apparatus_note.md`)\n\n"
        f"{ep.note.strip()}\n\n"
        f"## Data file previews\n\n"
        f"{previews}\n\n"
        "Write one complete, self-contained ```python ... ``` script that loads the "
        f"full data files from `{FAST_DATA_DIR}/` and writes `{FAST_ANSWER_PATH}`."
    )
    return [
        {"role": "system", "content": RL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_rl_records(
    family_names: tuple[str, ...] | list[str] = TRAIN_FAMILIES,
    *,
    n_per_family: int = 250,
    seed_start: int = DEFAULT_TRAIN_SEED_START,
    split: str = "train",
    variant: str = "calibrated",
    preview_rows: int = 5,
    shuffle: bool = True,
    shuffle_seed: int = 42,
) -> list[dict[str, Any]]:
    """Build JSON-serializable episode records ready for ``datasets.Dataset.from_list``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(VARIANTS)}")

    records: list[dict[str, Any]] = []
    for fname in family_names:
        fam = get_family(fname)
        prior = fam.prior_quantiles()
        for seed in range(seed_start, seed_start + n_per_family):
            ep = fam.episode(seed)
            ref = fam.reference_analysis(ep.files)
            tname = task_id(fam, split, seed)
            truth_doc = build_truth_doc(fam, ep, ref, prior, tname, split, variant, seed)
            records.append(
                {
                    "prompt": build_rl_prompt(
                        fam, ep, variant=variant, preview_rows=preview_rows
                    ),
                    "task_name": tname,
                    "family": fam.name,
                    "seed": seed,
                    "split": split,
                    "variant": variant,
                    "truth_json": json.dumps(truth_doc),
                    "files_json": json.dumps({"files": ep.files, "note": ep.note}),
                }
            )

    if shuffle and len(records) > 1:
        rng = np.random.default_rng(shuffle_seed)
        order = rng.permutation(len(records))
        records = [records[int(i)] for i in order]
    return records


def completion_text(completion: Any) -> str:
    """Normalize a TRL completion (raw string or chat-message list) into text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [
            str(m.get("content", ""))
            for m in completion
            if isinstance(m, dict) and m.get("content") is not None
        ]
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def extract_python_code(completion: Any) -> str | None:
    """Extract the final Python code block from a model completion.

    Prefers code blocks outside ``<think>...</think>`` tags so scratch snippets
    inside reasoning traces do not shadow the final script.
    """
    text = completion_text(completion)
    stripped = _THINK_BLOCK_RE.sub("", text)
    for candidate in (stripped, text):
        blocks = _PYTHON_FENCE_RE.findall(candidate)
        if blocks:
            return blocks[-1].strip()
    for candidate in (stripped, text):
        blocks = _ANY_FENCE_RE.findall(candidate)
        if blocks:
            return blocks[-1].strip()
    return None


@dataclass
class ExecutionOutcome:
    valid: bool
    code: str | None
    exec_result: dict[str, Any] | None
    scored: dict[str, Any] | None
    error: str | None = None


def execute_completion(
    completion: Any,
    files_payload: dict[str, Any] | str,
    truth_doc: dict[str, Any] | str,
    *,
    timeout_sec: float = 8.0,
) -> ExecutionOutcome:
    """Run the Python code in a completion inside an isolated workspace and score it."""
    code = extract_python_code(completion)
    if not code:
        return ExecutionOutcome(
            valid=False,
            code=None,
            exec_result=None,
            scored=None,
            error="no python code block found",
        )

    payload = json.loads(files_payload) if isinstance(files_payload, str) else files_payload
    truth = json.loads(truth_doc) if isinstance(truth_doc, str) else truth_doc

    with tempfile.TemporaryDirectory(prefix="pullgym_rl_") as tmp:
        root = Path(tmp)
        data_dir = root / FAST_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        for fname, text in payload["files"].items():
            (data_dir / fname).write_text(text, encoding="utf-8")
        (data_dir / "apparatus_note.md").write_text(payload["note"], encoding="utf-8")
        _set_readonly(data_dir)

        answer_path = root / FAST_ANSWER_PATH
        answer_path.parent.mkdir(parents=True, exist_ok=True)
        ws = EpisodeWorkspace(
            root=root,
            data_dir=data_dir,
            answer_path=answer_path,
            tool_timeout_sec=timeout_sec,
        )
        try:
            exec_res = ws.run_python(code, timeout=timeout_sec)
            scored = scoring.score_episode(str(answer_path), truth)
        finally:
            _set_writable(root)

    valid = bool(scored["details"]["valid"])
    err = scored["details"].get("error")
    if not valid and exec_res.get("timed_out"):
        err = f"python timed out after {timeout_sec}s"
    elif not valid and exec_res.get("exit_code", 0) != 0 and exec_res.get("stderr"):
        err = f"{err} | stderr: {exec_res['stderr'][-300:]}"
    return ExecutionOutcome(
        valid=valid,
        code=code,
        exec_result=exec_res,
        scored=scored,
        error=err,
    )


def _execute_batch(
    completions: list[Any],
    files_json: list[str],
    truth_json: list[str],
    *,
    timeout_sec: float = 8.0,
    max_workers: int = 8,
) -> list[ExecutionOutcome]:
    n = len(completions)
    if n == 0:
        return []
    workers = max(1, min(n, max_workers))
    if workers == 1:
        return [
            execute_completion(c, f, t, timeout_sec=timeout_sec)
            for c, f, t in zip(completions, files_json, truth_json)
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(execute_completion, c, f, t, timeout_sec=timeout_sec)
            for c, f, t in zip(completions, files_json, truth_json)
        ]
        return [fut.result() for fut in futs]


def binary_point_reward(
    outcome: ExecutionOutcome,
    *,
    k: float = 1.0,
    invalid_penalty: float = -1.0,
) -> float:
    """Arm A reward: +1.0 if valid and |q50 - truth| <= k * ref_sigma, 0.0 else."""
    if not outcome.valid or outcome.scored is None:
        return float(invalid_penalty)
    qs = outcome.scored["details"]["quantities"]
    hits = []
    for spec in qs.values():
        q50 = spec["answer"][2]
        hits.append(1.0 if abs(q50 - spec["truth"]) <= k * spec["ref_sigma"] else 0.0)
    return float(sum(hits) / len(hits))


def distributional_log_score_reward(
    outcome: ExecutionOutcome,
    *,
    floor: float = -10.0,
    cap: float = scoring.REWARD_CAP,
    invalid_penalty: float = -12.0,
) -> float:
    """Arm B reward: strictly proper clipped log-density score.

    ``invalid_penalty`` is set strictly below ``floor`` so RL can never prefer
    crashing or emitting malformed JSON over reporting an honest wide interval.
    """
    if invalid_penalty >= floor:
        raise ValueError("invalid_penalty must be strictly less than floor")
    if not outcome.valid or outcome.scored is None:
        return float(invalid_penalty)
    qs = outcome.scored["details"]["quantities"]
    vals = []
    for spec in qs.values():
        raw = scoring.log_score_reward(spec["answer"], spec["truth"], spec["ref_sigma"])
        vals.append(max(floor, min(cap, raw)))
    return float(sum(vals) / len(vals))


def make_binary_reward_fn(
    *,
    k: float = 1.0,
    invalid_penalty: float = -1.0,
    timeout_sec: float = 8.0,
    max_workers: int = 8,
) -> Callable[..., list[float]]:
    """Build a TRL ``GRPOTrainer``-compatible binary point-accuracy reward function."""

    def pullgym_binary_reward(
        completions: list[Any],
        files_json: list[str],
        truth_json: list[str],
        **kwargs: Any,
    ) -> list[float]:
        outcomes = _execute_batch(
            completions,
            files_json,
            truth_json,
            timeout_sec=timeout_sec,
            max_workers=max_workers,
        )
        return [
            binary_point_reward(oc, k=k, invalid_penalty=invalid_penalty)
            for oc in outcomes
        ]

    return pullgym_binary_reward


def make_log_score_reward_fn(
    *,
    floor: float = -10.0,
    cap: float = scoring.REWARD_CAP,
    invalid_penalty: float = -12.0,
    timeout_sec: float = 8.0,
    max_workers: int = 8,
) -> Callable[..., list[float]]:
    """Build a TRL ``GRPOTrainer``-compatible proper log-score reward function."""
    if invalid_penalty >= floor:
        raise ValueError("invalid_penalty must be strictly less than floor")

    def pullgym_log_score_reward(
        completions: list[Any],
        files_json: list[str],
        truth_json: list[str],
        **kwargs: Any,
    ) -> list[float]:
        outcomes = _execute_batch(
            completions,
            files_json,
            truth_json,
            timeout_sec=timeout_sec,
            max_workers=max_workers,
        )
        return [
            distributional_log_score_reward(
                oc, floor=floor, cap=cap, invalid_penalty=invalid_penalty
            )
            for oc in outcomes
        ]

    return pullgym_log_score_reward


def evaluate_completions_to_trials(
    records: list[dict[str, Any]],
    completions: list[Any],
    out_dir: Path,
    *,
    job_id: str,
    model_name: str,
    agent_name: str = "pullgym-rl",
    timeout_sec: float = 15.0,
    max_workers: int = 8,
) -> Path:
    """Score a batch of single-turn RL completions and write Harbor-compatible trials.

    Allows evaluating hundreds of episodes from ``vLLM.generate(...)`` in seconds
    and immediately tabulating results with ``pullgym.report.report(out_dir)``.
    """
    if len(records) != len(completions):
        raise ValueError("records and completions must have the same length")

    t0 = time.monotonic()
    outcomes = _execute_batch(
        completions,
        [r["files_json"] for r in records],
        [r["truth_json"] for r in records],
        timeout_sec=timeout_sec,
        max_workers=max_workers,
    )
    elapsed_per_ep = (time.monotonic() - t0) / max(len(records), 1)

    job_dir = Path(out_dir) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    agent_block = {"name": agent_name, "model_name": model_name}

    for rec, comp, oc in zip(records, completions, outcomes):
        trial_dir = job_dir / rec["task_name"]
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)

        truth_doc = json.loads(rec["truth_json"])
        if oc.scored is None:
            # No code block -> score missing answer via official fallback penalty.
            scored = scoring.score_episode(str(trial_dir / "_missing_answer.json"), truth_doc)
        else:
            scored = oc.scored

        cfg = {
            "task": {"name": f"pullgym/{rec['task_name']}", "family": rec["family"]},
            "trial_name": rec["task_name"],
            "job_id": job_id,
            "split": rec["split"],
            "variant": rec["variant"],
            "seed": rec["seed"],
            "agent": agent_block,
        }
        (trial_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (verifier_dir / "reward.json").write_text(
            json.dumps(scored["rewards"]), encoding="utf-8"
        )
        (verifier_dir / "details.json").write_text(
            json.dumps(scored["details"], indent=2), encoding="utf-8"
        )
        result_doc = {
            "harness_version": __version__,
            "config": {"agent": agent_block},
            "agent_info": {"name": agent_name, "model_info": {"name": model_name}},
            "timing": {"elapsed_sec": round(elapsed_per_ep, 3)},
            "rewards": scored["rewards"],
        }
        (trial_dir / "result.json").write_text(
            json.dumps(result_doc, indent=2), encoding="utf-8"
        )
        transcript = list(rec["prompt"]) + [
            {"role": "assistant", "content": completion_text(comp)}
        ]
        (trial_dir / "transcript.json").write_text(
            json.dumps(transcript, indent=2), encoding="utf-8"
        )
        log_msg = f"valid={oc.valid} error={oc.error}"
        (trial_dir / "trial.log").write_text(log_msg + "\n", encoding="utf-8")

    return job_dir
