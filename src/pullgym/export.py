"""Build Harbor task folders from a family and a list of seeds.

Each task folder is self-contained:

    <task>/
      task.toml
      instruction.md
      environment/Dockerfile        agent image: Python with numpy/scipy, data in /app/data
      environment/data/...          the simulated data and the apparatus note
      tests/Dockerfile              verifier image (separate mode): stdlib Python only
      tests/test.sh                 runs scoring.py on /logs/artifacts/answer.json
      tests/scoring.py              copied from pullgym.scoring
      tests/truth.json              the hidden truth: never in the agent's image
      solution/solve.sh             writes the reference analysis's answer (oracle)

The verifier runs in a fresh sandbox; the only file it takes from the agent's
run is /logs/artifacts/answer.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pullgym import scoring
from pullgym.families import Family

AGENT_BASE_IMAGE = "python:3.12-slim"
AGENT_PIP = "numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 matplotlib==3.9.2"
ANSWER_PATH = "/logs/artifacts/answer.json"
DATA_DIR = "/app/data"

VARIANTS = {
    "calibrated": (
        "Your answer is scored on calibration. Over many experiments like this one, "
        "the true value should fall between your `q16` and `q84` 68% of the time, and "
        "between `q025` and `q975` 95% of the time. Intervals that are too narrow and "
        "intervals that are too wide both lose score."
    ),
    "neutral": "",
}


def heldout_seeds(secret: str, family: str, n: int) -> list[int]:
    """Seeds for the held-out split, derived from a private secret."""
    seeds = []
    for i in range(n):
        h = hashlib.sha256(f"{secret}:{family}:{i}".encode()).digest()
        seeds.append(int.from_bytes(h[:8], "big") >> 1)
    return seeds


def task_id(family: Family, split: str, seed: int) -> str:
    if split == "dev":
        return f"{family.name}-dev-{seed:05d}"
    # Opaque id for held-out tasks, so the seed cannot be read off the name.
    return f"{family.name}-{split}-{hashlib.sha256(str(seed).encode()).hexdigest()[:10]}"


def instruction(family: Family, variant: str) -> str:
    q_lines = "\n".join(
        f"- `{q}`: {spec.description}, in {spec.unit}" for q, spec in family.quantities.items()
    )
    example_q = next(iter(family.quantities))
    extra = VARIANTS[variant]
    parts = [
        f"# {family.title}\n",
        family.task + "\n",
        f"The data files and an apparatus note are in `{DATA_DIR}`. Read the apparatus "
        "note before you start. Python 3 with numpy, scipy, pandas and matplotlib is "
        "available. Analyse the data however you see fit.\n",
        f"## What to report\n\n{q_lines}\n",
        "For each quantity, report your result as five quantiles of your uncertainty "
        "about the true value: the 2.5%, 16%, 50%, 84% and 97.5% points. A result "
        "`x ± s` with a Gaussian uncertainty corresponds to `x-2s, x-s, x, x+s, x+2s`. "
        "The quantiles must be strictly increasing.\n",
    ]
    if extra:
        parts.append(extra + "\n")
    parts.append(
        f"Write the answer to `{ANSWER_PATH}` (create the directory if needed), in "
        "exactly this form, with numbers only:\n\n"
        "```json\n"
        + json.dumps(
            {example_q: {k: f"<number>" for k in scoring.KEYS}}, indent=2
        ).replace('"<number>"', "<number>")
        + "\n```\n"
    )
    return "\n".join(parts)


def task_toml(family: Family, name: str, split: str, variant: str, seed: int) -> str:
    meta = {"family": family.name, "tier": family.tier, "split": split, "variant": variant}
    if split == "dev":
        meta["seed"] = seed
    meta_lines = "\n".join(f"{k} = {json.dumps(v)}" for k, v in meta.items())
    return f"""schema_version = "1.4"

[task]
name = "pullgym/{name}"
version = "0.1.0"
description = {json.dumps(family.title)}
keywords = ["pullgym", "calibration", "uncertainty", "physics"]

[metadata]
{meta_lines}

[verifier]
timeout_sec = 120.0
environment_mode = "separate"

[agent]
timeout_sec = 1800.0

[environment]
build_timeout_sec = 900.0
cpus = 1
memory_mb = 2048
"""


def export_task(
    family: Family, seed: int, split: str, variant: str, out_dir: Path
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {', '.join(VARIANTS)}")
    ep = family.episode(seed)
    ref = family.reference_analysis(ep.files)
    prior = family.prior_quantiles()
    name = task_id(family, split, seed)
    root = out_dir / name
    if root.exists():
        shutil.rmtree(root)
    (root / "environment" / "data").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "solution").mkdir()

    (root / "task.toml").write_text(task_toml(family, name, split, variant, seed))
    (root / "instruction.md").write_text(instruction(family, variant))

    for fname, text in ep.files.items():
        (root / "environment" / "data" / fname).write_text(text)
    (root / "environment" / "data" / "apparatus_note.md").write_text(ep.note)
    (root / "environment" / "Dockerfile").write_text(
        f"FROM {AGENT_BASE_IMAGE}\n"
        f"RUN pip install --no-cache-dir {AGENT_PIP}\n"
        "WORKDIR /app\n"
        f"COPY data {DATA_DIR}\n"
        "RUN mkdir -p /logs/artifacts\n"
    )

    truth_doc = {
        "family": family.name,
        "task": name,
        "split": split,
        "variant": variant,
        "quantities": {
            q: {
                "truth": ep.truth[q],
                "unit": family.quantities[q].unit,
                "ref_sigma": scoring.sigma_from_quantiles(ref[q]),
                "reference": ref[q],
                "fallback": prior[q],
            }
            for q in family.quantities
        },
    }
    if split == "dev":
        truth_doc["seed"] = seed
    (root / "tests" / "truth.json").write_text(json.dumps(truth_doc, indent=2))
    shutil.copyfile(scoring.__file__, root / "tests" / "scoring.py")
    (root / "tests" / "Dockerfile").write_text(
        f"FROM {AGENT_BASE_IMAGE}\nWORKDIR /tests\nCOPY . /tests/\n"
    )
    test_sh = root / "tests" / "test.sh"
    test_sh.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        "mkdir -p /logs/verifier\n"
        f"python3 /tests/scoring.py --answer {ANSWER_PATH} "
        "--truth /tests/truth.json --out /logs/verifier\n"
    )
    test_sh.chmod(0o755)

    answer = {q: dict(zip(scoring.KEYS, ref[q])) for q in family.quantities}
    solve = root / "solution" / "solve.sh"
    solve.write_text(
        "#!/bin/bash\n"
        "# Oracle: the reference analysis's answer for this episode.\n"
        "mkdir -p /logs/artifacts\n"
        f"cat > {ANSWER_PATH} <<'EOF'\n{json.dumps(answer, indent=2)}\nEOF\n"
    )
    solve.chmod(0o755)
    return root
