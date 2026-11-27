@AGENTS.md

# CLAUDE.md — Quick Reference for Claude Code

Read and follow all architectural rules, metrological contracts, and RLVR invariants in `AGENTS.md` (imported above). If a local gitignored roadmap file such as `plan-v0.2.0.md` exists in the workspace root, read it before implementing `v0.2.0` tasks.

## Critical Guardrails Summary
- **`src/pullgym/scoring.py` must remain 100% Python stdlib-only** (copied verbatim into Harbor's isolated verifier container; zero `numpy`/`scipy`/`pullgym` imports). Keep calibration keys omitted from `reward.json` when `valid` is `False`.
- **Zero truth leakage:** Never expose `truth`, `ref_sigma`, or secret seeds in `data/`, `apparatus_note.md`, subprocess env vars, or live `answer_status` messages (`scoring.load_answer` schema validation only).
- **Family design:** `reference_analysis(files)` must be 100% deterministic (`< 30 ms`/episode, zero unseeded MC), avoid multi-GB 4D grids, include zero glitch/drift spoilers in `apparatus_note.md` for unprompted diagnostic families, and pass `uv run pytest -m slow`, `uv run pullgym calibrate --n 2000`, and `uv run pullgym probe --n 1000`.
- **RLVR rigor:** Preserve `invalid_penalty < floor` (`-12.0 < -10.0`), keep `TRAIN_FAMILIES` (8 families) strictly disjoint from `OOD_TEST_FAMILIES`, build multi-turn `(completion_ids, action_mask)` turn-by-turn (`action_mask = 0` on observation tokens) with `loss_type="dr_grpo"`, and audit ID Tier-1 sharpness (`≈ 1.00`) + `pinball_skill` + `--paired` CIs.
- **Git & anonymity hygiene:** Never stage or commit author PII, `.DS_Store`, `plan-*.md`, `paper*/`, `*.secret`, `runs/`, `tasks/`, or `jobs/`.

## Verification Commands
```bash
uv run pytest -q
uv run pytest -m slow
uv run pullgym calibrate --n 2000
uv run pullgym probe --n 1000
```
