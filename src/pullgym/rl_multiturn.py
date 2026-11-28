"""Phase 5: multi-turn RL rollouts through the exact `pullgym run-fast` code-mode harness.

A training rollout is a normal `run-fast --tool-mode code` episode: the same
`OpenAICompatibleAgent` loop, persistent workspace, feedback text, turn notes
and termination rule. The only extra is that the chat function asks the vLLM
OpenAI server for token ids (`return_token_ids`) and records, for every turn,
the exact prompt tokens the server rendered and the tokens it sampled. Evaluation
is plain `pullgym run-fast` against the same server and chat template, so
training and evaluation see identical prompts.

From the recorded turns, `build_segments` assembles each episode's training
sequence turn by turn: `action_mask = 1` on sampled tokens and `0` on the
prompt and every environment observation. Before appending turn k it checks
that the server's turn-k prompt extends the stream so far token for token. If
it does not (non-canonical sampled tokenization, a template that rewrites
history), a new segment starts from the turn-k prompt, so the policy is never
trained on a context it did not see.

Rewards are terminal and computed from the scored episode (`details.json`) with
the controlled RL arms of `pullgym.rl`; nothing derived from the truth ever
reaches the policy.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pullgym.families import Family, get_family
from pullgym.rl import (
    DEFAULT_TRAIN_SEED_START,
    TRAIN_FAMILIES,
    ExecutionOutcome,
    binary_point_reward,
    distributional_log_score_reward,
)
from pullgym.runner import AgentError, ChatFn, OpenAICompatibleAgent, run_episode

# Phase 5 harness constants, shared by training rollouts and evaluation.
MT_MAX_TURNS = 4
MT_MAX_TOKENS = 1280
MT_OBS_TRUNCATE = (400, 800)
MT_TOOL_TIMEOUT_SEC = 15.0
TEMPLATE_PATH = Path(__file__).parent / "templates" / "qwen3_nothink_mt.jinja"

ARMS = ("binary", "log_score")


@dataclass
class TurnTokens:
    prompt_ids: list[int]
    completion_ids: list[int]
    finish_reason: str | None


@dataclass
class Segment:
    """One training sequence: tokens plus a mask that is 1 on sampled tokens only."""

    input_ids: list[int]
    action_mask: list[int]
    n_turns: int = 0

    @property
    def n_action(self) -> int:
        return int(sum(self.action_mask))


def build_segments(turns: list[TurnTokens], eos_id: int) -> tuple[list[Segment], int]:
    """Assemble an episode's training segments; returns (segments, n_prefix_breaks)."""
    segments: list[Segment] = []
    cur: Segment | None = None
    breaks = 0
    for t in turns:
        prompt = list(t.prompt_ids)
        if cur is not None and prompt[: len(cur.input_ids)] == cur.input_ids:
            obs = prompt[len(cur.input_ids):]
            cur.input_ids.extend(obs)
            cur.action_mask.extend([0] * len(obs))
        else:
            if cur is not None:
                segments.append(cur)
                breaks += 1
            cur = Segment(input_ids=prompt, action_mask=[0] * len(prompt))
        comp = list(t.completion_ids)
        cur.input_ids.extend(comp)
        cur.action_mask.extend([1] * len(comp))
        # vLLM may leave the stop token out of token_ids; the chat template puts
        # it back after the content, and the model did choose to stop here.
        if t.finish_reason == "stop" and (not comp or comp[-1] != eos_id):
            cur.input_ids.append(eos_id)
            cur.action_mask.append(1)
        cur.n_turns += 1
    if cur is not None:
        segments.append(cur)
    return segments, breaks


class TokenRecordingChat:
    """Wrap a chat function so every turn's prompt/completion token ids are kept."""

    def __init__(self, base: ChatFn) -> None:
        self.base = base
        self.turns: list[TurnTokens] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.base({**payload, "return_token_ids": True})
        choices = resp.get("choices") or []
        if not choices:
            raise AgentError("model response contained no choices")
        ch = choices[0]
        prompt_ids = resp.get("prompt_token_ids") or ch.get("prompt_token_ids")
        comp_ids = ch.get("token_ids")
        if prompt_ids is None or comp_ids is None:
            raise AgentError(
                "server returned no token ids; Phase 5 needs a vLLM server that "
                "supports return_token_ids"
            )
        self.turns.append(TurnTokens(list(prompt_ids), list(comp_ids), ch.get("finish_reason")))
        return resp


def rl_reward(
    details: dict[str, Any],
    arm: str,
    *,
    floor: float = -10.0,
    invalid_penalty: float = -12.0,
    binary_k: float = 1.0,
    binary_invalid: float = -1.0,
) -> float:
    """Terminal RL reward of a scored episode (`verifier/details.json`)."""
    oc = ExecutionOutcome(
        valid=bool(details.get("valid")), code=None, exec_result=None, scored={"details": details}
    )
    if arm == "log_score":
        return distributional_log_score_reward(oc, floor=floor, invalid_penalty=invalid_penalty)
    if arm == "binary":
        return binary_point_reward(oc, k=binary_k, invalid_penalty=binary_invalid)
    raise ValueError(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")


def group_advantages(rewards: list[float], *, scale: bool = True, eps: float = 1e-4) -> list[float]:
    """GRPO advantages: reward minus group mean, divided by the group std when
    `scale` (so both reward arms train with unit-scale advantages)."""
    r = np.asarray(rewards, dtype=float)
    if r.size == 0:
        return []
    a = r - r.mean()
    sd = r.std()
    if sd < 1e-8:
        return [0.0] * r.size
    if scale:
        a = a / (sd + eps)
    return [float(x) for x in a]


@dataclass
class Rollout:
    family: str
    seed: int
    trial_dir: Path
    segments: list[Segment] = field(default_factory=list)
    prefix_breaks: int = 0
    turns: int = 0
    valid: bool = False
    reward: float | None = None
    error: str | None = None


_PRIORS: dict[str, dict[str, list[float]]] = {}
_PRIOR_LOCK = threading.Lock()


def _prior(family: Family) -> dict[str, list[float]]:
    with _PRIOR_LOCK:
        if family.name not in _PRIORS:
            _PRIORS[family.name] = family.prior_quantiles()
        return _PRIORS[family.name]


def rollout_episode(
    family_name: str,
    seed: int,
    chat_fn: ChatFn,
    job_dir: Path,
    *,
    model: str,
    arm: str,
    eos_id: int,
    variant: str = "calibrated",
    split: str = "train",
    temperature: float = 0.85,
    extra_body: dict[str, Any] | None = None,
    max_turns: int = MT_MAX_TURNS,
    max_tokens: int = MT_MAX_TOKENS,
    obs_truncate: tuple[int, int] = MT_OBS_TRUNCATE,
    tool_timeout_sec: float = MT_TOOL_TIMEOUT_SEC,
) -> Rollout:
    """Run one run-fast code-mode episode and return its training segments and reward.

    Episodes that fail for infrastructure reasons (server error on turn 1,
    lost connection) come back with `error` set and no reward: never train on them.
    """
    fam = get_family(family_name)
    rec = TokenRecordingChat(chat_fn)
    agent = OpenAICompatibleAgent(
        model,
        chat_fn=rec,
        max_turns=max_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_mode="code",
        extra_body=extra_body,
        obs_truncate=obs_truncate,
    )
    oc = run_episode(
        fam, seed, split, variant, agent, Path(job_dir),
        prior=_prior(fam), tool_timeout_sec=tool_timeout_sec,
    )
    out = Rollout(family=family_name, seed=seed, trial_dir=oc.trial_dir, turns=len(rec.turns))
    if oc.error:
        out.error = oc.error
        return out
    if not rec.turns:
        out.error = "no model turns recorded"
        return out
    details = json.loads((oc.trial_dir / "verifier" / "details.json").read_text(encoding="utf-8"))
    out.valid = bool(details.get("valid"))
    out.reward = rl_reward(details, arm)
    out.segments, out.prefix_breaks = build_segments(rec.turns, eos_id)
    return out


def train_schedule(
    n_steps: int,
    prompts_per_step: int,
    *,
    shuffle_seed: int,
    families: tuple[str, ...] = TRAIN_FAMILIES,
    n_per_family: int = 250,
    seed_start: int = DEFAULT_TRAIN_SEED_START,
) -> list[list[tuple[str, int]]]:
    """Deterministic (family, seed) prompts per step from the RL training seed range.

    Cycles through a shuffled pool of `families x n_per_family` training tasks
    (reshuffled each epoch), as GRPO R2 did with its shuffled dataset.
    """
    pool = [(f, s) for f in families for s in range(seed_start, seed_start + n_per_family)]
    rng = np.random.default_rng(shuffle_seed)
    order: list[tuple[str, int]] = []
    need = n_steps * prompts_per_step
    while len(order) < need:
        order.extend(pool[int(i)] for i in rng.permutation(len(pool)))
    return [order[i * prompts_per_step:(i + 1) * prompts_per_step] for i in range(n_steps)]


def sampling_extra(
    *, top_p: float, top_k: int, presence_penalty: float, seed: int | None = None
) -> dict[str, Any]:
    """vLLM sampling fields for the OpenAI chat request (shared by training and eval)."""
    body: dict[str, Any] = {
        "top_p": top_p,
        "top_k": top_k,
        "presence_penalty": presence_penalty,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = seed
    return body


def check_segments(segments: list[Segment]) -> None:
    """Invariants a trainer relies on; raises AssertionError on violation."""
    for s in segments:
        assert len(s.input_ids) == len(s.action_mask)
        assert set(s.action_mask) <= {0, 1}
        assert s.action_mask and s.action_mask[0] == 0, "a segment must start with prompt tokens"

