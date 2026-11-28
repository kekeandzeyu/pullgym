"""CPU tests for Phase 5 multi-turn rollouts (no GPU, no vLLM: a fake server)."""

import json
import re

import pytest

from pullgym import scoring
from pullgym.families import get_family
from pullgym.rl import TRAIN_FAMILIES, DEFAULT_TRAIN_SEED_START
from pullgym.rl_multiturn import (
    MT_OBS_TRUNCATE,
    TEMPLATE_PATH,
    Segment,
    TurnTokens,
    build_segments,
    check_segments,
    group_advantages,
    rl_reward,
    rollout_episode,
    train_schedule,
)

# ---------------------------------------------------------------------------
# A toy tokenizer and a fake vLLM server that renders like qwen3_nothink_mt.jinja.

IM_START, IM_END = 2, 1
SPECIAL = {"<|im_start|>": IM_START, "<|im_end|>": IM_END}
_SPECIAL_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")


def tok(text: str) -> list[int]:
    ids: list[int] = []
    for part in _SPECIAL_RE.split(text):
        if part in SPECIAL:
            ids.append(SPECIAL[part])
        else:
            ids.extend(ord(c) + 10 for c in part)
    return ids


def detok(ids: list[int]) -> str:
    inv = {v: k for k, v in SPECIAL.items()}
    return "".join(inv.get(i, chr(i - 10) if i >= 10 else "?") for i in ids)


THINK = "<think>\n\n</think>\n\n"


def render(messages, *, keep_think: bool = True) -> str:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            body = (THINK if keep_think else "") + m["content"]
        else:
            body = m["content"]
        out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
    return "".join(out) + "<|im_start|>assistant\n" + THINK


class FakeServer:
    def __init__(self, replies, *, keep_think=True, token_ids=True):
        self.replies = list(replies)
        self.keep_think = keep_think
        self.token_ids = token_ids
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        text = self.replies.pop(0)
        choice = {"message": {"content": text}, "finish_reason": "stop"}
        resp = {"choices": [choice], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        if self.token_ids:
            choice["token_ids"] = tok(text)  # vLLM-style: stop token left out
            resp["prompt_token_ids"] = tok(render(payload["messages"], keep_think=self.keep_think))
        return resp


def _answer_code(family_name: str, seed: int) -> str:
    fam = get_family(family_name)
    ep = fam.episode(seed)
    ref = fam.reference_analysis(ep.files)
    answer = {q: dict(zip(scoring.KEYS, ref[q])) for q in fam.quantities}
    return (
        "```python\nimport json, os\n"
        "print('x' * 3000)\n"  # long output: must be truncated to head 400 + tail 800
        "os.makedirs('artifacts', exist_ok=True)\n"
        f"json.dump({answer!r}, open('artifacts/answer.json', 'w'))\n```"
    )


# ---------------------------------------------------------------------------


def test_template_file_matches_fake_render():
    jinja2 = pytest.importorskip("jinja2")
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "F"},
    ]
    tpl = jinja2.Environment().from_string(TEMPLATE_PATH.read_text())
    assert tpl.render(messages=msgs, add_generation_prompt=True) == render(msgs)


def test_build_segments_masks_observations_and_restores_eos():
    p1 = [5, 6, 7]
    c1 = [20, 21]
    obs = [IM_END, 8, 9]
    p2 = p1 + c1 + obs
    c2 = [30, IM_END]
    segs, breaks = build_segments(
        [TurnTokens(p1, c1, "stop"), TurnTokens(p2[:5] + p2[5:], c2, "stop")], IM_END
    )
    # c1 had no stop token: it is appended as an action token, and the next
    # prompt's copy of it is then part of the shared prefix.
    assert breaks == 0 and len(segs) == 1
    s = segs[0]
    assert s.input_ids == p1 + c1 + [IM_END] + [8, 9] + c2
    assert s.action_mask == [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    check_segments(segs)


def test_build_segments_starts_new_segment_on_prefix_break():
    t1 = TurnTokens([5, 6], [20], "length")  # truncated: no stop token appended
    t2 = TurnTokens([5, 99, 7], [30], "stop")  # history re-rendered differently
    segs, breaks = build_segments([t1, t2], IM_END)
    assert breaks == 1 and len(segs) == 2
    assert segs[0].action_mask == [0, 0, 1]
    assert segs[1].input_ids == [5, 99, 7, 30, IM_END]
    assert segs[1].action_mask == [0, 0, 0, 1, 1]


def test_rollout_episode_end_to_end_one_segment(tmp_path):
    fam, seed = "line_fit", DEFAULT_TRAIN_SEED_START
    server = FakeServer([_answer_code(fam, seed), "Done."])
    ro = rollout_episode(
        fam, seed, server, tmp_path, model="policy", arm="log_score", eos_id=IM_END,
        extra_body={"top_k": 20},
    )
    assert ro.error is None and ro.valid and ro.turns == 2
    # The exact reference answer scores near 0 under the clipped log score.
    details = json.loads((ro.trial_dir / "verifier" / "details.json").read_text())
    assert ro.reward == pytest.approx(rl_reward(details, "log_score"))
    assert ro.reward > -3.0
    assert rl_reward(details, "binary") in (0.0, 1.0)

    assert ro.prefix_breaks == 0 and len(ro.segments) == 1
    seg = ro.segments[0]
    check_segments(ro.segments)
    # Every action token is model output; observation text is fully masked.
    action_text = detok([i for i, m in zip(seg.input_ids, seg.action_mask) if m])
    assert action_text == server_replies_with_eos(fam, seed)
    masked_text = detok([i for i, m in zip(seg.input_ids, seg.action_mask) if not m])
    assert "answer_status" in masked_text and "turn: 1/4" in masked_text
    assert "chars omitted" in masked_text  # STDOUT truncated as in training
    head, tail = MT_OBS_TRUNCATE
    assert "x" * (head + tail) not in masked_text

    # Same request shape as run-fast code mode, plus token ids.
    p = server.payloads[0]
    assert p["return_token_ids"] is True and p["max_tokens"] == 1280 and p["top_k"] == 20
    assert "tools" not in p
    # No truth or reference width leaks into what the environment showed the policy.
    for spec in details["quantities"].values():
        assert repr(spec["truth"]) not in masked_text
        assert repr(spec["ref_sigma"]) not in masked_text


def server_replies_with_eos(fam, seed):
    return _answer_code(fam, seed) + "<|im_end|>" + "Done." + "<|im_end|>"


def test_rollout_with_stock_template_history_splits_segments(tmp_path):
    fam, seed = "weighted_mean", DEFAULT_TRAIN_SEED_START + 1
    server = FakeServer([_answer_code(fam, seed), "Done."], keep_think=False)
    ro = rollout_episode(fam, seed, server, tmp_path, model="p", arm="binary", eos_id=IM_END)
    assert ro.error is None and ro.prefix_breaks == 1 and len(ro.segments) == 2
    check_segments(ro.segments)


def test_rollout_without_token_ids_is_an_error_not_a_reward(tmp_path):
    fam, seed = "line_fit", DEFAULT_TRAIN_SEED_START + 2
    server = FakeServer([_answer_code(fam, seed)], token_ids=False)
    ro = rollout_episode(fam, seed, server, tmp_path, model="p", arm="log_score", eos_id=IM_END)
    assert ro.error and ro.reward is None and not ro.segments


def test_missing_answer_gets_invalid_penalty(tmp_path):
    fam, seed = "line_fit", DEFAULT_TRAIN_SEED_START + 3
    server = FakeServer(["I give up."])
    ro = rollout_episode(fam, seed, server, tmp_path, model="p", arm="log_score", eos_id=IM_END)
    assert ro.error is None and not ro.valid and ro.reward == -12.0 and ro.turns == 1
    ro_b = rollout_episode(
        fam, seed, FakeServer(["no"]), tmp_path / "b", model="p", arm="binary", eos_id=IM_END
    )
    assert ro_b.reward == -1.0


def test_group_advantages():
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    a = group_advantages([-12.0, -1.0, 0.0, 0.5])
    assert sum(a) == pytest.approx(0.0, abs=1e-9)
    assert sum(x * x for x in a) / len(a) == pytest.approx(1.0, rel=1e-3)
    raw = group_advantages([0.0, 2.0], scale=False)
    assert raw == [-1.0, 1.0]


def test_train_schedule_is_deterministic_and_in_train_range():
    s1 = train_schedule(30, 4, shuffle_seed=1)
    assert s1 == train_schedule(30, 4, shuffle_seed=1)
    assert s1 != train_schedule(30, 4, shuffle_seed=2)
    flat = [x for step in s1 for x in step]
    assert len(flat) == 120 and len(set(flat)) == 120
    assert all(f in TRAIN_FAMILIES and s >= DEFAULT_TRAIN_SEED_START for f, s in flat)
    # Wraps into a new epoch without error.
    assert len(train_schedule(600, 4, shuffle_seed=0)) == 600


def test_segment_n_action():
    assert Segment([1, 2, 3], [0, 1, 1]).n_action == 2
