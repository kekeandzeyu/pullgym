"""CPU checks of the Phase 5 trainer's loss on a tiny random Qwen3 (skipped without torch)."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

_spec = importlib.util.spec_from_file_location(
    "train_grpo_mt", Path(__file__).parents[1] / "scripts" / "train_grpo_mt.py"
)
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)


def _tiny_policy(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
    )
    model = transformers.AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    model = tg.wrap_lora(model, 4)
    # Non-zero LoRA B so adapter on/off differ (fresh LoRA starts at B = 0).
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    return model


IDS = [3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 14, 1]
MASK = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]


def test_action_logprobs_match_full_logits():
    model = _tiny_policy()
    model.eval()
    with torch.no_grad():
        got = tg.action_logprobs(model, IDS, MASK, temperature=0.85, chunk=2)
        logits = model(input_ids=torch.tensor([IDS])).logits[0].float() / 0.85
        lp = torch.log_softmax(logits, -1)
        want = torch.stack([lp[t - 1, IDS[t]] for t in range(1, len(IDS)) if MASK[t]])
    assert got.shape == (sum(MASK),)
    assert torch.allclose(got, want, atol=1e-5)


def test_first_token_cannot_be_action():
    model = _tiny_policy()
    with pytest.raises(ValueError):
        tg.action_logprobs(model, IDS, [1] + MASK[1:], temperature=1.0)


@pytest.mark.parametrize("beta", [0.0, 0.05])
def test_positive_advantage_step_raises_action_logprob(beta):
    model = _tiny_policy()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    opt = torch.optim.SGD(params, lr=0.1)
    with torch.no_grad():
        before = tg.action_logprobs(model, IDS, MASK, temperature=1.0).sum()
    s = tg.segment_backward(model, IDS, MASK, 1.0, temperature=1.0, beta=beta, norm=10.0, chunk=4)
    assert s["n_tok"] == sum(MASK)
    assert (s["kl"] > 0) == (beta > 0)
    # Observation tokens get no gradient signal: only action positions are scored.
    opt.step()
    with torch.no_grad():
        after = tg.action_logprobs(model, IDS, MASK, temperature=1.0).sum()
    assert after > before


def test_parse_args_defaults_match_phase5_harness():
    a = tg.parse_args(["--arm", "binary", "--seed", "2"])
    assert (a.max_turns, a.max_tokens, a.prompts_per_step, a.num_generations) == (4, 1280, 4, 8)
