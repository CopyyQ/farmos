import numpy as np
import torch

from kaggrl.actions import ActionCodec
from kaggrl.structured_actor_critic import StructuredActorCritic, export_numpy
from kaggrl.structured_numpy_runtime import StructuredNumpyPolicy


def _obs_for_hands(n=2):
    return {"player": 0, "farms": [{"hands": [[0, 0]] * n}]}


def test_numpy_export_matches_torch_greedy_tokens_value_and_state(tmp_path):
    torch.manual_seed(17)
    model = StructuredActorCritic(1024, 32)
    model.eval()
    obs = torch.randn(1, 4, 1024)
    with torch.inference_mode():
        tokens_t, values_t, state_t = model.greedy_sequence(obs)
    path = tmp_path / "policy.npz"
    export_numpy(model, path)
    runtime = StructuredNumpyPolicy.load(path)
    state_n = runtime.initial_state()
    got_tokens, got_values = [], []
    for t in range(obs.shape[1]):
        tokens, value, state_n = runtime.greedy_step(obs[0, t].numpy(), state_n)
        got_tokens.append(tokens); got_values.append(value)
    got_tokens = np.stack(got_tokens)
    assert np.array_equal(got_tokens, tokens_t[0].cpu().numpy())
    assert np.allclose(got_values, values_t[0].cpu().numpy(), atol=2e-5)
    assert np.allclose(state_n[0], state_t[0][0].cpu().numpy(), atol=2e-5)
    assert np.allclose(state_n[1], state_t[1][0].cpu().numpy(), atol=2e-5)

def test_loads_bc_checkpoint_without_mutating_policy_weights(tmp_path):
    torch.manual_seed(23)
    base = StructuredActorCritic(1024, 32)
    bc_state = {k: v.clone() for k, v in base.state_dict().items() if not k.startswith("value_head.")}
    ckpt = tmp_path / "bc.pt"
    torch.save({"model": bc_state, "input_dim": 1024, "hidden_dim": 32}, ckpt)
    loaded = StructuredActorCritic.from_bc_checkpoint(ckpt)
    for key, value in bc_state.items():
        assert torch.equal(loaded.state_dict()[key], value)
    assert torch.count_nonzero(loaded.value_head.weight) == 0
    assert torch.count_nonzero(loaded.value_head.bias) == 0


def test_runtime_decodes_a_valid_fixed_width_action(tmp_path):
    model = StructuredActorCritic(1024, 16)
    path = tmp_path / "policy.npz"
    export_numpy(model, path)
    runtime = StructuredNumpyPolicy.load(path)
    tokens, _, _ = runtime.greedy_step(np.zeros(1024, np.float32), runtime.initial_state())
    assert tokens.shape == (ActionCodec().width,)
    action = ActionCodec().decode(tokens, _obs_for_hands())
    assert set(action) == {"farmer", "hands", "market"}
    assert len(action["hands"]) == 2