import copy
import numpy as np
import torch

from kaggrl.structured_actor_critic import StructuredActorCritic, export_numpy
from kaggrl.structured_numpy_runtime import StructuredNumpyPolicy
from kaggrl.structured_ppo import evaluate_semantic_actions, gae, ppo_update


def test_numpy_sample_logp_matches_torch_semantic_evaluator(tmp_path):
    torch.manual_seed(41)
    model = StructuredActorCritic(1024, 32)
    model.eval()
    path = tmp_path / "m.npz"
    export_numpy(model, path)
    runtime = StructuredNumpyPolicy.load(path)
    obs = np.random.default_rng(4).standard_normal(1024).astype(np.float32)
    tokens, mask, logp_np, _, _ = runtime.sample_step(
        obs, runtime.initial_state(), hand_count=3,
        rng=np.random.default_rng(5), deterministic=False,
    )
    obs_t = torch.from_numpy(obs).view(1, 1, -1)
    act_t = torch.from_numpy(tokens.astype(np.int64)).view(1, 1, -1)
    mask_t = torch.from_numpy(mask.astype(np.float32)).view(1, 1, -1)
    logp_t, _, _ = evaluate_semantic_actions(model, obs_t, act_t, mask_t)
    assert np.isclose(float(logp_t.item()), logp_np, atol=2e-5)


def test_gae_stops_bootstrap_at_terminal():
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.2, 0.7])
    done = torch.tensor([0.0, 1.0])
    returns, adv = gae(rewards, values, done, gamma=1.0, lam=1.0)
    assert torch.allclose(returns, torch.tensor([1.0, 1.0]), atol=1e-6)
    assert torch.allclose(adv, returns - values, atol=1e-6)

def test_ppo_update_changes_learner_but_not_anchor():
    torch.manual_seed(7)
    learner = StructuredActorCritic(1024, 32)
    anchor = copy.deepcopy(learner).eval()
    before_anchor = {k: v.clone() for k, v in anchor.state_dict().items()}
    obs = torch.randn(1, 6, 1024)
    with torch.no_grad():
        actions, _, _ = learner.greedy_sequence(obs)
    masks = torch.zeros_like(actions, dtype=torch.float32)
    masks[..., 0] = 1.0
    masks[..., 3:51:3] = 1.0
    masks[..., 51::3] = 1.0
    old_logp, values, _ = evaluate_semantic_actions(learner, obs, actions, masks)
    rewards = torch.tensor([[0., 0., 0., .1, 0., 1.]])
    done = torch.tensor([[0., 0., 0., 0., 0., 1.]])
    returns, adv = gae(rewards[0], values.detach()[0], done[0])
    before = learner.farmer_op.weight.detach().clone()
    opt = torch.optim.Adam(learner.parameters(), lr=1e-3)
    metrics = ppo_update(
        learner, opt, obs, actions, masks, old_logp.detach(),
        returns.view(1, -1), adv.view(1, -1), anchor, bc_coef=0.05,
    )
    assert all(np.isfinite(float(v)) for v in metrics.values())
    assert not torch.equal(before, learner.farmer_op.weight.detach())
    for key, value in before_anchor.items():
        assert torch.equal(anchor.state_dict()[key], value)
    assert metrics["anchor_kl"] >= 0.0