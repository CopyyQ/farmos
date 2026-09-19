import torch


def test_recurrent_actor_critic_shapes():
    from kaggrl.model import RecurrentActorCritic
    model = RecurrentActorCritic(input_dim=64, hidden_dim=32, token_width=9, vocab_size=16)
    obs = torch.randn(2, 5, 64)
    logits, values, state = model.forward_sequence(obs)
    assert logits.shape == (2, 5, 9, 16)
    assert values.shape == (2, 5)
    assert state[0].shape == (2, 32)
    assert state[1].shape == (2, 32)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_ppo_smoke_update_changes_parameters():
    from kaggrl.model import RecurrentActorCritic
    from kaggrl.ppo import ppo_update
    torch.manual_seed(7)
    model = RecurrentActorCritic(input_dim=32, hidden_dim=16, token_width=6, vocab_size=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    obs = torch.randn(4, 8, 32)
    actions = torch.randint(0, 12, (4, 8, 6))
    with torch.no_grad():
        old_logp, old_values = model.evaluate_actions(obs, actions)
    returns = old_values + 0.5 * torch.randn_like(old_values)
    advantages = torch.randn_like(old_values)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = ppo_update(model, optimizer, obs, actions, old_logp, returns, advantages)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters()))
