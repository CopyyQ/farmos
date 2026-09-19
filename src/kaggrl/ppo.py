from __future__ import annotations
import torch


def ppo_update(model, optimizer, obs, actions, old_logp, returns, advantages,
               clip_ratio: float = 0.2, value_coef: float = 0.5,
               entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
    adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    new_logp, values = model.evaluate_actions(obs, actions)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy = model.entropy(obs).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    approx_kl = (old_logp - new_logp).mean().detach()
    clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean().detach()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "grad_norm": float(torch.as_tensor(grad_norm).detach()),
    }
