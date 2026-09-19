from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.distributions import Categorical


def _forward_chunks(model, obs, actions, memory_horizon=48):
    outputs = {}
    values = []
    steps = obs.shape[1]
    horizon = int(memory_horizon) if memory_horizon else steps
    for start in range(0, steps, horizon):
        end = min(steps, start + horizon)
        out, value, _ = model.forward_actor_critic(
            obs[:, start:end], teacher_actions=actions[:, start:end], state=None
        )
        for key, tensor in out.items():
            outputs.setdefault(key, []).append(tensor)
        values.append(value)
    return {k: torch.cat(v, dim=1) for k, v in outputs.items()}, torch.cat(values, dim=1)


def _stats(logits, target, active, exclude_zero=False):
    if exclude_zero:
        logits = logits[..., 1:]
        target = (target - 1).clamp_min(0)
    dist = Categorical(logits=logits)
    active = active.to(logits.dtype)
    return dist.log_prob(target) * active, dist.entropy() * active

def _semantic_stats(outputs, actions, masks):
    batch, steps, _ = actions.shape
    logp = torch.zeros(batch, steps, device=actions.device, dtype=torch.float32)
    entropy = torch.zeros_like(logp)

    farmer = actions[..., :3]
    farmer_mask = masks[..., :3]
    for key, index, exclude_zero in (
        ("farmer_op", 0, False),
        ("farmer_item", 1, True),
        ("farmer_qty", 2, True),
    ):
        lp, ent = _stats(outputs[key], farmer[..., index], farmer_mask[..., index], exclude_zero)
        logp += lp; entropy += ent

    hands = actions[..., 3:51].view(batch, steps, 16, 3)
    hand_mask = masks[..., 3:51].view(batch, steps, 16, 3)
    for key, index, exclude_zero in (
        ("hand_op", 0, False),
        ("hand_item", 1, True),
        ("hand_qty", 2, True),
    ):
        lp, ent = _stats(outputs[key], hands[..., index], hand_mask[..., index], exclude_zero)
        logp += lp.sum(-1); entropy += ent.sum(-1)

    market = actions[..., 51:].view(batch, steps, 10, 3)
    market_mask = masks[..., 51:].view(batch, steps, 10, 3)
    for key, index, exclude_zero in (
        ("market_op", 0, False),
        ("market_item", 1, True),
        ("market_qty", 2, True),
    ):
        lp, ent = _stats(outputs[key], market[..., index], market_mask[..., index], exclude_zero)
        logp += lp.sum(-1); entropy += ent.sum(-1)
    return logp, entropy


def evaluate_semantic_actions(model, obs, actions, masks, memory_horizon=48):
    outputs, values = _forward_chunks(model, obs, actions, memory_horizon)
    logp, entropy = _semantic_stats(outputs, actions, masks)
    return logp, values, entropy

def gae(rewards, values, done, gamma=0.995, lam=0.95):
    rewards = rewards.to(dtype=torch.float32)
    values = values.to(dtype=torch.float32)
    done = done.to(dtype=torch.float32)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - done[t]
        next_value = values[t + 1] if t + 1 < values.shape[0] else torch.zeros_like(values[t])
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        advantages[t] = running
    return advantages + values, advantages


def _kl_component(learner, anchor, active, exclude_zero=False):
    if exclude_zero:
        learner = learner[..., 1:]
        anchor = anchor[..., 1:]
    kl = F.kl_div(
        F.log_softmax(learner, dim=-1),
        F.softmax(anchor, dim=-1),
        reduction="none",
    ).sum(-1)
    return kl * active.to(kl.dtype)


def _masked_anchor_kl(learner, anchor, masks):
    batch, steps, _ = masks.shape
    total = torch.zeros((), device=masks.device)
    count = torch.zeros((), device=masks.device)
    farmer_mask = masks[..., :3]
    for key, idx, ex0 in (("farmer_op",0,False),("farmer_item",1,True),("farmer_qty",2,True)):
        active = farmer_mask[..., idx]
        total += _kl_component(learner[key], anchor[key], active, ex0).sum()
        count += active.sum()
    hand_mask = masks[..., 3:51].view(batch, steps, 16, 3)
    for key, idx, ex0 in (("hand_op",0,False),("hand_item",1,True),("hand_qty",2,True)):
        active = hand_mask[..., idx]
        total += _kl_component(learner[key], anchor[key], active, ex0).sum()
        count += active.sum()
    market_mask = masks[..., 51:].view(batch, steps, 10, 3)
    for key, idx, ex0 in (("market_op",0,False),("market_item",1,True),("market_qty",2,True)):
        active = market_mask[..., idx]
        total += _kl_component(learner[key], anchor[key], active, ex0).sum()
        count += active.sum()
    return (total / count.clamp_min(1.0)).clamp_min(0.0)

def ppo_update(
    model,
    optimizer,
    obs,
    actions,
    masks,
    old_logp,
    returns,
    advantages,
    bc_anchor_model,
    *,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=1e-3,
    bc_coef=0.05,
    max_grad_norm=0.5,
    memory_horizon=48,
):
    new_logp, values, entropy = evaluate_semantic_actions(
        model, obs, actions, masks, memory_horizon=memory_horizon
    )
    adv = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy_mean = entropy.mean()

    bc_anchor_model.eval()
    with torch.no_grad():
        anchor_outputs, _ = _forward_chunks(
            bc_anchor_model, obs, actions, memory_horizon=memory_horizon
        )
    learner_outputs, _ = _forward_chunks(
        model, obs, actions, memory_horizon=memory_horizon
    )
    anchor_kl = _masked_anchor_kl(learner_outputs, anchor_outputs, masks)
    loss = (
        policy_loss
        + float(value_coef) * value_loss
        - float(entropy_coef) * entropy_mean
        + float(bc_coef) * anchor_kl
    )
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
        "entropy": float(entropy_mean.detach()),
        "anchor_kl": float(anchor_kl.detach()),
        "approx_kl": float(approx_kl),
        "clip_fraction": float(clip_fraction),
        "grad_norm": float(torch.as_tensor(grad_norm).detach()),
    }
