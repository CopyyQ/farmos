from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Categorical


class RecurrentActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, token_width: int, vocab_size: int = 128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_width = int(token_width)
        self.vocab_size = int(vocab_size)
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.actor = nn.Linear(self.hidden_dim, self.token_width * self.vocab_size)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def initial_state(self, batch_size: int, device=None):
        p = next(self.parameters())
        device = device or p.device
        z = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=p.dtype)
        return z.clone(), z.clone()
    def forward_sequence(self, obs: torch.Tensor, state=None):
        if obs.ndim != 3 or obs.shape[-1] != self.input_dim:
            raise ValueError(f"expected [B,T,{self.input_dim}], got {tuple(obs.shape)}")
        batch, steps, _ = obs.shape
        h, c = state if state is not None else self.initial_state(batch, obs.device)
        logits_seq, values_seq = [], []
        for t in range(steps):
            x = torch.tanh(self.input_proj(obs[:, t]))
            h, c = self.lstm(x, (h, c))
            logits = self.actor(h).view(batch, self.token_width, self.vocab_size)
            value = self.critic(h).squeeze(-1)
            logits_seq.append(logits)
            values_seq.append(value)
        return torch.stack(logits_seq, 1), torch.stack(values_seq, 1), (h, c)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values, _ = self.forward_sequence(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions).sum(dim=-1)
        return logp, values

    def entropy(self, obs: torch.Tensor):
        logits, _, _ = self.forward_sequence(obs)
        return Categorical(logits=logits).entropy().sum(dim=-1)
