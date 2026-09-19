from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

OMIT_ID = 0
DIGIT_OFFSET = 1
END_ID = 11
VOCAB_SIZE = 12
START_ID = 12
DEFAULT_MAX_DIGITS = 24


def _digit_id(value: int) -> int:
    return DIGIT_OFFSET + int(value)


def _id_digit(token: int) -> int | None:
    value = int(token) - DIGIT_OFFSET
    return value if 0 <= value <= 9 else None


def encode_quantity(value: int | None) -> tuple[int, ...]:
    if value is None:
        return (OMIT_ID,)
    value = int(value)
    if value < 0:
        raise ValueError("canonical neural quantity must be nonnegative")
    return tuple(_digit_id(int(ch)) for ch in str(value)) + (END_ID,)


def decode_quantity(tokens: Sequence[int], *, max_digits: int = DEFAULT_MAX_DIGITS) -> int | None:
    values = [int(x) for x in tokens]
    if not values:
        raise ValueError("empty quantity token stream")
    if values[0] == OMIT_ID:
        if len(values) != 1:
            raise ValueError("OMIT must be the only quantity token")
        return None
    digits: list[str] = []
    for index, token in enumerate(values):
        if token == END_ID:
            if not digits:
                raise ValueError("END cannot terminate an empty quantity")
            if index != len(values) - 1:
                raise ValueError("tokens after END are not allowed")
            return int("".join(digits))
        digit = _id_digit(token)
        if digit is None:
            raise ValueError(f"invalid quantity token: {token}")
        digits.append(str(digit))
        if len(digits) > int(max_digits):
            raise ValueError("maximum digit count exceeded")
    raise ValueError("unterminated quantity digit stream")


class QuantityDecoder(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int = 96, token_dim: int = 32,
                 max_digits: int = DEFAULT_MAX_DIGITS):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.max_digits = int(max_digits)
        self.init = nn.Linear(int(context_dim), int(hidden_dim))
        self.embedding = nn.Embedding(VOCAB_SIZE + 1, int(token_dim))
        self.gru = nn.GRUCell(int(token_dim), int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), VOCAB_SIZE)

    def initial_state(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.init(context))

    def step(self, hidden: torch.Tensor, previous_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_hidden = self.gru(self.embedding(previous_token.long()), hidden)
        return self.output(next_hidden), next_hidden

    def teacher_logits_tensor(
        self,
        context: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("quantity context must be [batch, features]")
        if targets.ndim != 2 or target_mask.shape != targets.shape:
            raise ValueError("tensor quantity targets/mask must be [batch, steps]")
        if targets.shape[0] != context.shape[0]:
            raise ValueError("quantity target batch mismatch")
        if targets.shape[1] <= 0:
            raise ValueError("quantity tensor target width must be positive")
        targets = targets.to(device=context.device, dtype=torch.long)
        target_mask = target_mask.to(device=context.device, dtype=torch.bool)
        hidden = self.initial_state(context)
        previous = torch.full(
            (context.shape[0],), START_ID, dtype=torch.long, device=context.device
        )
        logits_steps = []
        for step_index in range(targets.shape[1]):
            logits, next_hidden = self.step(hidden, previous)
            active = target_mask[:, step_index]
            logits_steps.append(logits)
            hidden = torch.where(active.unsqueeze(-1), next_hidden, hidden)
            previous = torch.where(active, targets[:, step_index], previous)
        return torch.stack(logits_steps, dim=1), target_mask

    def teacher_logits(
        self,
        context: torch.Tensor,
        targets: Sequence[Sequence[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError("quantity context must be [batch, features]")
        if len(targets) != context.shape[0]:
            raise ValueError("quantity target batch mismatch")
        lengths = [len(tuple(tokens)) for tokens in targets]
        if not lengths or max(lengths) == 0:
            raise ValueError("quantity targets must be non-empty")
        max_len = max(lengths)
        hidden = self.initial_state(context)
        previous = torch.full(
            (context.shape[0],), START_ID, dtype=torch.long, device=context.device
        )
        logits_steps = []
        mask_steps = []
        for step_index in range(max_len):
            logits, next_hidden = self.step(hidden, previous)
            active = torch.tensor(
                [step_index < length for length in lengths],
                dtype=torch.bool, device=context.device,
            )
            logits_steps.append(logits)
            mask_steps.append(active)
            hidden = torch.where(active.unsqueeze(-1), next_hidden, hidden)
            next_previous = previous.clone()
            for row, tokens in enumerate(targets):
                if step_index < lengths[row]:
                    next_previous[row] = int(tokens[step_index])
            previous = next_previous
        return torch.stack(logits_steps, dim=1), torch.stack(mask_steps, dim=1)
