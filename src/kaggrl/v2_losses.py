from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_ledger import MARKET_OPS
from .v2_quantity import encode_quantity
from .v3_behavior import behavior_family

UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)
MARKET_ACTIVE_OPS = ("NOP_SLOT", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
UNIT_OP_TO_ID = {op: i for i, op in enumerate(UNIT_OPS)}
MARKET_OP_TO_ID = {op: i for i, op in enumerate(MARKET_OPS)}
MARKET_ACTIVE_OP_TO_ID = {op: i for i, op in enumerate(MARKET_ACTIVE_OPS)}
SEMANTIC_ITEM_WEIGHT = 0.5
SEMANTIC_QUANTITY_WEIGHT = 0.25

# A false STOP at an early market slot removes every later order from rollout.
# Weight only the STOP/CONTINUE decision by the amount of active teacher queue
# still behind the current slot. Active-op/item/quantity objectives stay unchanged.
MARKET_QUEUE_SURVIVAL_STEP = 0.35
MARKET_QUEUE_SURVIVAL_CAP = 3.0
# Episode starts are only 0.14% of the accepted training rows, while the
# reference policies always continue the market queue at step 0. Without a
# phase correction the global STOP prior wins at rollout startup. This
# multiplier affects only the hierarchical STOP/CONTINUE term; active-op,
# item and quantity objectives remain unchanged.
EARLY_MARKET_CONTINUE_WEIGHT = {0: 4.0, 1: 2.0}
# Step-0 market op is a perfectly consistent strategy signature in the
# accepted corpus (8/8 episodes per team), but only 80/57,520 train rows.
# Amplify only the active-op CE for slot 0 at step 0 so strategy slots learn
# distinct openings without distorting item/quantity or later-game policy.
EARLY_MARKET_ACTIVE_WEIGHT = {0: 32.0}

DEFAULT_LOSS_WEIGHTS = {
    "action": 1.0,
    "effect": 0.25,
    "future_resource": 0.15,
    "unit_task": 0.15,
    "opponent_effect": 0.10,
    "value": 0.05,
}


@dataclass
class DomainLosses:
    farmer: torch.Tensor
    hands: torch.Tensor
    market: torch.Tensor
    total: torch.Tensor


def _market_op(action: dict[str, Any]) -> str:
    kind = str(action.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(action.get("op", "NOP_SLOT"))


def _ce(logits: torch.Tensor, target: int) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError(f"expected rank-1 logits, got {tuple(logits.shape)}")
    target_tensor = torch.tensor(
        [int(target)], dtype=torch.long, device=logits.device,
    )
    return F.cross_entropy(logits.unsqueeze(0), target_tensor)


def _quantity_loss(logits: torch.Tensor | None, quantity: int | None) -> torch.Tensor:
    tokens = encode_quantity(quantity)
    if logits is None:
        raise ValueError("active quantity target has no quantity logits")
    if logits.ndim != 2 or logits.shape[0] < len(tokens):
        raise ValueError("quantity logits shorter than canonical token stream")
    target = torch.tensor(tokens, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits[: len(tokens)], target)


def _semantic_mean(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("cannot average an empty semantic loss")
    return torch.stack(parts).mean()


def _family_weight(family_weights, domain: str, family: str) -> float:
    if family_weights is None:
        return 1.0
    scoped = family_weights.get(domain)
    if isinstance(scoped, dict):
        return float(scoped.get(family, 1.0))
    return float(family_weights.get(family, 1.0))


def _unit_decision_loss(
    decision, target: dict[str, Any], family_weights=None,
) -> torch.Tensor:
    op = str(target.get("op", "PASS"))
    if op not in UNIT_OP_TO_ID:
        raise ValueError(f"unknown unit op target: {op}")
    family = behavior_family(target, "unit")
    weight = _family_weight(family_weights, "unit", family)
    loss = weight * _ce(decision.op_logits, UNIT_OP_TO_ID[op])
    if op in UNIT_ITEM_OPS:
        item = target.get("item")
        if item not in ITEM_TO_ID:
            raise ValueError(f"active unit item target is invalid: {item}")
        loss = loss + SEMANTIC_ITEM_WEIGHT * _ce(
            decision.item_logits, ITEM_TO_ID[item],
        )
    if op in UNIT_QUANTITY_OPS:
        loss = loss + SEMANTIC_QUANTITY_WEIGHT * _quantity_loss(
            decision.quantity_logits, target.get("quantity"),
        )
    return loss


def _market_active_target_id(logits: torch.Tensor, op: str) -> int:
    if op not in MARKET_ACTIVE_OP_TO_ID:
        raise ValueError(f"market active target is not an active operation: {op}")
    if int(logits.numel()) == len(MARKET_ACTIVE_OPS):
        return MARKET_ACTIVE_OP_TO_ID[op]
    if int(logits.numel()) == len(MARKET_OPS):
        return MARKET_OP_TO_ID[op]
    raise ValueError(
        "market active-op logits width does not match active or legacy vocabulary"
    )


def _market_active_semantic_loss(
    decision, target: dict[str, Any], op: str, family_weights=None,
    market_active_op_weights=None, active_weight: float = 1.0,
) -> torch.Tensor:
    family = behavior_family(target, "market")
    weight = (
        float(market_active_op_weights.get(op, 1.0))
        if market_active_op_weights is not None
        else _family_weight(family_weights, "market", family)
    )
    loss = max(1.0, float(active_weight)) * weight * _ce(
        decision.op_logits, _market_active_target_id(decision.op_logits, op),
    )
    if op in MARKET_ITEM_OPS:
        item = target.get("item")
        if item not in ITEM_TO_ID:
            raise ValueError(f"active market item target is invalid: {item}")
        loss = loss + SEMANTIC_ITEM_WEIGHT * _ce(
            decision.item_logits, ITEM_TO_ID[item],
        )
    if op in MARKET_QUANTITY_OPS:
        loss = loss + SEMANTIC_QUANTITY_WEIGHT * _quantity_loss(
            decision.quantity_logits, target.get("quantity"),
        )
    return loss


def _market_decision_loss(
    decision, target: dict[str, Any], family_weights=None,
    continue_weight: float = 1.0, market_active_op_weights=None,
    active_weight: float = 1.0,
) -> torch.Tensor:
    op = _market_op(target)
    if op not in MARKET_OP_TO_ID:
        raise ValueError(f"unknown market op target: {op}")

    continue_logits = getattr(decision, "continue_logits", None)
    if continue_logits is not None:
        continue_target = 0 if op == "STOP_QUEUE" else 1
        weight = max(1.0, float(continue_weight))
        loss = weight * _ce(continue_logits, continue_target)
        if op == "STOP_QUEUE":
            return loss
        return loss + _market_active_semantic_loss(
            decision, target, op, family_weights,
            market_active_op_weights=market_active_op_weights,
            active_weight=active_weight,
        )

    family = behavior_family(target, "market")
    weight = _family_weight(family_weights, "market", family)
    loss = weight * _ce(decision.op_logits, MARKET_OP_TO_ID[op])
    if op in MARKET_ITEM_OPS:
        item = target.get("item")
        if item not in ITEM_TO_ID:
            raise ValueError(f"active market item target is invalid: {item}")
        loss = loss + SEMANTIC_ITEM_WEIGHT * _ce(
            decision.item_logits, ITEM_TO_ID[item],
        )
    if op in MARKET_QUANTITY_OPS:
        loss = loss + SEMANTIC_QUANTITY_WEIGHT * _quantity_loss(
            decision.quantity_logits, target.get("quantity"),
        )
    return loss


def _queue_survival_weight(
    market_targets: list[dict[str, Any]], index: int,
) -> float:
    if index < 0 or index >= len(market_targets):
        raise IndexError("market target index out of range")
    if _market_op(market_targets[index]) == "STOP_QUEUE":
        return 1.0
    future_active = sum(
        1
        for later in market_targets[index + 1:]
        if _market_op(later) != "STOP_QUEUE"
    )
    return min(
        MARKET_QUEUE_SURVIVAL_CAP,
        1.0 + MARKET_QUEUE_SURVIVAL_STEP * float(future_active),
    )


def _market_phase_continue_weight(step: int) -> float:
    return float(EARLY_MARKET_CONTINUE_WEIGHT.get(max(0, int(step)), 1.0))


def _market_phase_active_weight(step: int) -> float:
    return float(EARLY_MARKET_ACTIVE_WEIGHT.get(max(0, int(step)), 1.0))


def _structured_step(state: Any) -> int:
    if isinstance(state, dict):
        return int(state.get("step", 0) or 0)
    return int(getattr(state, "step", 0) or 0)


def _weighted_active_mean(
    values: list[torch.Tensor], weights: torch.Tensor, active: list[bool],
) -> torch.Tensor:
    if len(values) != len(active) or len(values) != int(weights.numel()):
        raise ValueError("domain loss/weight batch mismatch")
    indices = [i for i, flag in enumerate(active) if flag]
    if not indices:
        return values[0].sum() * 0.0
    numer = sum(values[i] * weights[i] for i in indices)
    denom = weights[indices].sum().clamp_min(torch.finfo(weights.dtype).eps)
    return numer / denom


def _weighted_decision_mean(
    values_by_row: list[list[torch.Tensor]], weights: torch.Tensor,
) -> torch.Tensor:
    if len(values_by_row) != int(weights.numel()):
        raise ValueError("decision loss/weight batch mismatch")
    ref = next(
        (value for row in values_by_row for value in row),
        None,
    )
    if ref is None:
        raise ValueError("cannot average an empty decision loss domain")
    numer = ref.sum() * 0.0
    denom = weights.sum() * 0.0
    for row_index, row in enumerate(values_by_row):
        row_weight = weights[row_index]
        for value in row:
            numer = numer + value * row_weight
            denom = denom + row_weight
    return numer / denom.clamp_min(torch.finfo(weights.dtype).eps)


def action_loss(
    outputs, targets, masks=None, sample_weight=None, family_weights=None,
    market_continue_row_weight=None, market_active_op_weights=None,
    market_active_row_weight=None,
) -> DomainLosses:
    del masks
    rows = tuple(outputs.rows)
    targets = tuple(targets)
    if len(rows) != len(targets):
        raise ValueError("output/target batch mismatch")
    if not rows:
        raise ValueError("action loss requires at least one row")
    ref = rows[0].farmer.op_logits
    if sample_weight is None:
        weights = torch.ones(len(rows), device=ref.device, dtype=ref.dtype)
    else:
        weights = torch.as_tensor(sample_weight, device=ref.device, dtype=ref.dtype).reshape(-1)
    if weights.numel() != len(rows):
        raise ValueError("sample_weight must have one value per row")
    if market_continue_row_weight is None:
        continue_row_weights = [1.0] * len(rows)
    else:
        continue_row_weights = [float(value) for value in market_continue_row_weight]
        if len(continue_row_weights) != len(rows):
            raise ValueError("market_continue_row_weight must have one value per row")
        if any(value < 1.0 for value in continue_row_weights):
            raise ValueError("market_continue_row_weight values must be >= 1")

    if market_active_row_weight is None:
        active_row_weights = [1.0] * len(rows)
    else:
        active_row_weights = [float(value) for value in market_active_row_weight]
        if len(active_row_weights) != len(rows):
            raise ValueError("market_active_row_weight must have one value per row")
        if any(value < 1.0 for value in active_row_weights):
            raise ValueError("market_active_row_weight values must be >= 1")

    farmer_rows: list[torch.Tensor] = []
    hand_rows: list[torch.Tensor] = []
    market_rows: list[torch.Tensor] = []
    market_decision_rows: list[list[torch.Tensor]] = []
    hand_active: list[bool] = []
    market_active: list[bool] = []
    for row_index, (row, target) in enumerate(zip(rows, targets)):
        farmer_rows.append(_unit_decision_loss(
            row.farmer, target["farmer"], family_weights,
        ))

        hand_targets = list(target.get("hands") or [])
        if len(row.hands) != len(hand_targets):
            raise ValueError("hand output/target count mismatch")
        if hand_targets:
            hand_rows.append(_semantic_mean([
                _unit_decision_loss(decision, truth, family_weights)
                for decision, truth in zip(row.hands, hand_targets)
            ]))
            hand_active.append(True)
        else:
            hand_rows.append(ref.sum() * 0.0)
            hand_active.append(False)

        market_targets = list(target.get("market") or [])
        if len(row.market) != len(market_targets):
            raise ValueError("market output/target count mismatch")
        if market_targets:
            decision_losses = [
                _market_decision_loss(
                    decision,
                    truth,
                    family_weights,
                    continue_weight=(
                        _queue_survival_weight(market_targets, index)
                        * (continue_row_weights[row_index] if index == 0 else 1.0)
                    ),
                    market_active_op_weights=(
                        {}
                        if index == 0 and active_row_weights[row_index] > 1.0
                        else market_active_op_weights
                    ),
                    active_weight=(
                        active_row_weights[row_index] if index == 0 else 1.0
                    ),
                )
                for index, (decision, truth) in enumerate(
                    zip(row.market, market_targets)
                )
            ]
            market_rows.append(_semantic_mean(decision_losses))
            market_decision_rows.append(decision_losses)
            market_active.append(True)
        else:
            market_rows.append(ref.sum() * 0.0)
            market_decision_rows.append([])
            market_active.append(False)
    farmer = _weighted_active_mean(farmer_rows, weights, [True] * len(rows))
    hands = _weighted_active_mean(hand_rows, weights, hand_active)
    market = (
        _weighted_decision_mean(market_decision_rows, weights)
        if any(market_active) else ref.sum() * 0.0
    )

    domains = [farmer]
    if any(hand_active):
        domains.append(hands)
    if any(market_active):
        domains.append(market)
    total = _semantic_mean(domains)
    return DomainLosses(farmer=farmer, hands=hands, market=market, total=total)


def _target_dict(batch) -> dict[str, torch.Tensor]:
    value = getattr(batch, "auxiliary_targets", None)
    if not isinstance(value, dict):
        raise ValueError("batch.auxiliary_targets is required for pretraining loss")
    return value


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError(
            f"auxiliary shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    return F.mse_loss(prediction, target)


def _masked_unit_mse(
    prediction: torch.Tensor, target: torch.Tensor, unit_mask: torch.Tensor,
) -> torch.Tensor:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError("unit-task prediction/target shape mismatch")
    if prediction.ndim < 3 or unit_mask.shape != prediction.shape[:2]:
        raise ValueError("unit-task mask shape mismatch")
    per_unit = (prediction - target).square().flatten(2).mean(dim=-1)
    mask = unit_mask.to(device=prediction.device, dtype=prediction.dtype)
    per_row = (per_unit * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    active = mask.sum(dim=1) > 0
    return per_row[active].mean() if active.any() else per_unit.sum() * 0.0


def total_pretrain_loss(
    outputs, batch, weights: dict[str, float] | None = None,
    family_weights: dict[str, float] | None = None,
    market_active_op_weights: dict[str, float] | None = None,
):
    weights = dict(DEFAULT_LOSS_WEIGHTS if weights is None else weights)
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError("pretrain loss weights must define the frozen objective components")

    targets = _target_dict(batch)
    ref = outputs.aux.effect
    sample_weight = getattr(batch, "sample_weight", None)
    structured_states = getattr(batch, "structured_states", None)
    market_continue_row_weight = None
    market_active_row_weight = None
    if structured_states is not None:
        market_continue_row_weight = [
            _market_phase_continue_weight(_structured_step(state))
            for state in structured_states
        ]
        market_active_row_weight = [
            _market_phase_active_weight(_structured_step(state))
            for state in structured_states
        ]
    domains = action_loss(
        outputs, batch.canonical_actions, None, sample_weight,
        family_weights=family_weights,
        market_continue_row_weight=market_continue_row_weight,
        market_active_op_weights=market_active_op_weights,
        market_active_row_weight=market_active_row_weight,
    )
    effect = _mse(outputs.aux.effect, targets["effect"])
    future_resource = _mse(
        outputs.aux.future_resource, targets["future_resource"]
    )
    unit_task = _masked_unit_mse(
        outputs.aux.unit_task,
        targets["unit_task"],
        batch.own_unit_mask,
    )
    opponent_effect = _mse(
        outputs.aux.opponent_effect, targets["opponent_effect"]
    )
    terminal_money = _mse(
        outputs.aux.terminal_money, targets["terminal_money"]
    )
    terminal_margin = _mse(
        outputs.aux.terminal_margin, targets["terminal_margin"]
    )
    value = 0.5 * (terminal_money + terminal_margin)

    result = {
        "action": domains.total,
        "farmer": domains.farmer,
        "hands": domains.hands,
        "market": domains.market,
        "effect": effect,
        "future_resource": future_resource,
        "unit_task": unit_task,
        "opponent_effect": opponent_effect,
        "value": value,
    }
    result["total"] = sum(
        float(weights[name]) * result[name]
        for name in DEFAULT_LOSS_WEIGHTS
    )
    return result
