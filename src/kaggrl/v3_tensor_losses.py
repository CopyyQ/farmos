from __future__ import annotations

import torch
import torch.nn.functional as F

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_losses import DEFAULT_LOSS_WEIGHTS, OPTIONAL_ECONOMIC_LOSS_WEIGHT
from .v2_ledger import MARKET_OPS
from .v3_2_schema import ACTIVE_MARKET_OPS
from .v3_behavior import behavior_family
from .v3_tensor_targets import (
    MARKET_OP_TO_ID,
    UNIT_OP_TO_ID,
    TensorActionTargets,
)

UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)
ACTIVE_MARKET_OP_TO_ID = {op: i for i, op in enumerate(ACTIVE_MARKET_OPS)}
STOP_ID = MARKET_OP_TO_ID["STOP_QUEUE"]


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(values)
    mask_f = mask.to(values.dtype)
    numer = (values * weights * mask_f).sum()
    denom = (weights * mask_f).sum().clamp_min(
        torch.finfo(values.dtype).eps
    )
    return numer / denom


def _op_weight_lookup(
    *,
    domain: str,
    names,
    family_weights,
    device,
    dtype,
) -> torch.Tensor:
    values = []
    scoped = None
    op_scoped = None
    if family_weights is not None:
        candidate = family_weights.get(domain)
        scoped = candidate if isinstance(candidate, dict) else family_weights
        candidate_op = family_weights.get(f"{domain}_op")
        if isinstance(candidate_op, dict):
            op_scoped = candidate_op
    for name in names:
        family = behavior_family(
            {"op": name, "kind": name if name in {"STOP_QUEUE", "NOP_SLOT"} else "ORDER"},
            domain,
        )
        if op_scoped is not None and name in op_scoped:
            values.append(float(op_scoped[name]))
        else:
            values.append(
                1.0 if scoped is None else float(scoped.get(family, 1.0))
            )
    return torch.tensor(values, device=device, dtype=dtype)



def _masked_ce_per_decision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError("masked CE shape mismatch")
    result = torch.zeros(
        targets.shape,
        device=logits.device,
        dtype=torch.float32,
    )
    active = mask.to(device=logits.device, dtype=torch.bool)
    if not bool(active.any().item()):
        return result
    result[active] = F.cross_entropy(
        logits[active].float(),
        targets[active],
        reduction="none",
    )
    return result


def _quantity_loss_per_decision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    token_mask: torch.Tensor,
    decision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim < 3:
        raise ValueError("quantity logits must end in [tokens, vocab]")
    decision_shape = targets.shape[:-1]
    result = torch.zeros(
        decision_shape,
        device=logits.device,
        dtype=torch.float32,
    )
    active = (
        token_mask.any(dim=-1)
        if decision_mask is None
        else (decision_mask.to(torch.bool) & token_mask.any(dim=-1))
    )
    if not bool(active.any().item()):
        return result

    selected_logits = logits[active].float()
    selected_targets = targets[active]
    selected_mask = token_mask[active]
    vocab = selected_logits.shape[-1]
    flat_loss = F.cross_entropy(
        selected_logits.reshape(-1, vocab),
        selected_targets.reshape(-1),
        reduction="none",
    ).reshape(selected_targets.shape)
    mask = selected_mask.to(flat_loss.dtype)
    selected_loss = (
        (flat_loss * mask).sum(dim=-1)
        / mask.sum(dim=-1).clamp_min(1.0)
    )
    result[active] = selected_loss
    return result


def _unit_semantic_loss(
    op_logits: torch.Tensor,
    item_logits: torch.Tensor,
    quantity_logits: torch.Tensor,
    targets: TensorActionTargets,
    *,
    family_weights=None,
) -> torch.Tensor:
    # [B,U]
    op_target = targets.unit_op
    op_ce = _masked_ce_per_decision(
        op_logits,
        op_target,
        targets.unit_mask,
    )
    lookup = _op_weight_lookup(
        domain="unit",
        names=UNIT_OPS,
        family_weights=family_weights,
        device=op_logits.device,
        dtype=op_ce.dtype,
    )
    loss = op_ce * lookup[op_target]

    item_ids = torch.tensor(
        [UNIT_OP_TO_ID[name] for name in UNIT_ITEM_OPS],
        device=op_logits.device,
        dtype=torch.long,
    )
    needs_item = (
        op_target.unsqueeze(-1).eq(
            item_ids.view(1, 1, -1)
        ).any(dim=-1)
        & targets.unit_mask
    )
    item_ce = _masked_ce_per_decision(
        item_logits,
        targets.unit_item,
        needs_item,
    )
    loss = loss + 0.5 * item_ce

    quantity_ids = torch.tensor(
        [UNIT_OP_TO_ID[name] for name in UNIT_QUANTITY_OPS],
        device=op_logits.device,
        dtype=torch.long,
    )
    needs_quantity = (
        op_target.unsqueeze(-1).eq(
            quantity_ids.view(1, 1, -1)
        ).any(dim=-1)
        & targets.unit_mask
    )
    quantity_ce = _quantity_loss_per_decision(
        quantity_logits,
        targets.unit_quantity_tokens,
        targets.unit_quantity_token_mask,
        decision_mask=needs_quantity,
    )
    loss = loss + 0.25 * quantity_ce
    return loss


def _market_semantic_loss(
    outputs,
    targets: TensorActionTargets,
    *,
    step: torch.Tensor,
    family_weights=None,
    market_active_op_weights=None,
) -> torch.Tensor:
    # Returns [B,S] per market decision.
    market_op = targets.market_op
    is_stop = market_op.eq(STOP_ID)
    continue_target = (~is_stop).long()
    continue_ce = _masked_ce_per_decision(
        outputs.market_continue_logits,
        continue_target,
        targets.market_mask,
    )

    active_future = (
        targets.market_mask
        & ~is_stop
    ).to(torch.long)
    reverse_cumulative = torch.flip(
        torch.cumsum(torch.flip(active_future, dims=(1,)), dim=1),
        dims=(1,),
    )
    future_count = reverse_cumulative - active_future
    queue_weight = torch.minimum(
        torch.full_like(continue_ce, 3.0),
        1.0 + 0.35 * future_count.to(continue_ce.dtype),
    )
    phase_continue = torch.ones(
        market_op.shape[0],
        device=market_op.device,
        dtype=continue_ce.dtype,
    )
    phase_continue = torch.where(
        step.eq(0),
        torch.full_like(phase_continue, 4.0),
        phase_continue,
    )
    phase_continue = torch.where(
        step.eq(1),
        torch.full_like(phase_continue, 2.0),
        phase_continue,
    )
    continue_weight = queue_weight.clone()
    continue_weight[:, 0] *= phase_continue
    result = continue_ce * continue_weight

    active_index_lookup = torch.tensor(
        [int(ACTIVE_MARKET_OP_TO_ID.get(op, 0)) for op in MARKET_OPS],
        device=market_op.device,
        dtype=torch.long,
    )
    active_id = active_index_lookup[market_op]
    active_mask = targets.market_mask & ~is_stop
    active_ce = _masked_ce_per_decision(
        outputs.market_active_logits,
        active_id,
        active_mask,
    )

    active_lookup = torch.ones(
        len(ACTIVE_MARKET_OPS),
        device=active_ce.device,
        dtype=active_ce.dtype,
    )
    if market_active_op_weights is not None:
        active_lookup = torch.tensor(
            [
                float(market_active_op_weights.get(op, 1.0))
                for op in ACTIVE_MARKET_OPS
            ],
            device=active_ce.device,
            dtype=active_ce.dtype,
        )
    elif family_weights is not None:
        active_lookup = _op_weight_lookup(
            domain="market",
            names=ACTIVE_MARKET_OPS,
            family_weights=family_weights,
            device=active_ce.device,
            dtype=active_ce.dtype,
        )
    active_weight = active_lookup[active_id]
    step0 = step.eq(0)
    # Legacy objective disables class/family active-op reweighting for slot 0
    # when the step-0 32x opening multiplier is active.
    active_weight[:, 0] = torch.where(
        step0,
        torch.full_like(active_weight[:, 0], 32.0),
        active_weight[:, 0],
    )
    result = result + active_ce * active_weight

    item_ids = torch.tensor(
        [MARKET_OP_TO_ID[name] for name in MARKET_ITEM_OPS],
        device=market_op.device,
        dtype=torch.long,
    )
    needs_item = (
        market_op.unsqueeze(-1).eq(
            item_ids.view(1, 1, -1)
        ).any(dim=-1)
        & targets.market_mask
    )
    item_ce = _masked_ce_per_decision(
        outputs.market_item_logits,
        targets.market_item,
        needs_item,
    )
    result = result + 0.5 * item_ce

    quantity_ids = torch.tensor(
        [MARKET_OP_TO_ID[name] for name in MARKET_QUANTITY_OPS],
        device=market_op.device,
        dtype=torch.long,
    )
    needs_quantity = (
        market_op.unsqueeze(-1).eq(
            quantity_ids.view(1, 1, -1)
        ).any(dim=-1)
        & targets.market_mask
    )
    quantity_ce = _quantity_loss_per_decision(
        outputs.market_quantity_logits,
        targets.market_quantity_tokens,
        targets.market_quantity_token_mask,
        decision_mask=needs_quantity,
    )
    result = result + 0.25 * quantity_ce
    return result


def tensor_total_pretrain_loss(
    outputs,
    batch,
    targets: TensorActionTargets,
    *,
    step: torch.Tensor,
    weights: dict[str, float] | None = None,
    family_weights=None,
    market_active_op_weights=None,
):
    weights = dict(
        DEFAULT_LOSS_WEIGHTS if weights is None else weights
    )
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError(
            "pretrain loss weights must define frozen objective components"
        )
    device = outputs.unit_op_logits.device
    targets = targets.to(device)
    step = step.to(device=device, dtype=torch.long)
    sample_weight = getattr(batch, "sample_weight", None)
    if sample_weight is None:
        sample_weight = torch.ones(
            targets.batch_size,
            device=device,
            dtype=outputs.unit_op_logits.dtype,
        )
    else:
        sample_weight = sample_weight.to(
            device=device,
            dtype=outputs.unit_op_logits.dtype,
        )

    unit_loss = _unit_semantic_loss(
        outputs.unit_op_logits,
        outputs.unit_item_logits,
        outputs.unit_quantity_logits,
        targets,
        family_weights=family_weights,
    )
    farmer = _weighted_mean(
        unit_loss[:, 0],
        sample_weight,
        targets.unit_mask[:, 0],
    )
    weighted_domain_values = [4.0 * farmer]
    active_domain_count = 1
    if targets.max_units > 1:
        hand_mask = targets.unit_mask[:, 1:]
        hand_count = hand_mask.sum(dim=1)
        hand_row = (
            unit_loss[:, 1:] * hand_mask.to(unit_loss.dtype)
        ).sum(dim=1) / hand_count.clamp_min(1).to(unit_loss.dtype)
        hands = _weighted_mean(
            hand_row,
            sample_weight,
            hand_count.gt(0),
        )
        weighted_domain_values.append(hands)
        active_domain_count += 1
    else:
        hands = farmer * 0.0

    market_decision = _market_semantic_loss(
        outputs,
        targets,
        step=step,
        family_weights=family_weights,
        market_active_op_weights=market_active_op_weights,
    )
    market_weights = sample_weight.unsqueeze(1).expand_as(
        market_decision
    )
    market = _weighted_mean(
        market_decision,
        market_weights,
        targets.market_mask,
    )
    weighted_domain_values.append(2.0 * market)
    active_domain_count += 1
    action = torch.stack(weighted_domain_values).sum() / float(
        active_domain_count
    )

    aux_targets = getattr(batch, "auxiliary_targets", None)
    if not isinstance(aux_targets, dict):
        raise ValueError("batch.auxiliary_targets is required")

    def mse(prediction, name):
        prediction_fp32 = prediction.float()
        target = aux_targets[name].to(
            device=prediction.device,
            dtype=torch.float32,
        )
        return F.mse_loss(prediction_fp32, target)

    effect = mse(outputs.aux.effect, "effect")
    future_resource = mse(
        outputs.aux.future_resource, "future_resource"
    )
    opponent_effect = mse(
        outputs.aux.opponent_effect, "opponent_effect"
    )
    terminal_money = mse(
        outputs.aux.terminal_money, "terminal_money"
    )
    terminal_margin = mse(
        outputs.aux.terminal_margin, "terminal_margin"
    )
    unit_target = aux_targets["unit_task"].to(
        device=outputs.aux.unit_task.device,
        dtype=torch.float32,
    )
    per_unit = (
        outputs.aux.unit_task.float() - unit_target
    ).square().flatten(2).mean(dim=-1)
    unit_mask = batch.own_unit_mask.to(
        device=per_unit.device,
        dtype=per_unit.dtype,
    )
    per_row = (
        per_unit * unit_mask
    ).sum(dim=1) / unit_mask.sum(dim=1).clamp_min(1.0)
    active_rows = unit_mask.sum(dim=1).gt(0)
    unit_task = per_row[active_rows].mean()
    value = 0.5 * (terminal_money + terminal_margin)

    result = {
        "action": action,
        "farmer": farmer,
        "hands": hands,
        "market": market,
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
    short_economic = getattr(outputs.aux, "short_economic", None)
    if short_economic is not None and "short_economic" in aux_targets:
        economic = mse(short_economic, "short_economic")
        result["economic"] = economic
        result["total"] = (
            result["total"]
            + float(OPTIONAL_ECONOMIC_LOSS_WEIGHT) * economic
        )
    return result


def tensor_total_pretrain_loss_sequence(
    outputs,
    batch,
    targets: TensorActionTargets,
    *,
    step: torch.Tensor,
    batch_size: int,
    steps: int,
    weights: dict[str, float] | None = None,
    family_weights=None,
    market_active_op_weights=None,
):
    weights = dict(
        DEFAULT_LOSS_WEIGHTS if weights is None else weights
    )
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError(
            "pretrain loss weights must define frozen objective components"
        )
    batch_size = int(batch_size)
    steps = int(steps)
    if targets.batch_size != batch_size * steps:
        raise ValueError("sequence tensor target size mismatch")
    device = outputs.unit_op_logits.device
    targets = targets.to(device)
    step = step.to(device=device, dtype=torch.long)
    sample_weight = getattr(batch, "sample_weight", None)
    if sample_weight is None:
        sample_weight = torch.ones(
            targets.batch_size,
            device=device,
            dtype=outputs.unit_op_logits.dtype,
        )
    else:
        sample_weight = sample_weight.to(
            device=device,
            dtype=outputs.unit_op_logits.dtype,
        )
    sample_weight_bt = sample_weight.view(batch_size, steps)

    unit_loss = _unit_semantic_loss(
        outputs.unit_op_logits,
        outputs.unit_item_logits,
        outputs.unit_quantity_logits,
        targets,
        family_weights=family_weights,
    ).view(batch_size, steps, targets.max_units)
    unit_mask = targets.unit_mask.view(
        batch_size, steps, targets.max_units
    )
    farmer_values = unit_loss[:, :, 0]
    farmer_mask = unit_mask[:, :, 0]
    farmer_num = (
        farmer_values
        * sample_weight_bt
        * farmer_mask.to(farmer_values.dtype)
    ).sum(dim=0)
    farmer_den = (
        sample_weight_bt
        * farmer_mask.to(sample_weight_bt.dtype)
    ).sum(dim=0).clamp_min(torch.finfo(farmer_values.dtype).eps)
    farmer_t = farmer_num / farmer_den
    farmer = farmer_t.mean()

    if targets.max_units > 1:
        hand_mask = unit_mask[:, :, 1:]
        hand_count = hand_mask.sum(dim=-1)
        hand_row = (
            unit_loss[:, :, 1:]
            * hand_mask.to(unit_loss.dtype)
        ).sum(dim=-1) / hand_count.clamp_min(1).to(unit_loss.dtype)
        hand_active = hand_count.gt(0)
        hand_num = (
            hand_row
            * sample_weight_bt
            * hand_active.to(hand_row.dtype)
        ).sum(dim=0)
        hand_den = (
            sample_weight_bt
            * hand_active.to(sample_weight_bt.dtype)
        ).sum(dim=0).clamp_min(torch.finfo(hand_row.dtype).eps)
        hands_t = hand_num / hand_den
        hands = hands_t.mean()
        hand_domain_active_t = hand_active.any(dim=0)
    else:
        hands_t = torch.zeros_like(farmer_t)
        hands = farmer * 0.0
        hand_domain_active_t = torch.zeros(
            steps, device=device, dtype=torch.bool
        )

    market_decision = _market_semantic_loss(
        outputs,
        targets,
        step=step,
        family_weights=family_weights,
        market_active_op_weights=market_active_op_weights,
    )
    market_slots = market_decision.shape[1]
    market_decision = market_decision.view(
        batch_size, steps, market_slots
    )
    market_mask = targets.market_mask.view(
        batch_size, steps, market_slots
    )
    market_weight = sample_weight_bt.unsqueeze(-1)
    market_num = (
        market_decision
        * market_weight
        * market_mask.to(market_decision.dtype)
    ).sum(dim=(0, 2))
    market_den = (
        market_weight
        * market_mask.to(market_weight.dtype)
    ).sum(dim=(0, 2)).clamp_min(
        torch.finfo(market_decision.dtype).eps
    )
    market_t = market_num / market_den
    market = market_t.mean()

    domain_count_t = (
        2.0 + hand_domain_active_t.to(farmer_t.dtype)
    )
    action_t = (
        4.0 * farmer_t
        + 2.0 * market_t
        + hands_t * hand_domain_active_t.to(hands_t.dtype)
    ) / domain_count_t
    action = action_t.mean()

    aux_targets = getattr(batch, "auxiliary_targets", None)
    if not isinstance(aux_targets, dict):
        raise ValueError("batch.auxiliary_targets is required")

    def mse(prediction, name):
        prediction_fp32 = prediction.float()
        target = aux_targets[name].to(
            device=prediction.device,
            dtype=torch.float32,
        )
        return F.mse_loss(prediction_fp32, target)

    effect = mse(outputs.aux.effect, "effect")
    future_resource = mse(
        outputs.aux.future_resource, "future_resource"
    )
    opponent_effect = mse(
        outputs.aux.opponent_effect, "opponent_effect"
    )
    terminal_money = mse(
        outputs.aux.terminal_money, "terminal_money"
    )
    terminal_margin = mse(
        outputs.aux.terminal_margin, "terminal_margin"
    )
    unit_target = aux_targets["unit_task"].to(
        device=outputs.aux.unit_task.device,
        dtype=torch.float32,
    )
    per_unit = (
        outputs.aux.unit_task.float() - unit_target
    ).square().flatten(2).mean(dim=-1)
    own_unit_mask = batch.own_unit_mask.to(
        device=per_unit.device,
        dtype=per_unit.dtype,
    )
    per_row = (
        per_unit * own_unit_mask
    ).sum(dim=1) / own_unit_mask.sum(dim=1).clamp_min(1.0)
    unit_task = per_row.mean()
    value = 0.5 * (terminal_money + terminal_margin)

    result = {
        "action": action,
        "farmer": farmer,
        "hands": hands,
        "market": market,
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
    short_economic = getattr(outputs.aux, "short_economic", None)
    if short_economic is not None and "short_economic" in aux_targets:
        economic = mse(short_economic, "short_economic")
        result["economic"] = economic
        result["total"] = (
            result["total"]
            + float(OPTIONAL_ECONOMIC_LOSS_WEIGHT) * economic
        )
    return result
