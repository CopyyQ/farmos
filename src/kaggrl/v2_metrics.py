from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

import torch

from .v2_quantity import decode_quantity

UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}
UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_ITEM_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_QUANTITY_OPS = set(MARKET_ITEM_OPS)
METRIC_SCHEMA_VERSION = 1


def _steps(value):
    return [value] if isinstance(value, dict) else list(value)


def _market_op(action: dict[str, Any]) -> str:
    kind = str(action.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return kind
    return str(action.get("op", "NOP_SLOT"))


def _unit_parts(action: dict[str, Any]):
    op = str(action.get("op", "PASS"))
    return op, action.get("item"), action.get("quantity")


def _market_parts(action: dict[str, Any]):
    op = _market_op(action)
    return op, action.get("item"), action.get("quantity")


def _mask_row(masks, key: str, index: int, count: int) -> list[bool]:
    if masks is None:
        return [True] * count
    value = masks.get(key, masks.get(key + "_mask")) if isinstance(masks, dict) else None
    if value is None:
        return [True] * count
    if torch.is_tensor(value):
        row = value if value.ndim == 1 else value[index]
        values = row.detach().cpu().tolist()
    else:
        row = value[index] if value and isinstance(value[0], (list, tuple)) else value
        values = list(row)
    if len(values) < count:
        raise ValueError(f"{key} mask shorter than action sequence")
    return [bool(x) for x in values[:count]]


def _semantic_unit(pred: dict[str, Any], target: dict[str, Any]):
    pop, pitem, pqty = _unit_parts(pred)
    top, titem, tqty = _unit_parts(target)
    op_ok = pop == top
    item_active = top in UNIT_ITEM_OPS
    qty_active = top in UNIT_QUANTITY_OPS
    item_ok = (pitem == titem) if item_active else True
    qty_ok = (pqty == tqty) if qty_active else True
    return op_ok and item_ok and qty_ok, op_ok, item_ok, qty_ok, item_active, qty_active


def _semantic_market(pred: dict[str, Any], target: dict[str, Any]):
    pop, pitem, pqty = _market_parts(pred)
    top, titem, tqty = _market_parts(target)
    op_ok = pop == top
    item_active = top in MARKET_ITEM_OPS
    qty_active = top in MARKET_QUANTITY_OPS
    item_ok = (pitem == titem) if item_active else True
    qty_ok = (pqty == tqty) if qty_active else True
    confusion = int({pop, top} == {"STOP_QUEUE", "NOP_SLOT"})
    return op_ok and item_ok and qty_ok, op_ok, item_ok, qty_ok, item_active, qty_active, confusion


def _mean(values: list[float], default: float = 1.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def semantic_domain_metrics(pred, target, masks=None) -> dict[str, float | int]:
    predictions, targets = _steps(pred), _steps(target)
    if len(predictions) != len(targets):
        raise ValueError("prediction/target step count mismatch")
    farmer_sem, farmer_op = [], []
    farmer_item, farmer_qty = [], []
    hand_sem_steps, hand_op_steps = [], []
    hand_item_steps, hand_qty_steps, all_hands = [], [], []
    market_sem_steps, market_op_steps = [], []
    market_item_steps, market_qty_steps, market_sequence = [], [], []
    market_continue, market_active_op, market_active_sem = [], [], []
    market_buy_recall, market_sell_recall, market_hire_recall = [], [], []
    market_active_op_recall = {
        op: [] for op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
    }
    market_active_target_count = 0
    market_buy_target_count = 0
    market_sell_target_count = 0
    full_joint, stop_nop = [], 0

    for step_index, (pstep, tstep) in enumerate(zip(predictions, targets)):
        f = _semantic_unit(pstep["farmer"], tstep["farmer"])
        farmer_sem.append(float(f[0])); farmer_op.append(float(f[1]))
        if f[4]: farmer_item.append(float(f[2]))
        if f[5]: farmer_qty.append(float(f[3]))

        phands, thands = list(pstep.get("hands") or []), list(tstep.get("hands") or [])
        count = max(len(phands), len(thands))
        hmask = _mask_row(masks, "hands", step_index, count)
        hsem, hop, hitem, hqty = [], [], [], []
        for i in range(count):
            if not hmask[i]: continue
            if i >= len(phands) or i >= len(thands):
                hsem.append(0.0); hop.append(0.0); continue
            result = _semantic_unit(phands[i], thands[i])
            hsem.append(float(result[0])); hop.append(float(result[1]))
            if result[4]: hitem.append(float(result[2]))
            if result[5]: hqty.append(float(result[3]))
        hand_sem_steps.append(_mean(hsem)); hand_op_steps.append(_mean(hop))
        if hitem: hand_item_steps.append(_mean(hitem))
        if hqty: hand_qty_steps.append(_mean(hqty))
        all_hands.append(float(all(value == 1.0 for value in hsem)))
        pmarket, tmarket = list(pstep.get("market") or []), list(tstep.get("market") or [])
        mcount = max(len(pmarket), len(tmarket))
        mmask = _mask_row(masks, "market", step_index, mcount)
        msem, mop, mitem, mqty = [], [], [], []
        for i in range(mcount):
            if not mmask[i]:
                continue
            pred_slot = pmarket[i] if i < len(pmarket) else None
            target_slot = tmarket[i] if i < len(tmarket) else None
            if target_slot is not None:
                target_op = _market_op(target_slot)
                target_continue = target_op != "STOP_QUEUE"
                if pred_slot is None:
                    market_continue.append(0.0)
                else:
                    pred_op = _market_op(pred_slot)
                    market_continue.append(float(
                        (pred_op != "STOP_QUEUE") == target_continue
                    ))
                if target_op not in {"STOP_QUEUE", "NOP_SLOT"}:
                    market_active_target_count += 1
                    if pred_slot is None:
                        active_op_ok = 0.0
                        active_sem_ok = 0.0
                    else:
                        active_result = _semantic_market(pred_slot, target_slot)
                        active_op_ok = float(active_result[1])
                        active_sem_ok = float(active_result[0])
                    market_active_op.append(active_op_ok)
                    market_active_sem.append(active_sem_ok)
                    if target_op in market_active_op_recall:
                        market_active_op_recall[target_op].append(active_op_ok)
                    if target_op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                        market_buy_target_count += 1
                        market_buy_recall.append(active_op_ok)
                    elif target_op == "SELL":
                        market_sell_target_count += 1
                        market_sell_recall.append(active_op_ok)
                    elif target_op == "HIRE":
                        market_hire_recall.append(active_op_ok)
            if pred_slot is None or target_slot is None:
                msem.append(0.0)
                mop.append(0.0)
                continue
            result = _semantic_market(pred_slot, target_slot)
            msem.append(float(result[0])); mop.append(float(result[1]))
            if result[4]:
                mitem.append(float(result[2]))
            if result[5]:
                mqty.append(float(result[3]))
            stop_nop += int(result[6])
        market_sem_steps.append(_mean(msem)); market_op_steps.append(_mean(mop))
        if mitem: market_item_steps.append(_mean(mitem))
        if mqty: market_qty_steps.append(_mean(mqty))
        mseq = float(all(value == 1.0 for value in msem))
        market_sequence.append(mseq)
        full_joint.append(float(f[0] and all_hands[-1] == 1.0 and mseq == 1.0))

    return {
        "farmer_semantic_exact": _mean(farmer_sem),
        "farmer_op_accuracy": _mean(farmer_op),
        "farmer_item_accuracy": _mean(farmer_item),
        "farmer_quantity_accuracy": _mean(farmer_qty),
        "mean_hand_semantic_exact": _mean(hand_sem_steps),
        "hands_op_accuracy": _mean(hand_op_steps),
        "hands_item_accuracy": _mean(hand_item_steps),
        "hands_quantity_accuracy": _mean(hand_qty_steps),
        "all_hands_exact": _mean(all_hands),
        "market_slot_semantic_exact": _mean(market_sem_steps),
        "market_op_accuracy": _mean(market_op_steps),
        "market_item_accuracy": _mean(market_item_steps),
        "market_quantity_accuracy": _mean(market_qty_steps),
        "market_sequence_exact": _mean(market_sequence),
        "market_continue_accuracy": _mean(market_continue),
        "market_active_op_accuracy": _mean(market_active_op),
        "market_active_semantic_exact": _mean(market_active_sem),
        "market_buy_op_recall": _mean(market_buy_recall),
        "market_sell_op_recall": _mean(market_sell_recall),
        "market_hire_op_recall": _mean(market_hire_recall),
        "market_buy_seed_op_recall": _mean(market_active_op_recall["BUY_SEED"], default=0.0),
        "market_buy_product_op_recall": _mean(market_active_op_recall["BUY_PRODUCT"], default=0.0),
        "market_buy_animal_op_recall": _mean(market_active_op_recall["BUY_ANIMAL"], default=0.0),
        "market_buy_land_op_recall": _mean(market_active_op_recall["BUY_LAND"], default=0.0),
        "market_buy_seed_target_count": len(market_active_op_recall["BUY_SEED"]),
        "market_buy_product_target_count": len(market_active_op_recall["BUY_PRODUCT"]),
        "market_buy_animal_target_count": len(market_active_op_recall["BUY_ANIMAL"]),
        "market_buy_land_target_count": len(market_active_op_recall["BUY_LAND"]),
        "market_active_target_count": int(market_active_target_count),
        "market_buy_target_count": int(market_buy_target_count),
        "market_sell_target_count": int(market_sell_target_count),
        "full_joint_step_exact": _mean(full_joint),
        "stop_nop_confusions": int(stop_nop),
    }


def _slice_masks(masks, index: int):
    if masks is None:
        return None
    out = {}
    for key, value in masks.items():
        if torch.is_tensor(value):
            out[key] = value if value.ndim == 1 else value[index:index + 1]
        elif value and isinstance(value[0], (list, tuple)):
            out[key] = [value[index]]
        else:
            out[key] = value
    return out


def joint_step_exact(pred, target, masks=None) -> torch.Tensor:
    predictions, targets = _steps(pred), _steps(target)
    if len(predictions) != len(targets):
        raise ValueError("prediction/target step count mismatch")
    values = []
    for index, (pstep, tstep) in enumerate(zip(predictions, targets)):
        metrics = semantic_domain_metrics([pstep], [tstep], _slice_masks(masks, index))
        values.append(metrics["full_joint_step_exact"] == 1.0)
    return torch.tensor(values, dtype=torch.bool)


def _token_tuple(value) -> tuple[int, ...]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return tuple(int(x) for x in value)


def _edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(
                current[-1] + 1, previous[j] + 1,
                previous[j - 1] + (0 if left == right else 1),
            ))
        previous = current
    return previous[-1]


def quantity_metrics(pred_digits, target_digits, active) -> dict[str, float]:
    pred_values = list(pred_digits); target_values = list(target_digits)
    flags = active.detach().cpu().tolist() if torch.is_tensor(active) else list(active)
    if not (len(pred_values) == len(target_values) == len(flags)):
        raise ValueError("quantity metric batch mismatch")
    exact = []
    distances = []
    matches = 0
    token_total = 0
    for pred, target, enabled in zip(pred_values, target_values, flags):
        if not bool(enabled):
            continue
        p, t = _token_tuple(pred), _token_tuple(target)
        width = max(len(p), len(t))
        matches += sum(1 for i in range(width) if i < len(p) and i < len(t) and p[i] == t[i])
        token_total += width
        distances.append(float(_edit_distance(p, t)))
        try:
            exact.append(float(decode_quantity(p) == decode_quantity(t)))
        except ValueError:
            exact.append(0.0)
    return {
        "quantity_integer_exact": _mean(exact),
        "digit_token_accuracy": float(matches / token_total) if token_total else 1.0,
        "mean_digit_edit_distance": _mean(distances, default=0.0),
    }


def effect_metrics(pred: dict[str, Any], target: dict[str, Any], masks=None) -> dict[str, float]:
    del masks
    result: dict[str, float] = {}
    for key in sorted(set(pred) & set(target)):
        pvalue, tvalue = pred[key], target[key]
        if torch.is_tensor(pvalue) or torch.is_tensor(tvalue):
            p = torch.as_tensor(pvalue, dtype=torch.float32)
            t = torch.as_tensor(tvalue, dtype=torch.float32, device=p.device)
            if p.shape != t.shape:
                raise ValueError(f"effect shape mismatch for {key}")
            result[f"{key}_mae"] = float((p - t).abs().mean().item())
        elif isinstance(pvalue, (int, float)) and isinstance(tvalue, (int, float)):
            result[f"{key}_mae"] = abs(float(pvalue) - float(tvalue))
    return result


def _effect_payload(transition: dict[str, Any]) -> dict[str, Any]:
    value = transition.get("effects")
    if value is None and transition.get("effects_json") is not None:
        value = json.loads(transition["effects_json"])
    return value if isinstance(value, dict) else {}


def economic_activity_summary(transitions: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    summary: dict[str, int | float] = {
        "confirmed_moves": 0,
        "confirmed_hires": 0,
        "confirmed_acquisitions": 0,
        "confirmed_plants": 0,
        "confirmed_services": 0,
        "confirmed_harvests": 0,
        "harvest_units_gained": 0,
        "confirmed_deposits": 0,
        "confirmed_sales": 0,
        "money_delta_total": 0,
    }
    moves = {"NORTH", "SOUTH", "EAST", "WEST"}
    services = {"WATER", "FEED", "CARE", "FERTILIZE", "COLLECT_FERTILIZER"}
    acquisitions = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "PICKUP"}
    deposits = {"PLACE", "DROP"}
    for transition in transitions:
        effects = _effect_payload(transition)
        summary["money_delta_total"] += int(effects.get("money_delta", 0) or 0)
        for evidence in list(effects.get("action_evidence") or []):
            if not isinstance(evidence, dict) or str(evidence.get("status")) != "confirmed":
                continue
            op = str(evidence.get("op", ""))
            if op in moves: summary["confirmed_moves"] += 1
            if op == "HIRE": summary["confirmed_hires"] += 1
            if op in acquisitions: summary["confirmed_acquisitions"] += 1
            if op == "PLANT": summary["confirmed_plants"] += 1
            if op in services: summary["confirmed_services"] += 1
            if op in deposits: summary["confirmed_deposits"] += 1
            if op == "SELL": summary["confirmed_sales"] += 1
            if op == "HARVEST":
                summary["confirmed_harvests"] += 1
                observed = evidence.get("observed") or {}
                delta = observed.get("inventory_delta") or {}
                summary["harvest_units_gained"] += sum(max(0, int(v or 0)) for v in delta.values())
    return summary


def metric_schema() -> dict[str, Any]:
    keys = sorted(semantic_domain_metrics(
        [{"farmer": {"op": "PASS"}, "hands": [], "market": []}],
        [{"farmer": {"op": "PASS"}, "hands": [], "market": []}],
    ))
    return {"version": METRIC_SCHEMA_VERSION, "semantic_metric_keys": keys,
            "quantity_metric_keys": ["quantity_integer_exact", "digit_token_accuracy",
                                     "mean_digit_edit_distance"]}
