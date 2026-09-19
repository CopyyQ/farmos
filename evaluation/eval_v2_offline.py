from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from kaggrl.constants import ITEM_TO_ID, UNIT_OPS
from kaggrl.v2_metrics import semantic_domain_metrics
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_quantity import encode_quantity
from kaggrl.v2_training_data import (
    EpisodeSequence,
    SequenceChunk,
    V2EpisodeDataset,
    collate_v2_sequences,
)

EVALUATOR_VERSION = 1
VALID_MODES = {"teacher_forced", "free_running", "effect_only"}

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_dataset(checkpoint: Path, config: dict[str, Any]) -> Path:
    value = config.get("dataset_path")
    if not value:
        raise RuntimeError("checkpoint config has no dataset_path")
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    if path.is_file():
        return path.resolve()
    for parent in checkpoint.parents:
        candidate = parent / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _canonical(row_output) -> dict[str, Any]:
    return {
        "farmer": row_output.farmer.chosen_action,
        "hands": [decision.chosen_action for decision in row_output.hands],
        "market": [decision.chosen_action for decision in row_output.market],
    }

def _iter_episodes(dataset_path: Path, split: str, batch_size: int = 2048) -> Iterator[EpisodeSequence]:
    columns = [
        "episode_id", "seat", "step", "team_id", "team_name", "rank", "role", "split",
        "state_zlib", "canonical_action_json", "effects_json",
        "final_own_money", "final_margin", "terminal_result",
    ]
    parquet = pq.ParquetFile(dataset_path)
    current_key = None
    buffer: list[dict[str, Any]] = []
    completed: set[tuple[int, int]] = set()
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        for row in record_batch.to_pylist():
            if str(row.get("split")) != split or str(row.get("role")) != "active_best":
                continue
            key = (int(row["episode_id"]), int(row["seat"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                if current_key in completed:
                    raise RuntimeError("episode rows are not contiguous in dataset")
                completed.add(current_key)
                yield V2EpisodeDataset._prepare_episode(buffer)
                buffer, current_key = [], key
            buffer.append(row)
    if buffer:
        if current_key in completed:
            raise RuntimeError("episode rows are not contiguous in dataset")
        yield V2EpisodeDataset._prepare_episode(buffer)


def _one_step(row: dict[str, Any]) -> Any:
    chunk = SequenceChunk(int(row["episode_id"]), int(row["seat"]), (row,), False, False)
    return collate_v2_sequences([chunk], 1)

def _add_weighted(total: dict[str, float], weight: dict[str, float],
                  metrics: dict[str, float], rows: int) -> None:
    for key, value in metrics.items():
        if key == "stop_nop_confusions":
            total[key] = total.get(key, 0.0) + float(value)
            weight[key] = 1.0
        else:
            total[key] = total.get(key, 0.0) + float(value) * rows
            weight[key] = weight.get(key, 0.0) + rows


def _finish_weighted(total: dict[str, float], weight: dict[str, float]) -> dict[str, float]:
    result = {}
    for key, value in total.items():
        result[key] = value if key == "stop_nop_confusions" else value / max(weight[key], 1.0)
    if "stop_nop_confusions" in result:
        result["stop_nop_confusions"] = int(result["stop_nop_confusions"])
    return result


def _tensor_mae(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, int]:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise RuntimeError(f"offline effect shape mismatch: {prediction.shape} != {target.shape}")
    diff = (prediction - target).abs()
    return float(diff.sum().detach().cpu()), int(diff.numel())


def _masked_unit_mae(prediction, target, mask) -> tuple[float, int]:
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    active = mask.unsqueeze(-1).expand_as(prediction)
    diff = (prediction - target).abs()[active]
    return float(diff.sum().detach().cpu()), int(diff.numel())

def _accumulate_aux(aux, batch, sums: dict[str, float], counts: dict[str, int]) -> None:
    targets = batch.flat.auxiliary_targets
    pairs = {
        "effect": (aux.effect, targets["effect"]),
        "future_resource": (aux.future_resource, targets["future_resource"]),
        "opponent_effect": (aux.opponent_effect, targets["opponent_effect"]),
        "terminal_money": (aux.terminal_money, targets["terminal_money"]),
        "terminal_margin": (aux.terminal_margin, targets["terminal_margin"]),
    }
    for name, (prediction, target) in pairs.items():
        value, count = _tensor_mae(prediction, target)
        sums[name] += value; counts[name] += count
    value, count = _masked_unit_mae(
        aux.unit_task, targets["unit_task"], batch.flat.own_unit_mask,
    )
    sums["unit_task"] += value; counts["unit_task"] += count


def _effect_report(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {f"{name}_mae": sums[name] / max(counts[name], 1) for name in sums}


def _auxiliary_step(model, batch, state):
    encoded = model.encoder(batch.flat)
    h, c, intent = model.core.step(encoded.fused, state)
    aux = model._auxiliary(encoded, h, intent)
    return aux, (h, c)


def _finite_mapping(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_mapping(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_mapping(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True

def _evaluate_episode(model, episode: EpisodeSequence, mode: str, seed: int,
                      effect_sums, effect_counts):
    state = None
    generated_previous: dict[str, Any] = {}
    predictions, targets, records = [], [], []
    rng = np.random.default_rng(seed + episode.episode_id * 2 + episode.seat)
    for source_row in episode.rows:
        row = dict(source_row)
        if mode == "free_running":
            row["previous_action"] = generated_previous
        batch = _one_step(row)
        if mode == "effect_only":
            aux, state = _auxiliary_step(model, batch, state)
        else:
            output = model.sample_step(batch.flat, state, rng, deterministic=True)
            state = output.recurrent_state
            aux = output.aux
            predicted = _canonical(output.rows[0])
            target = batch.flat.canonical_actions[0]
            predictions.append(predicted); targets.append(target)
            records.append((source_row, predicted, target))
            if mode == "free_running":
                generated_previous = predicted
        _accumulate_aux(aux, batch, effect_sums, effect_counts)
        state = tuple(value.detach() for value in state)
    semantic = None if mode == "effect_only" else semantic_domain_metrics(
        predictions, targets, masks=None,
    )
    return len(episode.rows), semantic, records


def _load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError("invalid v2 checkpoint")
    model = RecurrentIntentPolicy()
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return payload, model

def evaluate_checkpoint(checkpoint, split: str, mode: str) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    if mode not in VALID_MODES:
        raise ValueError(f"unknown offline evaluation mode: {mode}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    payload, model = _load_model(checkpoint)
    dataset = _resolve_dataset(checkpoint, payload.get("config") or {})
    dataset_sha = _sha256(dataset)
    expected_sha = payload.get("dataset_sha256")
    if expected_sha and expected_sha != dataset_sha:
        raise RuntimeError("checkpoint dataset SHA does not match evaluation corpus")

    semantic_total: dict[str, float] = {}
    semantic_weight: dict[str, float] = {}
    effect_sums = defaultdict(float); effect_counts = defaultdict(int)
    slice_storage: dict[str, Any] = {}
    audit = _new_audit()
    rows = episodes = 0
    with torch.no_grad():
        for episode in _iter_episodes(dataset, split):
            count, semantic, records = _evaluate_episode(
                model, episode, mode, 20260917, effect_sums, effect_counts,
            )
            rows += count; episodes += 1
            if semantic is not None:
                _add_weighted(semantic_total, semantic_weight, semantic, count)
                for row, predicted, target in records:
                    step_metrics = semantic_domain_metrics(predicted, target, masks=None)
                    for category, label in _slice_labels(row, target).items():
                        _add_slice(slice_storage, category, label, step_metrics)
                    _update_record_audit(audit, row, predicted, target, step_metrics)
    if rows == 0:
        raise RuntimeError(f"no active_best rows found for split={split}")

    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "mode": mode, "split": split, "rows": rows, "episodes": episodes,
        "split_definition": {"split": split, "roles": ["active_best"]},
        "dataset_sha256": dataset_sha, "checkpoint_sha256": _sha256(checkpoint),
        "effect": _effect_report(effect_sums, effect_counts),
    }
    if mode != "effect_only":
        report["semantic"] = _finish_weighted(semantic_total, semantic_weight)
        report["slices"] = _finish_slices(slice_storage)
        report["audits"] = _finish_audit(audit)
    report["finite"] = _finite_mapping(report)
    return report

UNIT_QUANTITY_OPS = {"PICKUP", "PLACE"}
MARKET_QUANTITY_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}
MARKET_ITEM_OPS = set(MARKET_QUANTITY_OPS)
UNIT_ITEM_OPS = {"PICKUP", "PLACE", "PLANT"}


def _market_op(action: dict[str, Any]) -> str:
    kind = str(action.get("kind", "ORDER"))
    return kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(action.get("op", "NOP_SLOT"))

def _unit_schema_valid(command: Any) -> bool:
    if not isinstance(command, dict):
        return False
    op = str(command.get("op", ""))
    if op not in UNIT_OPS:
        return False
    if op in UNIT_ITEM_OPS and command.get("item") not in ITEM_TO_ID:
        return False
    if op in UNIT_QUANTITY_OPS:
        quantity = command.get("quantity")
        if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0):
            return False
    return True


def _market_slot_schema_valid(slot: Any) -> bool:
    if not isinstance(slot, dict):
        return False
    kind = str(slot.get("kind", "ORDER"))
    if kind in {"STOP_QUEUE", "NOP_SLOT"}:
        return slot.get("op") in {None, ""}
    if kind != "ORDER":
        return False
    op = str(slot.get("op", ""))
    if op not in MARKET_QUANTITY_OPS | {"HIRE", "BUY_LAND"}:
        return False
    if op in MARKET_ITEM_OPS and slot.get("item") not in ITEM_TO_ID:
        return False
    if op in MARKET_QUANTITY_OPS:
        quantity = slot.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return False
    return True

def _canonical_action_schema_valid(action: Any) -> bool:
    if not isinstance(action, dict) or not _unit_schema_valid(action.get("farmer")):
        return False
    hands = action.get("hands")
    market = action.get("market")
    if not isinstance(hands, list) or not all(_unit_schema_valid(item) for item in hands):
        return False
    if not isinstance(market, list) or not market or len(market) > 10:
        return False
    for index, slot in enumerate(market):
        if not _market_slot_schema_valid(slot):
            return False
        if _market_op(slot) == "STOP_QUEUE" and index != len(market) - 1:
            return False
    return True

def _signature(action: dict[str, Any] | None, domain: str):
    if action is None:
        return ("MISSING",)
    if domain == "unit":
        op = str(action.get("op", "PASS")); values = [op]
        if op in UNIT_ITEM_OPS: values.append(action.get("item"))
        if op in UNIT_QUANTITY_OPS: values.append(action.get("quantity"))
        return tuple(values)
    op = _market_op(action); values = [op]
    if op in MARKET_ITEM_OPS: values.append(action.get("item"))
    if op in MARKET_QUANTITY_OPS: values.append(action.get("quantity"))
    return tuple(values)


def _edit_distance(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (0 if a == b else 1)))
        previous = current
    return previous[-1]

def _rank_band(rank: int) -> str:
    rank = int(rank or 0)
    if rank <= 3: return "top3"
    if rank <= 6: return "4-6"
    if rank <= 10: return "7-10"
    return "other"


def _day_band(step: int) -> str:
    day = int(step) // 24
    for lo, hi in ((0, 5), (6, 11), (12, 17), (18, 23), (24, 29)):
        if lo <= day <= hi:
            return f"{lo}-{hi}"
    return "other"


def _hand_band(count: int) -> str:
    if count == 0: return "0"
    if count <= 4: return "1-4"
    if count <= 8: return "5-8"
    if count <= 16: return "9-16"
    return "17+"


def _market_count(action: dict[str, Any]) -> int:
    return sum(1 for slot in action.get("market") or [] if _market_op(slot) not in {"STOP_QUEUE", "NOP_SLOT"})


def _market_band(count: int) -> str:
    if count == 0: return "0"
    if count <= 2: return "1-2"
    if count <= 5: return "3-5"
    return "6-10"

def _slice_labels(row: dict[str, Any], target: dict[str, Any]) -> dict[str, str]:
    hands = len(target.get("hands") or [])
    market_orders = _market_count(target)
    return {
        "team": f"{int(row.get('team_id', 0))}:{row.get('team_name', '')}",
        "rank_band": _rank_band(int(row.get("rank", 0) or 0)),
        "day_band": _day_band(int(row.get("step", 0))),
        "hand_count_band": _hand_band(hands),
        "market_order_count_band": _market_band(market_orders),
    }


def _add_slice(storage, category: str, label: str, metrics: dict[str, float]):
    category_map = storage.setdefault(category, {})
    bucket = category_map.setdefault(label, {"rows": 0, "total": {}, "weight": {}})
    bucket["rows"] += 1
    _add_weighted(bucket["total"], bucket["weight"], metrics, 1)


def _finish_slices(storage) -> dict[str, Any]:
    result = {}
    for category, values in storage.items():
        result[category] = {}
        for label, bucket in sorted(values.items()):
            metrics = _finish_weighted(bucket["total"], bucket["weight"])
            result[category][label] = {"rows": bucket["rows"], **metrics}
    return result


def _quantity_pairs(predicted: dict[str, Any], target: dict[str, Any]):
    pairs = []
    units_p = [predicted.get("farmer") or {}, *(predicted.get("hands") or [])]
    units_t = [target.get("farmer") or {}, *(target.get("hands") or [])]
    for index, truth in enumerate(units_t):
        op = str(truth.get("op", "PASS"))
        if op in UNIT_QUANTITY_OPS:
            guess = units_p[index] if index < len(units_p) else {}
            pairs.append((guess.get("quantity"), truth.get("quantity")))
    for index, truth in enumerate(target.get("market") or []):
        op = _market_op(truth)
        if op in MARKET_QUANTITY_OPS:
            markets = predicted.get("market") or []
            guess = markets[index] if index < len(markets) else {}
            pairs.append((guess.get("quantity"), truth.get("quantity")))
    return pairs

def _new_audit() -> dict[str, Any]:
    return {
        "active_quantity_count": 0, "quantity_exact_count": 0,
        "digit_matches": 0, "digit_total": 0, "edit_distance_sum": 0.0,
        "quantity_abs_error_sum": 0.0,
        "market_position_correct": 0, "market_position_total": 0,
        "stop_nop_confusions": 0, "mismatches": [],
        "schema_valid": True,
        "queue_class_counts": {"STOP_QUEUE": 0, "NOP_SLOT": 0},
    }


def _update_quantity_audit(audit, predicted, target):
    for guess, truth in _quantity_pairs(predicted, target):
        p = encode_quantity(guess); t = encode_quantity(truth)
        audit["active_quantity_count"] += 1
        audit["quantity_exact_count"] += int(guess == truth)
        width = max(len(p), len(t))
        audit["digit_matches"] += sum(
            1 for i in range(width) if i < len(p) and i < len(t) and p[i] == t[i]
        )
        audit["digit_total"] += width
        audit["edit_distance_sum"] += _edit_distance(p, t)
        pnum = 0 if guess is None else int(guess)
        tnum = 0 if truth is None else int(truth)
        audit["quantity_abs_error_sum"] += abs(pnum - tnum)


def _update_market_audit(audit, predicted, target):
    pmarket, tmarket = list(predicted.get("market") or []), list(target.get("market") or [])
    for index in range(max(len(pmarket), len(tmarket))):
        guess = pmarket[index] if index < len(pmarket) else None
        truth = tmarket[index] if index < len(tmarket) else None
        audit["market_position_total"] += 1
        audit["market_position_correct"] += int(
            _signature(guess, "market") == _signature(truth, "market")
        )
        if guess is not None and truth is not None:
            if {_market_op(guess), _market_op(truth)} == {"STOP_QUEUE", "NOP_SLOT"}:
                audit["stop_nop_confusions"] += 1

def _quantity_impact(predicted, target) -> float:
    total = 0.0
    for guess, truth in _quantity_pairs(predicted, target):
        pnum = 0 if guess is None else int(guess)
        tnum = 0 if truth is None else int(truth)
        total += abs(pnum - tnum)
    return total


def _update_record_audit(audit, row, predicted, target, step_metrics):
    audit["schema_valid"] = bool(audit["schema_valid"] and _canonical_action_schema_valid(predicted))
    market = predicted.get("market") if isinstance(predicted, dict) else None
    if isinstance(market, list):
        for slot in market:
            if isinstance(slot, dict):
                op = _market_op(slot)
                if op in audit["queue_class_counts"]:
                    audit["queue_class_counts"][op] += 1
    _update_quantity_audit(audit, predicted, target)
    before_correct = audit["market_position_correct"]
    before_total = audit["market_position_total"]
    _update_market_audit(audit, predicted, target)
    if float(step_metrics["full_joint_step_exact"]) == 1.0:
        return
    market_errors = (audit["market_position_total"] - before_total) - (
        audit["market_position_correct"] - before_correct
    )
    impact = _quantity_impact(predicted, target) + float(market_errors)
    audit["mismatches"].append({
        "episode_id": int(row["episode_id"]), "seat": int(row["seat"]),
        "step": int(row["step"]), "team_id": int(row.get("team_id", 0)),
        "rank": int(row.get("rank", 0) or 0), "impact_proxy": impact,
        "predicted": predicted, "target": target,
    })


def _finish_audit(audit) -> dict[str, Any]:
    active = audit["active_quantity_count"]
    market_total = audit["market_position_total"]
    mismatches = sorted(
        audit["mismatches"], key=lambda row: (-row["impact_proxy"], row["episode_id"], row["step"])
    )[:50]
    return {
        "quantity": {
            "active_count": active,
            "integer_exact": audit["quantity_exact_count"] / active if active else 1.0,
            "digit_token_accuracy": audit["digit_matches"] / audit["digit_total"] if audit["digit_total"] else 1.0,
            "mean_digit_edit_distance": audit["edit_distance_sum"] / active if active else 0.0,
            "mean_absolute_quantity_error": audit["quantity_abs_error_sum"] / active if active else 0.0,
        },
        "market_order_position_accuracy": audit["market_position_correct"] / market_total if market_total else 1.0,
        "stop_nop_confusions": int(audit["stop_nop_confusions"]),
        "queue_class_counts": {
            "STOP_QUEUE": int(audit["queue_class_counts"]["STOP_QUEUE"]),
            "NOP_SLOT": int(audit["queue_class_counts"]["NOP_SLOT"]),
        },
        "schema_valid": bool(audit["schema_valid"]),
        "top_mismatches": mismatches,
    }

def write_offline_report(checkpoint, split: str, output_path, mode: str = "free_running"):
    report = evaluate_checkpoint(checkpoint, split, mode)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"offline report is immutable: {output}")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RL v2 checkpoint offline")
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--mode", default="free_running", choices=tuple(sorted(VALID_MODES)))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.output:
        report = write_offline_report(args.checkpoint, args.split, args.output, args.mode)
    else:
        report = evaluate_checkpoint(args.checkpoint, args.split, args.mode)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
