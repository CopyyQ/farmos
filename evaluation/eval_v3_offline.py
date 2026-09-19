from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggrl.v2_metrics import semantic_domain_metrics
from kaggrl.v2_training_data import (
    SequenceChunk,
    V2EpisodeDataset,
    collate_v2_sequences,
)
from kaggrl.v3_model import TemporalIntentPolicy
from training.train_v3_bc import (
    _add_attention,
    _attention_accumulator,
    _canonical,
    _finish_attention,
    _plain_histograms,
    _update_histograms,
)

ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
VALID_MODES = {"expert_history", "free_history", "effect_only"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_dataset(checkpoint: Path, payload: dict[str, Any]) -> Path:
    value = (payload.get("config") or {}).get("dataset_path")
    if not value:
        raise RuntimeError("checkpoint config has no dataset_path")
    path = Path(value)
    if path.is_file():
        return path.resolve()
    for parent in checkpoint.parents:
        candidate = parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(path)


def _load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("v3 offline checkpoint architecture mismatch")
    model = TemporalIntentPolicy().eval()
    model.load_state_dict(payload["model_state"], strict=True)
    return payload, model


def _one_step(row: dict[str, Any]):
    chunk = SequenceChunk(
        int(row["episode_id"]), int(row["seat"]), (row,), False, False,
    )
    return collate_v2_sequences([chunk], 1)


def _fraction(histogram: dict[str, int], op: str) -> float:
    total = sum(int(value) for value in histogram.values())
    return float(histogram.get(op, 0)) / max(total, 1)


def _op_fractions(candidate, expert):
    return {
        "farmer_pass": _fraction(candidate["farmer"], "PASS"),
        "hands_pass": _fraction(candidate["hands"], "PASS"),
        "market_stop_queue": _fraction(candidate["market"], "STOP_QUEUE"),
        "market_nop_slot": _fraction(candidate["market"], "NOP_SLOT"),
        "expert_farmer_pass": _fraction(expert["farmer"], "PASS"),
        "expert_hands_pass": _fraction(expert["hands"], "PASS"),
        "expert_market_stop_queue": _fraction(expert["market"], "STOP_QUEUE"),
        "expert_market_nop_slot": _fraction(expert["market"], "NOP_SLOT"),
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _accumulate_aux(aux, batch, sums, counts):
    targets = batch.flat.auxiliary_targets
    pairs = {
        "effect": (aux.effect, targets["effect"]),
        "future_resource": (aux.future_resource, targets["future_resource"]),
        "opponent_effect": (aux.opponent_effect, targets["opponent_effect"]),
        "terminal_money": (aux.terminal_money, targets["terminal_money"]),
        "terminal_margin": (aux.terminal_margin, targets["terminal_margin"]),
    }
    for name, (prediction, target) in pairs.items():
        target = target.to(prediction)
        diff = (prediction - target).abs()
        sums[name] += float(diff.sum().detach().cpu())
        counts[name] += int(diff.numel())
    target = targets["unit_task"].to(aux.unit_task)
    mask = batch.flat.own_unit_mask.to(aux.unit_task.device).unsqueeze(-1)
    active = mask.expand_as(aux.unit_task)
    diff = (aux.unit_task - target).abs()[active]
    sums["unit_task"] += float(diff.sum().detach().cpu())
    counts["unit_task"] += int(diff.numel())


def _effect_report(sums, counts):
    return {
        f"{name}_mae": float(sums[name]) / max(int(counts[name]), 1)
        for name in sums
    }


def _new_histograms():
    return {
        "farmer": defaultdict(int),
        "hands": defaultdict(int),
        "market": defaultdict(int),
    }


def _evaluate_mode(model, dataset, mode: str, seed: int):
    if mode not in VALID_MODES:
        raise ValueError(f"unknown v3 offline mode: {mode}")
    predictions, targets = [], []
    predicted_hist = _new_histograms()
    expert_hist = _new_histograms()
    attention = _attention_accumulator()
    effect_sums, effect_counts = defaultdict(float), defaultdict(int)
    rows = 0
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for episode in dataset:
            state = None
            generated_previous = {}
            for source_row in episode.rows:
                row = dict(source_row)
                if mode == "free_history":
                    row["previous_action"] = generated_previous
                batch = _one_step(row)
                if mode == "effect_only":
                    encoded = model.encoder(batch.flat)
                    fused, intent, state, _ = model.core.step(
                        encoded.fused,
                        batch.flat.previous_action_global,
                        batch.flat.previous_effect,
                        batch.flat.economy,
                        state,
                    )
                    aux = model._auxiliary(encoded, fused, intent)
                else:
                    output = model.sample_step(
                        batch.flat, state, rng, deterministic=True,
                    )
                    state = output.temporal_state
                    aux = output.aux
                    predicted = _canonical(output.rows[0])
                    target = batch.flat.canonical_actions[0]
                    predictions.append(predicted)
                    targets.append(target)
                    _update_histograms(predicted_hist, predicted)
                    _update_histograms(expert_hist, target)
                    _add_attention(attention, output)
                    if mode == "free_history":
                        generated_previous = predicted
                _accumulate_aux(aux, batch, effect_sums, effect_counts)
                state = model.core.detach_state(state)
                rows += 1
    result = {
        "mode": mode,
        "rows": rows,
        "episodes": len(dataset),
        "effect": _effect_report(effect_sums, effect_counts),
    }
    if mode != "effect_only":
        candidate = _plain_histograms(predicted_hist)
        expert = _plain_histograms(expert_hist)
        result.update({
            "semantic": semantic_domain_metrics(predictions, targets, masks=None),
            "op_histograms": candidate,
            "expert_op_histograms": expert,
            "op_fractions": _op_fractions(candidate, expert),
            "attention": _finish_attention(attention),
        })
    result["finite"] = _finite(result)
    return result


def _gaps(expert, free):
    left, right = expert["semantic"], free["semantic"]
    return {
        "farmer_op": float(left["farmer_op_accuracy"] - right["farmer_op_accuracy"]),
        "hands_op": float(left["hands_op_accuracy"] - right["hands_op_accuracy"]),
        "market_op": float(left["market_op_accuracy"] - right["market_op_accuracy"]),
    }


def evaluate_v3_offline(checkpoint, split: str = "val") -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    payload, model = _load_model(checkpoint)
    dataset_path = _resolve_dataset(checkpoint, payload)
    dataset_sha = _sha256(dataset_path)
    expected_sha = payload.get("dataset_sha256")
    if expected_sha and expected_sha != dataset_sha:
        raise RuntimeError("v3 offline dataset SHA mismatch")
    dataset = V2EpisodeDataset(dataset_path, split, {"active_best"})
    seed = int((payload.get("config") or {}).get("seed", 20260917))
    expert = _evaluate_mode(model, dataset, "expert_history", seed + 11)
    free = _evaluate_mode(model, dataset, "free_history", seed + 23)
    effect_only = _evaluate_mode(model, dataset, "effect_only", seed + 37)
    report = {
        "architecture_version": ARCHITECTURE_VERSION,
        "split": split,
        "split_definition": {"split": split, "roles": ["active_best"]},
        "dataset_sha256": dataset_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "expert_history": expert,
        "free_history": free,
        "effect_only": effect_only,
        "gaps": _gaps(expert, free),
    }
    report["finite"] = _finite(report)
    return report


def classify_primary_failure(offline: dict[str, Any], closed_loop: dict[str, Any]) -> str:
    if offline.get("runtime_ok", closed_loop.get("runtime_ok", True)) is False:
        return "runtime_or_parity"
    if offline.get("legality_ok", closed_loop.get("legality_ok", True)) is False:
        return "legality_over_mask"
    if bool(offline.get("action_logits_collapsed", closed_loop.get("action_logits_collapsed", False))):
        return "action_logit_collapse"
    if offline.get("temporal_memory_ok", closed_loop.get("temporal_memory_ok", True)) is False:
        return "temporal_memory_collapse"
    if offline.get("teacher_free_gap_ok", True) is False:
        return "teacher_free_distribution_shift"
    if closed_loop.get("economic_chain_ok", True) is False:
        return "insufficient_recovery_semantics"
    return "no_primary_failure"


def write_v3_offline_report(checkpoint, split: str, output_path):
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    report = evaluate_v3_offline(checkpoint, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
