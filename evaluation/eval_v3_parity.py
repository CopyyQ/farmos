from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_model import TemporalIntentPolicy
from kaggrl.v3_numpy_runtime import RuntimeStateV3, V3NumpyPolicy
from kaggrl.v3_temporal import TemporalState


def _torch_state(raw: dict[str, Any] | None) -> TemporalState | None:
    if raw is None:
        return None
    return TemporalState(
        h=torch.tensor([raw["h"]], dtype=torch.float32),
        c=torch.tensor([raw["c"]], dtype=torch.float32),
        memory=torch.tensor([raw["memory"]], dtype=torch.float32),
        valid_length=torch.tensor([raw["valid_length"]], dtype=torch.long),
        write_pos=torch.tensor([raw["write_pos"]], dtype=torch.long),
    )


def _numpy_state(raw: dict[str, Any] | None) -> RuntimeStateV3 | None:
    if raw is None:
        return None
    return RuntimeStateV3(
        h=np.asarray(raw["h"], np.float32),
        c=np.asarray(raw["c"], np.float32),
        memory=np.asarray(raw["memory"], np.float32),
        valid_length=int(raw["valid_length"]),
        write_pos=int(raw["write_pos"]),
        previous_action=raw.get("previous_action") or {},
    )


def _batch_from_fixture(fixture: dict[str, Any]):
    row = {
        "state": fixture["structured_state"],
        "previous_action": fixture.get("previous_action") or {},
        "previous_effect": fixture.get("previous_effect") or {},
        "canonical_action": fixture["canonical_action"],
        "effects": {}, "terminal_result": 0, "final_margin": 0,
    }
    return collate_transitions([row])


def compare_v3_fixture(checkpoint, npz_path, fixture, atol: float = 5e-4):
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = TemporalIntentPolicy().eval()
    model.load_state_dict(payload["model_state"], strict=True)
    batch = _batch_from_fixture(fixture)
    with torch.no_grad():
        torch_trace = model.trace_sample_step(
            batch, _torch_state(fixture.get("recurrent_state_before")),
            np.random.default_rng(0), deterministic=True,
        )
    runtime = V3NumpyPolicy.load(npz_path)
    numpy_trace = runtime.trace_step(
        fixture["structured_state"], fixture.get("previous_effect") or {},
        fixture.get("previous_action") or {},
        _numpy_state(fixture.get("recurrent_state_before")),
        np.random.default_rng(0), deterministic=True,
    )
    left = torch_trace["decisions"]
    right = numpy_trace["decisions"]
    same_count = len(left) == len(right)
    mask_equal = same_count
    action_equal = same_count
    ops_equal = same_count
    max_raw = 0.0
    max_masked = 0.0
    for lrow, rrow in zip(left, right):
        ops_equal = ops_equal and lrow["ops"] == rrow["ops"]
        mask_equal = mask_equal and lrow["legal_mask"] == rrow["legal_mask"]
        action_equal = action_equal and lrow["chosen_op"] == rrow["chosen_op"]
        max_raw = max(
            max_raw,
            float(np.max(np.abs(
                np.asarray(lrow["raw_logits"], np.float64)
                - np.asarray(rrow["raw_logits"], np.float64)
            ))),
        )
        # Illegal entries may use different finite -inf sentinels in
        # Torch and NumPy. mask_equal already checks legality parity, so logit
        # parity is meaningful only on entries that are legal on both sides.
        legal = np.asarray(lrow["legal_mask"], dtype=bool)
        if legal.any():
            left_masked = np.asarray(lrow["masked_logits"], np.float64)[legal]
            right_masked = np.asarray(rrow["masked_logits"], np.float64)[legal]
            max_masked = max(
                max_masked,
                float(np.max(np.abs(left_masked - right_masked))),
            )
    fixture_action_equal = (
        numpy_trace["canonical_action"] == fixture["canonical_action"]
        and numpy_trace["engine_action"] == fixture["engine_action"]
    )
    passed = bool(
        same_count and ops_equal and mask_equal and action_equal
        and fixture_action_equal and max_raw <= atol and max_masked <= atol
    )
    return {
        "passed": passed,
        "decision_count_equal": bool(same_count),
        "ops_equal": bool(ops_equal),
        "mask_equal": bool(mask_equal),
        "action_equal": bool(action_equal),
        "fixture_action_equal": bool(fixture_action_equal),
        "max_raw_logit_abs_error": float(max_raw),
        "max_masked_logit_abs_error": float(max_masked),
        "atol": float(atol),
    }


def capture_feature_snapshot(
    structured_state: dict[str, Any],
    previous_action: dict[str, Any] | None = None,
    previous_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import hashlib
    from kaggrl.v2_tensorize import tensorize_state

    encoded = tensorize_state(
        structured_state, previous_effect or {}, previous_action or {},
    )
    digest = hashlib.sha256()
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    stats: dict[str, dict[str, float | bool]] = {}
    values: dict[str, list[Any]] = {}
    family_sha256: dict[str, str] = {}
    for name, value in vars(encoded).items():
        array = value.detach().cpu().contiguous().numpy()
        shapes[name] = list(array.shape)
        dtypes[name] = str(array.dtype)
        values[name] = array.tolist()
        family_sha256[name] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        finite = bool(np.isfinite(array).all())
        stats[name] = {
            "finite": finite,
            "min": float(array.min()) if array.size else 0.0,
            "max": float(array.max()) if array.size else 0.0,
            "mean": float(array.mean()) if array.size else 0.0,
        }
    return {
        "feature_sha256": digest.hexdigest(),
        "shapes": shapes,
        "dtypes": dtypes,
        "stats": stats,
        "values": values,
        "family_sha256": family_sha256,
    }
def compare_feature_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    names = sorted(set(reference.get("values", {})) | set(candidate.get("values", {})))
    mismatched = []
    max_abs_error: dict[str, float] = {}
    for name in names:
        if name not in reference.get("values", {}) or name not in candidate.get("values", {}):
            mismatched.append(name)
            continue
        if reference["shapes"].get(name) != candidate["shapes"].get(name):
            mismatched.append(name)
            continue
        left = np.asarray(reference["values"][name])
        right = np.asarray(candidate["values"][name])
        equal = np.array_equal(left, right) if left.dtype.kind in "biu" else np.allclose(
            left, right, atol=atol, rtol=rtol, equal_nan=False,
        )
        max_abs_error[name] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        if not bool(equal):
            mismatched.append(name)
    return {
        "passed": not mismatched,
        "mismatched_families": mismatched,
        "max_abs_error": max_abs_error,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def capture_observation_feature_snapshot(
    observation: dict[str, Any],
    previous_action: dict[str, Any] | None = None,
    previous_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from dataclasses import asdict
    from kaggrl.v2_observation import normalize_observation

    return capture_feature_snapshot(
        asdict(normalize_observation(observation)),
        previous_action or {},
        previous_effect or {},
    )
