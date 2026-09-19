from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .constants import ITEM_TO_ID, UNIT_OPS
from .v2_ledger import MARKET_OPS
from .v2_tensorize import (
    COMMODITY_FEATURES, ECONOMY_FEATURES, EFFECT_FEATURES, PREV_ACTION_GLOBAL_FEATURES,
    PREV_UNIT_ACTION_FEATURES, PREV_UNIT_EFFECT_FEATURES, SHOP_NAMES, TILE_FEATURES, UNIT_FEATURES,
)

FORMAT_VERSION = 1
EXCLUDED_TRAINING_PREFIXES = (
    "effect_head.", "future_resource_head.", "unit_task_head.", "opponent_effect_head.",
)


def feature_schema_payload() -> dict:
    return {
        "unit_ops": list(UNIT_OPS), "market_ops": list(MARKET_OPS),
        "item_to_id": dict(ITEM_TO_ID), "tile_features": list(TILE_FEATURES),
        "unit_features": list(UNIT_FEATURES), "prev_unit_action_features": list(PREV_UNIT_ACTION_FEATURES),
        "prev_unit_effect_features": list(PREV_UNIT_EFFECT_FEATURES),
        "commodity_features": list(COMMODITY_FEATURES),
        "economy_features": list(ECONOMY_FEATURES), "prev_action_global_features": list(PREV_ACTION_GLOBAL_FEATURES),
        "effect_features": list(EFFECT_FEATURES), "shop_names": list(SHOP_NAMES),
    }


def feature_schema_sha256() -> str:
    payload = json.dumps(feature_schema_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_parameter_sha256(model) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8")); h.update(b"\0")
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(str(arr.dtype).encode("ascii")); h.update(str(arr.shape).encode("ascii")); h.update(arr.tobytes())
    return h.hexdigest()


def exported_parameter_sha256(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(key for key in arrays if key.startswith("p__")):
        arr = np.asarray(arrays[name])
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(str(arr.dtype).encode("ascii")); h.update(str(arr.shape).encode("ascii")); h.update(arr.tobytes())
    return h.hexdigest()


def export_v2_numpy(model, path: str | Path, include_training_heads: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "feature_schema_sha256": np.asarray(feature_schema_sha256()),
        "model_parameter_sha256": np.asarray(model_parameter_sha256(model)),
        "parameter_count": np.asarray(sum(p.numel() for p in model.parameters()), dtype=np.int64),
    }
    for name, tensor in model.state_dict().items():
        if not include_training_heads and name.startswith(EXCLUDED_TRAINING_PREFIXES):
            continue
        arrays["p__" + name.replace(".", "__")] = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["exported_parameter_sha256"] = np.asarray(exported_parameter_sha256(arrays))
    np.savez_compressed(path, **arrays)
    return path
