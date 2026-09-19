from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .v2_export import (
    EXCLUDED_TRAINING_PREFIXES,
    exported_parameter_sha256,
    feature_schema_sha256,
    model_parameter_sha256,
)
from .v3_export import TEMPORAL_CONFIG
from .v3_2_schema import (
    ARCHITECTURE_VERSION, FORMAT_VERSION, STRATEGY_CORE_SCALE,
    OPENING_ACTIVE_INTENT_SCALE,
)


def runtime_schema_sha256(strategy_count: int) -> str:
    payload = {
        "feature_schema_sha256": feature_schema_sha256(),
        "architecture_version": ARCHITECTURE_VERSION,
        "strategy_count": int(strategy_count),
        "strategy_core_scale": float(STRATEGY_CORE_SCALE),
        "opening_active_intent_scale": float(OPENING_ACTIVE_INTENT_SCALE),
        "temporal_config": TEMPORAL_CONFIG,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_v3_2_numpy(
    model,
    path: str | Path,
    *,
    include_training_heads: bool = False,
    default_strategy_slot: int | None = None,
) -> Path:
    if getattr(model, "ARCHITECTURE_VERSION", None) != ARCHITECTURE_VERSION:
        raise ValueError("export_v3_2_numpy requires TemporalIntentPolicyV32")
    strategy_count = int(getattr(model, "strategy_count", 0) or 0)
    if strategy_count <= 0:
        raise ValueError("V3.2 export requires strategy conditioning")
    if default_strategy_slot is None:
        raise ValueError("V3.2 strategy export requires a default strategy slot")
    if not 0 <= int(default_strategy_slot) < strategy_count:
        raise ValueError("default strategy slot is out of range")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "architecture_version": np.asarray(ARCHITECTURE_VERSION),
        "feature_schema_sha256": np.asarray(feature_schema_sha256()),
        "runtime_schema_sha256": np.asarray(
            runtime_schema_sha256(strategy_count)
        ),
        "temporal_config_json": np.asarray(
            json.dumps(TEMPORAL_CONFIG, sort_keys=True)
        ),
        "model_parameter_sha256": np.asarray(
            model_parameter_sha256(model)
        ),
        "parameter_count": np.asarray(
            sum(parameter.numel() for parameter in model.parameters()),
            dtype=np.int64,
        ),
        "strategy_count": np.asarray(strategy_count, dtype=np.int32),
        "strategy_core_scale": np.asarray(
            STRATEGY_CORE_SCALE, dtype=np.float32,
        ),
        "opening_active_intent_scale": np.asarray(
            OPENING_ACTIVE_INTENT_SCALE, dtype=np.float32,
        ),
        "default_strategy_slot": np.asarray(
            int(default_strategy_slot), dtype=np.int32,
        ),
    }
    for name, tensor in model.state_dict().items():
        if (
            not include_training_heads
            and name.startswith(EXCLUDED_TRAINING_PREFIXES)
        ):
            continue
        arrays["p__" + name.replace(".", "__")] = (
            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    arrays["exported_parameter_sha256"] = np.asarray(
        exported_parameter_sha256(arrays)
    )
    np.savez_compressed(path, **arrays)
    return path
