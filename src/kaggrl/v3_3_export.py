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
from .v3_3_schema import (
    ARCHITECTURE_VERSION,
    ECONOMIC_MARKET_FEATURES,
    ECONOMIC_MARKET_SCALE,
    FORMAT_VERSION,
    OPENING_ACTIVE_INTENT_SCALE,
    SHORT_ECONOMIC_FEATURES,
    STRATEGY_CORE_SCALE,
)

V33_EXCLUDED_TRAINING_PREFIXES = (
    *EXCLUDED_TRAINING_PREFIXES,
    "short_economic_head.",
)


def runtime_schema_sha256(strategy_count: int) -> str:
    payload = {
        "feature_schema_sha256": feature_schema_sha256(),
        "architecture_version": ARCHITECTURE_VERSION,
        "strategy_count": int(strategy_count),
        "strategy_core_scale": float(STRATEGY_CORE_SCALE),
        "opening_active_intent_scale": float(OPENING_ACTIVE_INTENT_SCALE),
        "economic_market_scale": float(ECONOMIC_MARKET_SCALE),
        "economic_market_features": list(ECONOMIC_MARKET_FEATURES),
        "short_economic_features": list(SHORT_ECONOMIC_FEATURES),
        "temporal_config": TEMPORAL_CONFIG,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_v3_3_numpy(
    model,
    path: str | Path,
    *,
    include_training_heads: bool = False,
    default_strategy_slot: int | None = None,
) -> Path:
    if getattr(model, "ARCHITECTURE_VERSION", None) != ARCHITECTURE_VERSION:
        raise ValueError("export_v3_3_numpy requires TemporalIntentPolicyV33")
    strategy_count = int(getattr(model, "strategy_count", 0) or 0)
    if strategy_count <= 0:
        raise ValueError("V3.3 export requires strategy conditioning")
    if default_strategy_slot is None:
        raise ValueError("V3.3 strategy export requires a default strategy slot")
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
        "economic_market_scale": np.asarray(
            ECONOMIC_MARKET_SCALE, dtype=np.float32,
        ),
        "economic_market_features_json": np.asarray(
            json.dumps(list(ECONOMIC_MARKET_FEATURES))
        ),
        "short_economic_features_json": np.asarray(
            json.dumps(list(SHORT_ECONOMIC_FEATURES))
        ),
        "default_strategy_slot": np.asarray(
            int(default_strategy_slot), dtype=np.int32,
        ),
    }
    for name, tensor in model.state_dict().items():
        if (
            not include_training_heads
            and name.startswith(V33_EXCLUDED_TRAINING_PREFIXES)
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
