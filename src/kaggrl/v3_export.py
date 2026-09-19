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

FORMAT_VERSION = 3
ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
STRATEGY_ARCHITECTURE_VERSION = "rl_v3_1_strategy_temporal_attention"
TEMPORAL_CONFIG = {
    "hidden_dim": 256,
    "attention_dim": 128,
    "heads": 4,
    "window": 32,
    "blocks": 1,
    "dropout": 0.0,
}


def runtime_schema_sha256(
    architecture_version: str = ARCHITECTURE_VERSION,
    strategy_count: int = 0,
) -> str:
    payload = {
        "feature_schema_sha256": feature_schema_sha256(),
        "architecture_version": str(architecture_version),
        "temporal_config": TEMPORAL_CONFIG,
    }
    if str(architecture_version) == STRATEGY_ARCHITECTURE_VERSION:
        payload["strategy_count"] = int(strategy_count)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def export_v3_numpy(
    model, path: str | Path, include_training_heads: bool = False,
    default_strategy_slot: int | None = None,
) -> Path:
    if getattr(model, "ARCHITECTURE_VERSION", None) != ARCHITECTURE_VERSION:
        raise ValueError("export_v3_numpy requires TemporalIntentPolicy")
    strategy_count = int(getattr(model, "strategy_count", 0) or 0)
    architecture_version = (
        STRATEGY_ARCHITECTURE_VERSION if strategy_count else ARCHITECTURE_VERSION
    )
    if strategy_count:
        if default_strategy_slot is None:
            raise ValueError("strategy export requires a default strategy slot")
        if not 0 <= int(default_strategy_slot) < strategy_count:
            raise ValueError("default strategy slot is out of range")
    elif default_strategy_slot is not None:
        raise ValueError("base V3 export does not accept a strategy slot")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "architecture_version": np.asarray(architecture_version),
        "feature_schema_sha256": np.asarray(feature_schema_sha256()),
        "runtime_schema_sha256": np.asarray(runtime_schema_sha256(
            architecture_version, strategy_count,
        )),
        "temporal_config_json": np.asarray(json.dumps(TEMPORAL_CONFIG, sort_keys=True)),
        "model_parameter_sha256": np.asarray(model_parameter_sha256(model)),
        "parameter_count": np.asarray(
            sum(parameter.numel() for parameter in model.parameters()), dtype=np.int64,
        ),
    }
    if strategy_count:
        arrays["strategy_count"] = np.asarray(strategy_count, dtype=np.int32)
        arrays["default_strategy_slot"] = np.asarray(
            int(default_strategy_slot), dtype=np.int32,
        )
    for name, tensor in model.state_dict().items():
        if not include_training_heads and name.startswith(EXCLUDED_TRAINING_PREFIXES):
            continue
        arrays["p__" + name.replace(".", "__")] = (
            tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        )
    arrays["exported_parameter_sha256"] = np.asarray(
        exported_parameter_sha256(arrays)
    )
    np.savez_compressed(path, **arrays)
    return path
