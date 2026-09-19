from __future__ import annotations

import hashlib
import json

import numpy as np

from .v2_numpy_runtime import (
    _feature_schema_sha256,
    _linear,
    _parameter_arrays_sha256,
)
from .v3_numpy_runtime import TEMPORAL_CONFIG
from .v3_2_numpy_runtime import V32NumpyPolicy
from .v3_3_economy import economic_market_features_from_shadow_ledger
from .v3_3_schema import (
    ACTIVE_MARKET_OPS,
    ARCHITECTURE_VERSION,
    ECONOMIC_MARKET_DIM,
    ECONOMIC_MARKET_FEATURES,
    ECONOMIC_MARKET_SCALE,
    FORMAT_VERSION,
    OPENING_ACTIVE_INTENT_SCALE,
    SHORT_ECONOMIC_FEATURES,
    STRATEGY_CORE_SCALE,
)



def runtime_schema_sha256(strategy_count: int) -> str:
    payload = {
        "feature_schema_sha256": _feature_schema_sha256(),
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


class V33NumpyPolicy(V32NumpyPolicy):
    FORMAT_VERSION = FORMAT_VERSION

    def __init__(self, arrays):
        if int(arrays["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported V3.3 numpy format version")
        if str(arrays["architecture_version"]) != ARCHITECTURE_VERSION:
            raise ValueError("V3.3 architecture version mismatch")

        strategy_count = int(arrays.get("strategy_count", 0))
        if strategy_count <= 0:
            raise ValueError("V3.3 runtime requires positive strategy_count")
        default_strategy_slot = int(
            arrays.get("default_strategy_slot", -1)
        )
        if not 0 <= default_strategy_slot < strategy_count:
            raise ValueError("invalid default strategy slot")

        if str(arrays["feature_schema_sha256"]) != _feature_schema_sha256():
            raise ValueError("V3.3 feature schema hash mismatch")
        if str(arrays["runtime_schema_sha256"]) != runtime_schema_sha256(
            strategy_count
        ):
            raise ValueError("V3.3 runtime schema hash mismatch")
        temporal = json.loads(str(arrays["temporal_config_json"]))
        if temporal != TEMPORAL_CONFIG:
            raise ValueError("V3.3 temporal config mismatch")

        expected = str(arrays.get("exported_parameter_sha256", ""))
        if not expected or expected != _parameter_arrays_sha256(arrays):
            raise ValueError("V3.3 exported parameter hash mismatch")

        if json.loads(str(arrays["economic_market_features_json"])) != list(
            ECONOMIC_MARKET_FEATURES
        ):
            raise ValueError("V3.3 economic feature schema mismatch")
        if json.loads(str(arrays["short_economic_features_json"])) != list(
            SHORT_ECONOMIC_FEATURES
        ):
            raise ValueError("V3.3 short economic schema mismatch")

        self.format_version = FORMAT_VERSION
        self.model_parameter_sha256 = str(arrays["model_parameter_sha256"])
        self.exported_parameter_sha256 = expected
        self.parameter_count = int(arrays["parameter_count"])
        self.architecture_version = ARCHITECTURE_VERSION
        self.strategy_count = strategy_count
        self.default_strategy_slot = default_strategy_slot
        self.w = {
            key[3:].replace("__", "."): np.asarray(value, np.float32)
            for key, value in arrays.items()
            if key.startswith("p__")
        }

        embedding = self.w.get("strategy_embedding.weight")
        if (
            embedding is None
            or embedding.shape != (self.strategy_count, 128)
        ):
            raise ValueError("strategy embedding shape mismatch")
        continue_weight = self.w.get("market_continue_head.weight")
        active_weight = self.w.get("market_active_op_head.weight")
        if continue_weight is None or continue_weight.shape != (2, 192):
            raise ValueError("market continuation head shape mismatch")
        if (
            active_weight is None
            or active_weight.shape != (len(ACTIVE_MARKET_OPS), 192)
        ):
            raise ValueError("market active-op head shape mismatch")

        opening_weight = self.w.get("opening_strategy_head.weight")
        opening_bias = self.w.get("opening_strategy_head.bias")
        if (
            opening_weight is None
            or opening_weight.shape != (len(ACTIVE_MARKET_OPS), 128)
            or opening_bias is None
            or opening_bias.shape != (len(ACTIVE_MARKET_OPS),)
        ):
            raise ValueError("opening strategy head shape mismatch")
        self.has_opening_strategy_head = True

        economic_continue_weight = self.w.get(
            "economic_continue_head.weight"
        )
        economic_active_weight = self.w.get(
            "economic_active_head.weight"
        )
        if (
            economic_continue_weight is None
            or economic_continue_weight.shape != (2, ECONOMIC_MARKET_DIM)
        ):
            raise ValueError("economic continuation head shape mismatch")
        if (
            economic_active_weight is None
            or economic_active_weight.shape
            != (len(ACTIVE_MARKET_OPS), ECONOMIC_MARKET_DIM)
        ):
            raise ValueError("economic active-op head shape mismatch")

        if float(arrays.get("strategy_core_scale", -1.0)) != float(
            STRATEGY_CORE_SCALE
        ):
            raise ValueError("strategy core scale mismatch")
        if float(arrays.get("opening_active_intent_scale", -1.0)) != float(
            OPENING_ACTIVE_INTENT_SCALE
        ):
            raise ValueError("opening strategy scale mismatch")
        if float(arrays.get("economic_market_scale", -1.0)) != float(
            ECONOMIC_MARKET_SCALE
        ):
            raise ValueError("economic market scale mismatch")

        self.relative_age = self._relative_age()

    def _economic_market_residual(self, ledger):
        features = np.asarray(
            economic_market_features_from_shadow_ledger(ledger),
            dtype=np.float32,
        )
        continue_residual = _linear(
            features,
            self.w["economic_continue_head.weight"],
            self.w["economic_continue_head.bias"],
        )
        active_residual = _linear(
            features,
            self.w["economic_active_head.weight"],
            self.w["economic_active_head.bias"],
        )
        scale = np.float32(ECONOMIC_MARKET_SCALE)
        return (
            np.asarray(continue_residual, np.float32) * scale,
            np.asarray(active_residual, np.float32) * scale,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({key: data[key] for key in data.files})
