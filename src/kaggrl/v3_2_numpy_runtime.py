from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .constants import ITEM_TO_ID
from .v2_numpy_runtime import (
    ID_TO_ITEM,
    MARKET_ITEM_OPS,
    MARKET_QUANTITY_OPS,
    _feature_schema_sha256,
    _linear,
    _parameter_arrays_sha256,
)
from .v3_numpy_runtime import TEMPORAL_CONFIG, V3NumpyPolicy
from .v3_2_schema import (
    ACTIVE_MARKET_OPS,
    ARCHITECTURE_VERSION,
    CONTINUE_ID,
    FORMAT_VERSION,
    STOP_ID,
    STRATEGY_CORE_SCALE,
    OPENING_ACTIVE_INTENT_SCALE,
)
ACTIVE_MARKET_OP_TO_ID = {
    op: i for i, op in enumerate(ACTIVE_MARKET_OPS)
}


def _runtime_schema_sha256(strategy_count: int) -> str:
    payload = {
        "feature_schema_sha256": _feature_schema_sha256(),
        "architecture_version": ARCHITECTURE_VERSION,
        "strategy_count": int(strategy_count),
        "strategy_core_scale": float(STRATEGY_CORE_SCALE),
        "opening_active_intent_scale": float(OPENING_ACTIVE_INTENT_SCALE),
        "temporal_config": TEMPORAL_CONFIG,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class V32NumpyPolicy(V3NumpyPolicy):
    FORMAT_VERSION = FORMAT_VERSION

    def __init__(self, arrays):
        if int(arrays["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported V3.2 numpy format version")
        if str(arrays["architecture_version"]) != ARCHITECTURE_VERSION:
            raise ValueError("V3.2 architecture version mismatch")

        strategy_count = int(arrays.get("strategy_count", 0))
        if strategy_count <= 0:
            raise ValueError("V3.2 runtime requires positive strategy_count")
        default_strategy_slot = int(
            arrays.get("default_strategy_slot", -1)
        )
        if not 0 <= default_strategy_slot < strategy_count:
            raise ValueError("invalid default strategy slot")

        if str(arrays["feature_schema_sha256"]) != _feature_schema_sha256():
            raise ValueError("V3.2 feature schema hash mismatch")
        if str(arrays["runtime_schema_sha256"]) != _runtime_schema_sha256(
            strategy_count
        ):
            raise ValueError("V3.2 runtime schema hash mismatch")
        temporal = json.loads(str(arrays["temporal_config_json"]))
        if temporal != TEMPORAL_CONFIG:
            raise ValueError("V3.2 temporal config mismatch")

        expected = str(arrays.get("exported_parameter_sha256", ""))
        if not expected or expected != _parameter_arrays_sha256(arrays):
            raise ValueError("V3.2 exported parameter hash mismatch")

        self.format_version = FORMAT_VERSION
        self.model_parameter_sha256 = str(
            arrays["model_parameter_sha256"]
        )
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
        self.has_opening_strategy_head = (
            opening_weight is not None or opening_bias is not None
        )
        if self.has_opening_strategy_head:
            if (
                opening_weight is None
                or opening_weight.shape != (len(ACTIVE_MARKET_OPS), 128)
                or opening_bias is None
                or opening_bias.shape != (len(ACTIVE_MARKET_OPS),)
            ):
                raise ValueError("opening strategy head shape mismatch")
        if float(arrays.get("strategy_core_scale", -1.0)) != float(
            STRATEGY_CORE_SCALE
        ):
            raise ValueError("strategy core scale mismatch")
        if float(arrays.get("opening_active_intent_scale", -1.0)) != float(
            OPENING_ACTIVE_INTENT_SCALE
        ):
            raise ValueError("opening active intent scale mismatch")
        self.relative_age = self._relative_age()

    def _condition_core_input(self, fused, strategy_slot: int | None):
        slot = self._resolve_strategy_slot(strategy_slot)
        embedding = self.w["strategy_embedding.weight"][slot]
        core_context = np.concatenate([embedding, embedding]).astype(
            np.float32, copy=False,
        )
        if core_context.shape != np.asarray(fused).shape:
            raise ValueError("strategy core context shape mismatch")
        return (
            np.asarray(fused, np.float32)
            + np.float32(STRATEGY_CORE_SCALE) * core_context
        ).astype(np.float32)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({key: data[key] for key in data.files})

    def _market_decision(
        self,
        slot,
        actor_ctx,
        hidden,
        global_h,
        intent,
        previous,
        ledger,
        teacher,
        rng,
        deterministic,
        strategy_context=None,
    ):
        legal = ledger.legal_market_mask(slot, {})
        hidden = self._decoder_step(
            actor_ctx,
            previous,
            hidden,
            global_h,
            intent,
            ledger,
            0.0,
            float(max(0, 9 - slot)) / 10.0,
        )
        continue_logits = _linear(
            hidden,
            self.w["market_continue_head.weight"],
            self.w["market_continue_head.bias"],
        )
        active_logits = _linear(
            hidden,
            self.w["market_active_op_head.weight"],
            self.w["market_active_op_head.bias"],
        )
        if slot == 0 and int(getattr(ledger, "step", 0)) == 0:
            width = len(ACTIVE_MARKET_OPS)
            if strategy_context is None:
                raise ValueError(
                    "strategy context is required for opening active residual"
                )
            context = np.asarray(strategy_context, np.float32).reshape(-1)
            if context.size < width:
                raise ValueError(
                    "strategy context is too narrow for opening active residual"
                )
            if self.has_opening_strategy_head:
                opening_residual = _linear(
                    context,
                    self.w["opening_strategy_head.weight"],
                    self.w["opening_strategy_head.bias"],
                )
            else:
                # Legacy format-v4 archives used the first seven strategy
                # dimensions directly. Keep them loadable for reproducibility.
                opening_residual = context[:width]
            active_logits = (
                np.asarray(active_logits, np.float32)
                + np.float32(OPENING_ACTIVE_INTENT_SCALE)
                * np.asarray(opening_residual, np.float32)
            ).astype(np.float32)
        item_logits = _linear(
            hidden,
            self.w["item_head.weight"],
            self.w["item_head.bias"],
        )

        active_mask = np.asarray(
            [bool(legal.ops.get(op, False)) for op in ACTIVE_MARKET_OPS],
            dtype=bool,
        )
        continue_mask = np.asarray(
            [True, bool(active_mask.any())], dtype=bool,
        )
        logp = 0.0
        quantity_count = 0

        if teacher is None:
            continue_id, lp = self._draw(
                continue_logits, continue_mask, rng, deterministic,
            )
            logp += lp
            self._capture_op_trace(
                f"market:{slot}:continue",
                ("STOP", "CONTINUE"),
                continue_logits,
                continue_mask,
                "STOP" if continue_id == STOP_ID else "CONTINUE",
                deterministic,
                teacher_forced=False,
            )
            if continue_id == STOP_ID:
                chosen = {
                    "kind": "STOP_QUEUE", "op": None,
                    "item": None, "quantity": None, "raw": [],
                }
                op = "STOP_QUEUE"
            else:
                active_id, lp = self._draw(
                    active_logits, active_mask, rng, deterministic,
                )
                logp += lp
                op = ACTIVE_MARKET_OPS[active_id]
                self._capture_op_trace(
                    f"market:{slot}:active",
                    ACTIVE_MARKET_OPS,
                    active_logits,
                    active_mask,
                    op,
                    deterministic,
                    teacher_forced=False,
                )
                if op == "NOP_SLOT":
                    chosen = {
                        "kind": "NOP_SLOT", "op": None,
                        "item": None, "quantity": None, "raw": [],
                    }
                else:
                    chosen = {
                        "kind": "ORDER", "op": op,
                        "item": None, "quantity": None, "raw": [op],
                    }
        else:
            chosen = dict(teacher)
            kind = str(chosen.get("kind", "ORDER"))
            op = (
                kind
                if kind in {"STOP_QUEUE", "NOP_SLOT"}
                else str(chosen.get("op", "NOP_SLOT"))
            )
            continue_target = (
                STOP_ID if op == "STOP_QUEUE" else CONTINUE_ID
            )
            logp += self._target_logp(
                continue_logits, continue_mask, continue_target,
            )
            self._capture_op_trace(
                f"market:{slot}:continue",
                ("STOP", "CONTINUE"),
                continue_logits,
                continue_mask,
                "STOP" if continue_target == STOP_ID else "CONTINUE",
                deterministic,
                teacher_forced=True,
            )
            if op != "STOP_QUEUE":
                if op not in ACTIVE_MARKET_OP_TO_ID:
                    raise ValueError(
                        f"unknown V3.2 active market op target: {op}"
                    )
                logp += self._target_logp(
                    active_logits,
                    active_mask,
                    ACTIVE_MARKET_OP_TO_ID[op],
                )
                self._capture_op_trace(
                    f"market:{slot}:active",
                    ACTIVE_MARKET_OPS,
                    active_logits,
                    active_mask,
                    op,
                    deterministic,
                    teacher_forced=True,
                )

        if op in MARKET_ITEM_OPS:
            item_mask = self._item_mask(legal.items.get(op, {}))
            if teacher is None:
                item_id, lp = self._draw(
                    item_logits, item_mask, rng, deterministic,
                )
                logp += lp
                chosen["item"] = ID_TO_ITEM[item_id]
            else:
                logp += self._target_logp(
                    item_logits,
                    item_mask,
                    ITEM_TO_ID[chosen["item"]],
                )

            max_value = self._quantity_max_from_legal(
                legal, "market", op, chosen.get("item"),
            )
            if op == "SELL" and (max_value is None or max_value < 1):
                raise RuntimeError("SELL action has no legal inventory bound")
            if teacher is None:
                tokens, quantity, lp, _ = self._quantity_sample(
                    hidden, True, rng, deterministic, max_value=max_value,
                )
                chosen["quantity"] = quantity
                logp += lp
            else:
                tokens, lp, _ = self._quantity_evaluate(
                    hidden, chosen.get("quantity"), True, max_value=max_value,
                )
                logp += lp
            quantity_count += len(tokens)

        chosen["raw"] = self._market_raw(
            op, chosen.get("item"), chosen.get("quantity"),
        )
        ledger.apply_market(chosen)
        return (
            chosen,
            hidden,
            self._semantic_embedding(chosen, "market"),
            float(logp),
            quantity_count,
        )
