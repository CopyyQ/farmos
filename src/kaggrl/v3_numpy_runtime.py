from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .v2_ledger import ShadowLedger
from .v2_numpy_runtime import (
    V2NumpyPolicy,
    _feature_schema_sha256,
    _layer_norm,
    _linear,
    _parameter_arrays_sha256,
    _sigmoid,
    _softmax,
    _tensorize_state,
)

ARCHITECTURE_VERSION = "rl_v3_temporal_attention"
STRATEGY_ARCHITECTURE_VERSION = "rl_v3_1_strategy_temporal_attention"
FORMAT_VERSION = 3
TEMPORAL_CONFIG = {
    "hidden_dim": 256,
    "attention_dim": 128,
    "heads": 4,
    "window": 32,
    "blocks": 1,
    "dropout": 0.0,
}


def _runtime_schema_sha256(
    architecture_version: str = ARCHITECTURE_VERSION,
    strategy_count: int = 0,
):
    payload = {
        "feature_schema_sha256": _feature_schema_sha256(),
        "architecture_version": str(architecture_version),
        "temporal_config": TEMPORAL_CONFIG,
    }
    if str(architecture_version) == STRATEGY_ARCHITECTURE_VERSION:
        payload["strategy_count"] = int(strategy_count)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RuntimeStateV3:
    h: np.ndarray
    c: np.ndarray
    memory: np.ndarray
    valid_length: int
    write_pos: int
    previous_action: dict[str, Any]


@dataclass
class RuntimeStepOutputV3:
    canonical_action: dict[str, Any]
    engine_action: dict[str, Any]
    logp: float
    terminal_money: float
    terminal_margin: float
    recurrent_state: RuntimeStateV3
    quantity_token_count: int
    fused_temporal: np.ndarray
    intent: np.ndarray
    attention_weights: np.ndarray
    attention_entropy: np.ndarray
    mean_attended_age: np.ndarray


class V3NumpyPolicy(V2NumpyPolicy):
    FORMAT_VERSION = FORMAT_VERSION

    def __init__(self, arrays):
        if int(arrays["format_version"]) != self.FORMAT_VERSION:
            raise ValueError("unsupported v3 numpy format version")
        architecture = str(arrays["architecture_version"])
        if architecture not in {ARCHITECTURE_VERSION, STRATEGY_ARCHITECTURE_VERSION}:
            raise ValueError("v3 architecture version mismatch")
        strategy_count = (
            int(arrays.get("strategy_count", 0))
            if architecture == STRATEGY_ARCHITECTURE_VERSION else 0
        )
        if architecture == STRATEGY_ARCHITECTURE_VERSION and strategy_count <= 0:
            raise ValueError("strategy architecture requires positive strategy_count")
        default_strategy_slot = (
            int(arrays.get("default_strategy_slot", -1)) if strategy_count else -1
        )
        if strategy_count and not 0 <= default_strategy_slot < strategy_count:
            raise ValueError("invalid default strategy slot")
        if str(arrays["feature_schema_sha256"]) != _feature_schema_sha256():
            raise ValueError("v3 feature schema hash mismatch")
        if str(arrays["runtime_schema_sha256"]) != _runtime_schema_sha256(
            architecture, strategy_count,
        ):
            raise ValueError("v3 runtime schema hash mismatch")
        temporal = json.loads(str(arrays["temporal_config_json"]))
        if temporal != TEMPORAL_CONFIG:
            raise ValueError("v3 temporal config mismatch")
        expected = str(arrays.get("exported_parameter_sha256", ""))
        if not expected or expected != _parameter_arrays_sha256(arrays):
            raise ValueError("v3 exported parameter hash mismatch")
        self.model_parameter_sha256 = str(arrays["model_parameter_sha256"])
        self.exported_parameter_sha256 = expected
        self.parameter_count = int(arrays["parameter_count"])
        self.architecture_version = architecture
        self.strategy_count = int(strategy_count)
        self.default_strategy_slot = int(default_strategy_slot)
        self.w = {
            key[3:].replace("__", "."): np.asarray(value, np.float32)
            for key, value in arrays.items() if key.startswith("p__")
        }
        if self.strategy_count:
            embedding = self.w.get("strategy_embedding.weight")
            if embedding is None or embedding.shape != (self.strategy_count, 128):
                raise ValueError("strategy embedding shape mismatch")
        self.relative_age = self._relative_age()

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({key: data[key] for key in data.files})

    @staticmethod
    def _relative_age():
        position = np.arange(32, dtype=np.float32)[:, None]
        div = np.exp(
            np.arange(0, 128, 2, dtype=np.float32)
            * np.float32(-math.log(10000.0) / 128.0)
        ).astype(np.float32)
        value = np.zeros((32, 128), dtype=np.float32)
        value[:, 0::2] = np.sin(position * div).astype(np.float32)
        value[:, 1::2] = np.cos(position * div).astype(np.float32)
        return value

    def initial_state(self):
        return RuntimeStateV3(
            h=np.zeros(256, np.float32),
            c=np.zeros(256, np.float32),
            memory=np.zeros((32, 128), np.float32),
            valid_length=0,
            write_pos=0,
            previous_action={},
        )

    @staticmethod
    def _linear_no_bias(x, weight):
        return np.asarray(x, np.float32) @ np.asarray(weight, np.float32).T

    def _resolve_strategy_slot(self, strategy_slot: int | None) -> int:
        if not self.strategy_count:
            if strategy_slot is not None:
                raise ValueError("base V3 runtime does not accept a strategy slot")
            return -1
        slot = self.default_strategy_slot if strategy_slot is None else int(strategy_slot)
        if not 0 <= slot < self.strategy_count:
            raise ValueError("strategy slot is out of range")
        return slot

    def _strategy_context(self, strategy_slot: int | None):
        slot = self._resolve_strategy_slot(strategy_slot)
        if slot < 0:
            return None
        return np.asarray(
            self.w["strategy_embedding.weight"][slot], np.float32,
        )

    def _condition_intent(self, intent, strategy_slot: int | None):
        context = self._strategy_context(strategy_slot)
        if context is None:
            return np.asarray(intent, np.float32)
        return (
            np.asarray(intent, np.float32) + context
        ).astype(np.float32)

    def _condition_core_input(self, fused, strategy_slot: int | None):
        del strategy_slot
        return np.asarray(fused, np.float32)

    def _temporal_step(self, fused, features, recurrent_state, strategy_slot=None):
        state = self.initial_state() if recurrent_state is None else recurrent_state
        fused = self._condition_core_input(fused, strategy_slot)
        gates = (
            _linear(fused, self.w["core.lstm.weight_ih"], self.w["core.lstm.bias_ih"])
            + _linear(state.h, self.w["core.lstm.weight_hh"], self.w["core.lstm.bias_hh"])
        )
        i, f, g, o = np.split(gates, 4)
        i = _sigmoid(i); f = _sigmoid(f); o = _sigmoid(o); g = np.tanh(g)
        c = (f * state.c + i * g).astype(np.float32)
        h = (o * np.tanh(c)).astype(np.float32)
        token_input = np.concatenate([
            h,
            features["previous_action_global"],
            features["previous_effect"],
            features["economy"],
        ]).astype(np.float32)
        token = _linear(
            token_input,
            self.w["core.token_proj.weight"],
            self.w["core.token_proj.bias"],
        )
        token = _layer_norm(
            token,
            self.w["core.token_norm.weight"],
            self.w["core.token_norm.bias"],
        ).astype(np.float32)
        memory = np.array(state.memory, dtype=np.float32, copy=True)
        memory[int(state.write_pos)] = token
        valid_length = min(32, int(state.valid_length) + 1)
        write_pos = (int(state.write_pos) + 1) % 32
        start = (write_pos - valid_length) % 32
        indices = [(start + offset) % 32 for offset in range(valid_length)]
        ordered = memory[np.asarray(indices, dtype=np.int64)]
        positions = np.arange(valid_length, dtype=np.int64)
        ages = (valid_length - 1 - positions).clip(0, 31)
        kv_input = (ordered + self.relative_age[ages]).astype(np.float32)

        q = self._linear_no_bias(token, self.w["core.q_proj.weight"])
        k = self._linear_no_bias(kv_input, self.w["core.k_proj.weight"])
        v = self._linear_no_bias(kv_input, self.w["core.v_proj.weight"])
        head_dim = 128 // 4
        q = q.reshape(4, head_dim)
        k = k.reshape(valid_length, 4, head_dim).transpose(1, 0, 2)
        v = v.reshape(valid_length, 4, head_dim).transpose(1, 0, 2)
        scores = np.einsum("hd,hwd->hw", q, k) / np.float32(math.sqrt(head_dim))
        active_weights = _softmax(scores, axis=-1).astype(np.float32)
        context = np.einsum("hw,hwd->hd", active_weights, v).reshape(128)
        context = self._linear_no_bias(context, self.w["core.o_proj.weight"])
        weights = np.zeros((4, 32), dtype=np.float32)
        weights[:, :valid_length] = active_weights
        entropy = -(
            active_weights * np.log(np.maximum(active_weights, np.float32(1e-12)))
        ).sum(axis=-1).astype(np.float32)
        mean_age = (
            active_weights * ages[None, :].astype(np.float32)
        ).sum(axis=-1).astype(np.float32)
        attention_hidden = _linear(
            context,
            self.w["core.attention_to_hidden.weight"],
            self.w["core.attention_to_hidden.bias"],
        )
        gate = _sigmoid(_linear(
            np.concatenate([h, attention_hidden]),
            self.w["core.fusion_gate.weight"],
            self.w["core.fusion_gate.bias"],
        ))
        fused_temporal = _layer_norm(
            h + gate * attention_hidden,
            self.w["core.fusion_norm.weight"],
            self.w["core.fusion_norm.bias"],
        ).astype(np.float32)
        intent = np.tanh(_linear(
            fused_temporal,
            self.w["core.intent.0.weight"],
            self.w["core.intent.0.bias"],
        )).astype(np.float32)
        intent = self._condition_intent(intent, strategy_slot)
        next_state = RuntimeStateV3(
            h=h.copy(), c=c.copy(), memory=memory,
            valid_length=valid_length, write_pos=write_pos,
            previous_action=dict(state.previous_action),
        )
        return (
            fused_temporal, intent, next_state,
            weights, entropy, mean_age,
        )

    def debug_encode(
        self, structured_state, previous_effect=None,
        previous_action=None, recurrent_state=None, strategy_slot=None,
    ):
        if previous_action is None and isinstance(recurrent_state, RuntimeStateV3):
            previous_action = recurrent_state.previous_action
        previous_action = {} if previous_action is None else previous_action
        effect_value = {} if previous_effect is None else previous_effect
        base = super().debug_encode(
            structured_state, effect_value, previous_action, None,
        )
        features = _tensorize_state(
            structured_state, effect_value, previous_action,
        )
        fused_temporal, intent, next_state, weights, entropy, mean_age = (
            self._temporal_step(
                base["fused"], features, recurrent_state, strategy_slot,
            )
        )
        base.update({
            "temporal_h": next_state.h,
            "c": next_state.c,
            "fused_temporal": fused_temporal,
            "intent": intent,
            "temporal_state": next_state,
            "attention_weights": weights,
            "attention_entropy": entropy,
            "mean_attended_age": mean_age,
        })
        return base

    def _decode_joint(
        self, structured_state, previous_effect, previous_action,
        recurrent_state, rng, deterministic, teacher_action=None,
        strategy_slot=None,
    ):
        debug = self.debug_encode(
            structured_state, previous_effect, previous_action, recurrent_state,
            strategy_slot=strategy_slot,
        )
        global_h = debug["fused_temporal"]
        intent = debug["intent"]
        strategy_context = self._strategy_context(strategy_slot)
        ledger = ShadowLedger.from_state(structured_state)
        hidden = np.tanh(_linear(
            np.concatenate([global_h, intent]),
            self.w["decoder_init.0.weight"],
            self.w["decoder_init.0.bias"],
        )).astype(np.float32)
        previous = self.w["start_action"].astype(np.float32).copy()
        own_ctx = debug["own_unit_ctx"]
        own_count = own_ctx.shape[0]
        if own_count < 1:
            raise ValueError("own unit set must include the main farmer")
        teacher_hands = list((teacher_action or {}).get("hands") or [])
        if teacher_action is not None and len(teacher_hands) != own_count - 1:
            raise ValueError("teacher hand count does not match state")
        logp = 0.0
        quantity_count = 0
        farmer_teacher = (
            (teacher_action or {}).get("farmer")
            if teacher_action is not None else None
        )
        farmer, hidden, previous, lp, count = self._unit_decision(
            "farmer", own_ctx[0], hidden, global_h, intent,
            previous, ledger,
            float(max(0, own_count - 1)) / float(own_count),
            farmer_teacher, rng, deterministic,
        )
        logp += lp
        quantity_count += count
        hands = []
        for hand_index in range(own_count - 1):
            teacher = (
                teacher_hands[hand_index]
                if teacher_action is not None else None
            )
            action, hidden, previous, lp, count = self._unit_decision(
                f"hand:{hand_index}", own_ctx[hand_index + 1],
                hidden, global_h, intent, previous, ledger,
                float(max(0, own_count - hand_index - 2)) / float(own_count),
                teacher, rng, deterministic,
            )
            hands.append(action)
            logp += lp
            quantity_count += count
        market_teacher = (
            list((teacher_action or {}).get("market") or [])
            if teacher_action is not None else None
        )
        if teacher_action is not None and not market_teacher:
            market_teacher = [{
                "kind": "STOP_QUEUE", "op": None,
                "item": None, "quantity": None, "raw": [],
            }]
        market = []
        limit = min(10, len(market_teacher)) if market_teacher is not None else 10
        for slot in range(limit):
            teacher = market_teacher[slot] if market_teacher is not None else None
            slot_ctx = self.w["market_slot_embedding.weight"][slot]
            action, hidden, previous, lp, count = self._market_decision(
                slot, slot_ctx, hidden, global_h, intent,
                previous, ledger, teacher, rng, deterministic,
                strategy_context=strategy_context,
            )
            market.append(action)
            logp += lp
            quantity_count += count
            if action.get("kind") == "STOP_QUEUE":
                break
        if not market:
            raise RuntimeError("market decoder emitted no slot")
        canonical = {"farmer": farmer, "hands": hands, "market": market}
        joint = np.concatenate([global_h, intent])
        terminal_money = float(_linear(
            joint,
            self.w["terminal_money_head.weight"],
            self.w["terminal_money_head.bias"],
        )[0])
        terminal_margin = float(_linear(
            joint,
            self.w["terminal_margin_head.weight"],
            self.w["terminal_margin_head.bias"],
        )[0])
        temporal_state = debug["temporal_state"]
        state = RuntimeStateV3(
            h=temporal_state.h.copy(),
            c=temporal_state.c.copy(),
            memory=temporal_state.memory.copy(),
            valid_length=int(temporal_state.valid_length),
            write_pos=int(temporal_state.write_pos),
            previous_action=canonical,
        )
        return RuntimeStepOutputV3(
            canonical_action=canonical,
            engine_action=self._engine_action(canonical),
            logp=float(logp),
            terminal_money=terminal_money,
            terminal_margin=terminal_margin,
            recurrent_state=state,
            quantity_token_count=int(quantity_count),
            fused_temporal=global_h.copy(),
            intent=intent.copy(),
            attention_weights=debug["attention_weights"].copy(),
            attention_entropy=debug["attention_entropy"].copy(),
            mean_attended_age=debug["mean_attended_age"].copy(),
        )

    def step(
        self, structured_state, previous_effect, previous_action,
        recurrent_state, rng, deterministic=False, strategy_slot=None,
    ):
        if previous_action is None and isinstance(recurrent_state, RuntimeStateV3):
            previous_action = recurrent_state.previous_action
        previous_action = {} if previous_action is None else previous_action
        rng = np.random.default_rng() if rng is None else rng
        effect_value = {} if previous_effect is None else previous_effect
        return self._decode_joint(
            structured_state, effect_value, previous_action,
            recurrent_state, rng, bool(deterministic), teacher_action=None,
            strategy_slot=strategy_slot,
        )

    def trace_step(
        self, structured_state, previous_effect, previous_action,
        recurrent_state, rng, deterministic=False, strategy_slot=None,
    ):
        self._trace_sink = []
        try:
            output = self.step(
                structured_state, previous_effect, previous_action,
                recurrent_state, rng, deterministic=deterministic,
                strategy_slot=strategy_slot,
            )
            return {
                "output": output,
                "decisions": list(self._trace_sink),
                "canonical_action": output.canonical_action,
                "engine_action": output.engine_action,
                "fused_temporal": output.fused_temporal.copy(),
                "intent": output.intent.copy(),
                "recurrent_state": output.recurrent_state,
            }
        finally:
            self._trace_sink = None

    def evaluate_action(
        self, structured_state, previous_effect, previous_action,
        recurrent_state, canonical_action, strategy_slot=None,
    ):
        if previous_action is None and isinstance(recurrent_state, RuntimeStateV3):
            previous_action = recurrent_state.previous_action
        previous_action = {} if previous_action is None else previous_action
        effect_value = {} if previous_effect is None else previous_effect
        return self._decode_joint(
            structured_state, effect_value, previous_action,
            recurrent_state, np.random.default_rng(0), True,
            teacher_action=canonical_action, strategy_slot=strategy_slot,
        )
