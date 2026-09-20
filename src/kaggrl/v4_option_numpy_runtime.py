from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCHITECTURE_VERSION = "farmos_v4_step_strategy_options_v1"
FORMAT_VERSION = 1


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float32)
    positive = value >= 0
    out = np.empty_like(value)
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp = np.exp(value[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out.astype(np.float32)


def _softmax(value):
    value = np.asarray(value, dtype=np.float32)
    shifted = value - np.max(value)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def _linear(value, weight, bias):
    return (
        np.asarray(value, np.float32)
        @ np.asarray(weight, np.float32).T
        + np.asarray(bias, np.float32)
    ).astype(np.float32)


def _parameter_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(k for k in arrays if k.startswith("p__")):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class V4OptionNumpyState:
    h: np.ndarray
    c: np.ndarray


@dataclass
class V4OptionNumpyOutput:
    route_id: int
    market_mode: str
    confidence: float
    route_confidence: float
    route_gate_probability: float
    market_confidence: float
    phase_id: int
    phase_confidence: float
    predicted_step_norm: float
    predicted_remaining_norm: float
    value: float
    route_values: np.ndarray
    market_values: np.ndarray
    route_logits: np.ndarray
    market_logits: np.ndarray
    phase_logits: np.ndarray
    state: V4OptionNumpyState


class V4OptionNumpyPolicy:
    def __init__(self, arrays: dict[str, np.ndarray]):
        if int(arrays["format_version"]) != FORMAT_VERSION:
            raise ValueError("unsupported V4 option NumPy format")
        if str(arrays["architecture_version"]) != ARCHITECTURE_VERSION:
            raise ValueError("V4 option architecture mismatch")

        expected = str(arrays.get("parameter_sha256", ""))
        if not expected or expected != _parameter_sha256(arrays):
            raise ValueError("V4 option parameter hash mismatch")

        self.input_dim = int(arrays["input_dim"])
        self.hidden_dim = int(arrays["hidden_dim"])
        self.clock_dim = int(arrays["clock_dim"])
        self.route_ids = tuple(
            int(value) for value in np.asarray(arrays["route_ids"]).tolist()
        )
        self.route_gate_threshold = float(
            arrays.get("route_gate_threshold", 0.50)
        )
        self.objective_version = str(arrays.get("objective_version", ""))
        self.margin_scale = float(arrays.get("margin_scale", 10000.0))
        self.counterfactual_q_schema = str(
            arrays.get("counterfactual_q_schema", "")
        )
        self.counterfactual_q_steps = tuple(
            int(value)
            for value in np.asarray(
                arrays.get(
                    "counterfactual_q_steps",
                    np.asarray([], dtype=np.int32),
                )
            ).tolist()
        )
        self.market_modes = tuple(
            str(value)
            for value in json.loads(str(arrays["market_modes_json"]))
        )
        self.w = {
            key[3:].replace("__", "."): np.asarray(value, np.float32)
            for key, value in arrays.items()
            if key.startswith("p__")
        }
        self._validate_shapes()

    def _validate_shapes(self):
        hidden = self.hidden_dim
        required = {
            "input_proj.0.weight": (256, self.input_dim),
            "input_proj.0.bias": (256,),
            "input_proj.2.weight": (hidden, 256),
            "input_proj.2.bias": (hidden,),
            "clock_proj.weight": (hidden, self.clock_dim),
            "clock_proj.bias": (hidden,),
            "lstm.weight_ih_l0": (4 * hidden, hidden),
            "lstm.weight_hh_l0": (4 * hidden, hidden),
            "lstm.bias_ih_l0": (4 * hidden,),
            "lstm.bias_hh_l0": (4 * hidden,),
            "route_head.weight": (len(self.route_ids), hidden),
            "route_head.bias": (len(self.route_ids),),
            "route_clock_head.weight": (
                len(self.route_ids), self.clock_dim,
            ),
            "route_gate_head.weight": (1, hidden),
            "route_gate_head.bias": (1,),
            "route_gate_clock_head.weight": (1, self.clock_dim),
            "market_head.weight": (len(self.market_modes), hidden),
            "market_head.bias": (len(self.market_modes),),
            "market_clock_head.weight": (
                len(self.market_modes), self.clock_dim,
            ),
            "phase_head.weight": (5, hidden),
            "phase_head.bias": (5,),
            "phase_clock_head.weight": (5, self.clock_dim),
            "value_head.weight": (1, hidden),
            "value_head.bias": (1,),
            "value_clock_head.weight": (1, self.clock_dim),
        }
        q_required = {
            "route_value_head.weight": (len(self.route_ids), hidden),
            "route_value_head.bias": (len(self.route_ids),),
            "route_value_clock_head.weight": (
                len(self.route_ids), self.clock_dim,
            ),
            "market_value_head.weight": (len(self.market_modes), hidden),
            "market_value_head.bias": (len(self.market_modes),),
            "market_value_clock_head.weight": (
                len(self.market_modes), self.clock_dim,
            ),
        }
        if self.objective_version == "terminal_margin_advantage_weighted_bc_v1":
            required.update(q_required)
        for name, shape in required.items():
            value = self.w.get(name)
            if value is None or tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"V4 option parameter shape mismatch {name}: "
                    f"{None if value is None else value.shape} != {shape}"
                )

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(Path(path), allow_pickle=False)
        return cls({key: data[key] for key in data.files})

    def initial_state(self) -> V4OptionNumpyState:
        return V4OptionNumpyState(
            h=np.zeros(self.hidden_dim, dtype=np.float32),
            c=np.zeros(self.hidden_dim, dtype=np.float32),
        )

    def step(
        self,
        features,
        clock_context,
        state: V4OptionNumpyState | None = None,
    ) -> V4OptionNumpyOutput:
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"expected V4 feature shape {(self.input_dim,)}, got {x.shape}"
            )
        clock_context = np.asarray(clock_context, dtype=np.float32)
        if clock_context.shape != (self.clock_dim,):
            raise ValueError(
                f"expected clock context shape {(self.clock_dim,)}, "
                f"got {clock_context.shape}"
            )
        state = self.initial_state() if state is None else state

        z = np.tanh(_linear(
            x,
            self.w["input_proj.0.weight"],
            self.w["input_proj.0.bias"],
        )).astype(np.float32)
        z = np.tanh(_linear(
            z,
            self.w["input_proj.2.weight"],
            self.w["input_proj.2.bias"],
        )).astype(np.float32)
        clock_z = _linear(
            clock_context,
            self.w["clock_proj.weight"],
            self.w["clock_proj.bias"],
        )
        z = np.tanh(z + clock_z).astype(np.float32)

        gates = (
            _linear(
                z,
                self.w["lstm.weight_ih_l0"],
                self.w["lstm.bias_ih_l0"],
            )
            + _linear(
                state.h,
                self.w["lstm.weight_hh_l0"],
                self.w["lstm.bias_hh_l0"],
            )
        )
        i, f, g, o = np.split(gates, 4)
        i = _sigmoid(i)
        f = _sigmoid(f)
        g = np.tanh(g).astype(np.float32)
        o = _sigmoid(o)
        c = (f * state.c + i * g).astype(np.float32)
        h = (o * np.tanh(c)).astype(np.float32)
        next_state = V4OptionNumpyState(h=h, c=c)

        route_logits = (
            _linear(
                h,
                self.w["route_head.weight"],
                self.w["route_head.bias"],
            )
            + np.asarray(
                clock_context, np.float32
            ) @ self.w["route_clock_head.weight"].T
        ).astype(np.float32)
        route_gate_logit = float(
            _linear(
                h,
                self.w["route_gate_head.weight"],
                self.w["route_gate_head.bias"],
            )[0]
            + np.asarray(clock_context, np.float32)
            @ self.w["route_gate_clock_head.weight"][0]
        )
        route_gate_probability = float(_sigmoid(
            np.asarray([route_gate_logit], dtype=np.float32)
        )[0])
        market_logits = (
            _linear(
                h,
                self.w["market_head.weight"],
                self.w["market_head.bias"],
            )
            + np.asarray(
                clock_context, np.float32
            ) @ self.w["market_clock_head.weight"].T
        ).astype(np.float32)
        phase_logits = (
            _linear(
                h,
                self.w["phase_head.weight"],
                self.w["phase_head.bias"],
            )
            + np.asarray(
                clock_context, np.float32
            ) @ self.w["phase_clock_head.weight"].T
        ).astype(np.float32)
        # CLOCK_FEATURES ordering is schema-stable:
        # step_norm=0, remaining_steps_norm=5.
        clock = np.asarray(
            [clock_context[0], clock_context[5]],
            dtype=np.float32,
        )
        value = (
            _linear(
                h,
                self.w["value_head.weight"],
                self.w["value_head.bias"],
            )
            + np.asarray([
                np.asarray(clock_context, np.float32)
                @ self.w["value_clock_head.weight"][0]
            ], dtype=np.float32)
        ).astype(np.float32)

        if "route_value_head.weight" in self.w:
            route_values = (
                _linear(
                    h,
                    self.w["route_value_head.weight"],
                    self.w["route_value_head.bias"],
                )
                + np.asarray(clock_context, np.float32)
                @ self.w["route_value_clock_head.weight"].T
            ).astype(np.float32)
        else:
            route_values = np.zeros(
                len(self.route_ids), dtype=np.float32
            )
        if "market_value_head.weight" in self.w:
            market_values = (
                _linear(
                    h,
                    self.w["market_value_head.weight"],
                    self.w["market_value_head.bias"],
                )
                + np.asarray(clock_context, np.float32)
                @ self.w["market_value_clock_head.weight"].T
            ).astype(np.float32)
        else:
            market_values = np.zeros(
                len(self.market_modes), dtype=np.float32
            )

        route_prob = _softmax(route_logits)
        market_prob = _softmax(market_logits)
        phase_prob = _softmax(phase_logits)
        route_index = int(np.argmax(route_prob))
        market_index = int(np.argmax(market_prob))
        phase_index = int(np.argmax(phase_prob))
        route_confidence = float(route_prob[route_index])
        market_confidence = float(market_prob[market_index])

        return V4OptionNumpyOutput(
            route_id=self.route_ids[route_index],
            market_mode=self.market_modes[market_index],
            confidence=(
                market_confidence
                if route_gate_probability < 0.5
                else min(route_confidence, market_confidence)
            ),
            route_confidence=route_confidence,
            route_gate_probability=route_gate_probability,
            market_confidence=market_confidence,
            phase_id=phase_index,
            phase_confidence=float(phase_prob[phase_index]),
            predicted_step_norm=float(clock[0]),
            predicted_remaining_norm=float(clock[1]),
            value=float(value[0]),
            route_values=route_values,
            market_values=market_values,
            route_logits=route_logits,
            market_logits=market_logits,
            phase_logits=phase_logits,
            state=next_state,
        )
