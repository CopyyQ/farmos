from __future__ import annotations

import numpy as np

from .actions import ActionCodec
from .constants import ITEM_CLASSES, MARKET_OPS, QTY_CLASSES, UNIT_OPS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _linear(x, w, b):
    return x @ w.T + b


def _softmax(x):
    y = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(y)
    return e / np.sum(e, axis=-1, keepdims=True)


class StructuredNumpyPolicy:
    def __init__(self, arrays):
        self.input_dim = int(arrays["input_dim"])
        self.hidden_dim = int(arrays["hidden_dim"])
        self.max_hands = int(arrays["max_hands"])
        self.max_market_orders = int(arrays["max_market_orders"])
        self.w = {k[3:].replace("__", "."): v for k, v in arrays.items() if k.startswith("p__")}
        self.codec = ActionCodec(self.max_hands, self.max_market_orders)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls({k: data[k] for k in data.files})

    def initial_state(self):
        z = np.zeros(self.hidden_dim, dtype=np.float32)
        return z.copy(), z.copy()
    def _trunk_step(self, obs, state):
        h, c = state
        e = np.tanh(_linear(obs, self.w["input_proj.weight"], self.w["input_proj.bias"]))
        gates = (
            _linear(e, self.w["lstm.weight_ih"], self.w["lstm.bias_ih"])
            + _linear(h, self.w["lstm.weight_hh"], self.w["lstm.bias_hh"])
        )
        i, f, g, o = np.split(gates, 4)
        i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
        g = np.tanh(g)
        c2 = f * c + i * g
        h2 = o * np.tanh(c2)
        return h2.astype(np.float32), c2.astype(np.float32)

    def _gru_step(self, x, h):
        gi = _linear(x, self.w["market_cell.weight_ih"], self.w["market_cell.bias_ih"])
        gh = _linear(h, self.w["market_cell.weight_hh"], self.w["market_cell.bias_hh"])
        i_r, i_z, i_n = np.split(gi, 3)
        h_r, h_z, h_n = np.split(gh, 3)
        r = _sigmoid(i_r + h_r)
        z = _sigmoid(i_z + h_z)
        n = np.tanh(i_n + r * h_n)
        return ((1.0 - z) * n + z * h).astype(np.float32)

    def _heads(self, h):
        return {
            "farmer_op": _linear(h, self.w["farmer_op.weight"], self.w["farmer_op.bias"]),
            "farmer_item": _linear(h, self.w["farmer_item.weight"], self.w["farmer_item.bias"]),
            "farmer_qty": _linear(h, self.w["farmer_qty.weight"], self.w["farmer_qty.bias"]),
            "hand_op": _linear(h, self.w["hand_op.weight"], self.w["hand_op.bias"]).reshape(self.max_hands, len(UNIT_OPS)),
            "hand_item": _linear(h, self.w["hand_item.weight"], self.w["hand_item.bias"]).reshape(self.max_hands, ITEM_CLASSES),
            "hand_qty": _linear(h, self.w["hand_qty.weight"], self.w["hand_qty.bias"]).reshape(self.max_hands, QTY_CLASSES),
        }
    def _market_greedy(self, h):
        prev = np.zeros(3, dtype=np.int64)
        dec_h = h.astype(np.float32, copy=True)
        triples = []
        for _ in range(self.max_market_orders):
            emb = np.concatenate((
                self.w["market_op_emb.weight"][prev[0]],
                self.w["market_item_emb.weight"][prev[1]],
                self.w["market_qty_emb.weight"][prev[2]],
            )).astype(np.float32)
            dec_h = self._gru_step(emb, dec_h)
            lo = _linear(dec_h, self.w["market_op.weight"], self.w["market_op.bias"])
            li = _linear(dec_h, self.w["market_item.weight"], self.w["market_item.bias"])
            lq = _linear(dec_h, self.w["market_qty.weight"], self.w["market_qty.bias"])
            prev = np.asarray([np.argmax(lo), np.argmax(li), np.argmax(lq)], dtype=np.int64)
            triples.append(prev.copy())
        return triples

    def greedy_step(self, observation, state):
        obs = np.asarray(observation, dtype=np.float32).reshape(self.input_dim)
        h, c = self._trunk_step(obs, state)
        out = self._heads(h)
        tokens = np.zeros(self.codec.width, dtype=np.int16)
        tokens[0:3] = [np.argmax(out["farmer_op"]), np.argmax(out["farmer_item"]), np.argmax(out["farmer_qty"])]
        for i in range(self.max_hands):
            base = 3 + 3 * i
            tokens[base:base + 3] = [
                np.argmax(out["hand_op"][i]),
                np.argmax(out["hand_item"][i]),
                np.argmax(out["hand_qty"][i]),
            ]
        start = (1 + self.max_hands) * 3
        for i, tri in enumerate(self._market_greedy(h)):
            tokens[start + 3 * i:start + 3 * i + 3] = tri
        value = float(_linear(h, self.w["value_head.weight"], self.w["value_head.bias"])[0])
        return tokens, value, (h, c)

    @staticmethod
    def _draw(logits, rng, deterministic=False, allowed=None):
        logits = np.asarray(logits, dtype=np.float64).copy()
        if allowed is not None:
            blocked = np.ones(logits.shape[0], dtype=bool)
            blocked[np.asarray(list(allowed), dtype=np.int64)] = False
            logits[blocked] = -1e30
        probs = _softmax(logits)
        if deterministic:
            choice = int(np.argmax(probs))
        else:
            choice = int(rng.choice(len(probs), p=probs))
        return choice, float(np.log(max(probs[choice], 1e-30)))

    def sample_step(self, observation, state, hand_count, rng, deterministic=False):
        obs = np.asarray(observation, dtype=np.float32).reshape(self.input_dim)
        h, c = self._trunk_step(obs, state)
        out = self._heads(h)
        tokens = np.zeros(self.codec.width, dtype=np.int16)
        mask = np.zeros(self.codec.width, dtype=np.uint8)
        logp = 0.0

        op, lp = self._draw(out["farmer_op"], rng, deterministic)
        tokens[0] = op; mask[0] = 1; logp += lp
        opname = UNIT_OPS[op]
        if opname in {"PICKUP", "PLACE", "PLANT"}:
            item, lp = self._draw(out["farmer_item"], rng, deterministic, range(1, ITEM_CLASSES))
            tokens[1] = item; mask[1] = 1; logp += lp
        if opname in {"PICKUP", "PLACE"}:
            qty, lp = self._draw(out["farmer_qty"], rng, deterministic, range(1, QTY_CLASSES))
            tokens[2] = qty; mask[2] = 1; logp += lp

        for i in range(min(int(hand_count), self.max_hands)):
            base = 3 + 3 * i
            op, lp = self._draw(out["hand_op"][i], rng, deterministic)
            tokens[base] = op; mask[base] = 1; logp += lp
            opname = UNIT_OPS[op]
            if opname in {"PICKUP", "PLACE", "PLANT"}:
                item, lp = self._draw(out["hand_item"][i], rng, deterministic, range(1, ITEM_CLASSES))
                tokens[base + 1] = item; mask[base + 1] = 1; logp += lp
            if opname in {"PICKUP", "PLACE"}:
                qty, lp = self._draw(out["hand_qty"][i], rng, deterministic, range(1, QTY_CLASSES))
                tokens[base + 2] = qty; mask[base + 2] = 1; logp += lp
        start = (1 + self.max_hands) * 3
        prev = np.zeros(3, dtype=np.int64)
        dec_h = h.astype(np.float32, copy=True)
        for slot in range(self.max_market_orders):
            emb = np.concatenate((
                self.w["market_op_emb.weight"][prev[0]],
                self.w["market_item_emb.weight"][prev[1]],
                self.w["market_qty_emb.weight"][prev[2]],
            )).astype(np.float32)
            dec_h = self._gru_step(emb, dec_h)
            lo = _linear(dec_h, self.w["market_op.weight"], self.w["market_op.bias"])
            li = _linear(dec_h, self.w["market_item.weight"], self.w["market_item.bias"])
            lq = _linear(dec_h, self.w["market_qty.weight"], self.w["market_qty.bias"])
            base = start + 3 * slot
            op, lp = self._draw(lo, rng, deterministic)
            tokens[base] = op; mask[base] = 1; logp += lp
            opname = MARKET_OPS[op]
            if opname == "NONE":
                break
            if opname in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
                item, lp = self._draw(li, rng, deterministic, range(1, ITEM_CLASSES))
                qty, lp2 = self._draw(lq, rng, deterministic, range(1, QTY_CLASSES))
                tokens[base + 1] = item; tokens[base + 2] = qty
                mask[base + 1] = 1; mask[base + 2] = 1
                logp += lp + lp2
            prev = tokens[base:base + 3].astype(np.int64)

        value = float(_linear(h, self.w["value_head.weight"], self.w["value_head.bias"])[0])
        return tokens, mask, float(logp), value, (h, c)