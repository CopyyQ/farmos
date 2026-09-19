from dataclasses import asdict

import numpy as np
import pytest
import torch

from kaggrl.v2_export import export_v2_numpy
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_numpy_runtime import V2NumpyPolicy
from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_tensorize import collate_transitions


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _stop():
    return {"kind": "STOP_QUEUE", "op": None, "item": None, "quantity": None, "raw": []}


def _structured(hands: int):
    own_tiles = [[None for _ in range(10)] for _ in range(10)]
    own_tiles[2][4] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                       "watered_today": True, "consecutive_unwatered": 0,
                       "yield_units": 3, "max_lifespan_step": 120, "fertilized_until_day": 5}
    hand_pos = [[i % 10, (i // 10) % 10] for i in range(hands)]
    obs = {
        "player": 0, "step": 101, "day": 4, "hour": 5,
        "farms": [
            {"money": 4500, "farmer": [4, 4], "hands": hand_pos, "hires_today": 3,
             "unlocked_quadrants": ["NW", "NE"], "tiles": own_tiles},
            {"money": 3900, "farmer": [8, 8], "hands": [[7, 8]], "hires_today": 1,
             "unlocked_quadrants": ["NW"],
             "tiles": [[None for _ in range(10)] for _ in range(10)]},
        ],
        "private": {
            "shed": {"WHEAT": 70, "MILK": 12}, "seeds": {"WHEAT": 8, "MELON": 2},
            "inventories": [{"FERTILIZER": 1}] + [{"WHEAT": i + 1} for i in range(hands)],
        },
        "market": {
            "inventory": {"WHEAT": 9990, "MILK": 10010},
            "prices": {"WHEAT": 27, "MILK": 155},
        },
        "town": {"unlocked_shops": ["BAKERY", "BAKERY", "PIZZA_SHOP"]},
    }
    return asdict(normalize_observation(obs))


def _row(hands: int):
    previous = {
        "farmer": _unit("EAST"),
        "hands": [_unit("PICKUP", "WHEAT", 1000) for _ in range(hands)],
        "market": [{"kind": "ORDER", "op": "HIRE", "item": None,
                    "quantity": None, "raw": ["HIRE"]}, _stop()],
    }
    current = {
        "farmer": _unit(), "hands": [_unit() for _ in range(hands)], "market": [_stop()],
    }
    return {
        "state": _structured(hands), "previous_action": previous,
        "previous_effect": {"money_delta": -20, "hand_count_delta": 1},
        "canonical_action": current, "effects": {}, "terminal_result": 0, "final_margin": 0,
    }


def test_numpy_export_matches_torch_encoder_core_for_dynamic_hands(tmp_path):
    torch.manual_seed(17)
    model = RecurrentIntentPolicy().eval()
    batch = collate_transitions([_row(17)])
    with torch.no_grad():
        encoded = model.encoder(batch)
        h, c, intent = model.core.step(encoded.fused, None)
    path = tmp_path / "v2_policy.npz"
    export_v2_numpy(model, path)
    runtime = V2NumpyPolicy.load(path)
    debug = runtime.debug_encode(
        batch.structured_states[0], batch.previous_effect[0].numpy(),
        batch.previous_actions[0], None,
    )
    assert np.allclose(debug["fused"], encoded.fused[0].numpy(), atol=3e-5, rtol=3e-5)
    assert np.allclose(debug["own_unit_ctx"], encoded.own_unit_ctx[0, :18].numpy(), atol=3e-5, rtol=3e-5)
    assert np.allclose(debug["h"], h[0].numpy(), atol=3e-5, rtol=3e-5)
    assert np.allclose(debug["c"], c[0].numpy(), atol=3e-5, rtol=3e-5)
    assert np.allclose(debug["intent"], intent[0].numpy(), atol=3e-5, rtol=3e-5)


def test_numpy_runtime_handles_zero_and_thirty_two_hands_without_fixed_cap(tmp_path):
    torch.manual_seed(19)
    model = RecurrentIntentPolicy().eval()
    path = tmp_path / "v2_policy.npz"
    export_v2_numpy(model, path)
    runtime = V2NumpyPolicy.load(path)
    for hands in (0, 32):
        batch = collate_transitions([_row(hands)])
        debug = runtime.debug_encode(
            batch.structured_states[0], batch.previous_effect[0].numpy(),
            batch.previous_actions[0], None,
        )
        assert debug["own_unit_ctx"].shape[0] == hands + 1
        assert np.isfinite(debug["fused"]).all()


def _torch_canonical(row_output):
    return {
        "farmer": row_output.farmer.chosen_action,
        "hands": [d.chosen_action for d in row_output.hands],
        "market": [d.chosen_action for d in row_output.market],
    }


def _strip_raw(action):
    def clean(command):
        return {k: v for k, v in command.items() if k not in {"raw", "_mask_fields"}}
    return {
        "farmer": clean(action["farmer"]),
        "hands": [clean(x) for x in action["hands"]],
        "market": [clean(x) for x in action["market"]],
    }


def _masked_logp(logits, mask, index):
    masked = logits.masked_fill(~mask, -1e9)
    return torch.log_softmax(masked, dim=-1)[int(index)]

from kaggrl.constants import ITEM_TO_ID, UNIT_OPS
from kaggrl.v2_ledger import MARKET_OPS
from kaggrl.v2_quantity import DIGIT_OFFSET, END_ID, OMIT_ID, VOCAB_SIZE


def _quantity_logp(decision, positive):
    if decision.quantity_logits is None:
        return decision.op_logits.new_zeros(())
    tokens = tuple(decision.quantity_tokens or ())
    total = decision.quantity_logits.new_zeros(())
    max_value = getattr(decision, "quantity_max_value", None)
    consumed_digits = []
    for step, token in enumerate(tokens):
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=decision.quantity_logits.device)
        if step == 0:
            if positive:
                mask[DIGIT_OFFSET + 1:DIGIT_OFFSET + 10] = True
            else:
                mask[OMIT_ID] = True
                mask[DIGIT_OFFSET:DIGIT_OFFSET + 10] = True
        else:
            mask[DIGIT_OFFSET:DIGIT_OFFSET + 10] = True
            mask[END_ID] = True
            if len(consumed_digits) >= 5:
                mask.zero_(); mask[END_ID] = True
            if len(consumed_digits) == 1 and consumed_digits[0] == 0:
                mask.zero_(); mask[END_ID] = True
        if max_value is not None:
            for token_id in range(VOCAB_SIZE):
                if not bool(mask[token_id]):
                    continue
                if token_id == OMIT_ID:
                    mask[token_id] = bool(not positive)
                    continue
                if token_id == END_ID:
                    if not consumed_digits:
                        mask[token_id] = False
                    else:
                        current = int("".join(str(value) for value in consumed_digits))
                        mask[token_id] = current <= int(max_value)
                    continue
                digit = token_id - DIGIT_OFFSET
                if not consumed_digits and digit == 0 and positive:
                    mask[token_id] = False
                    continue
                candidate = int("".join(
                    str(value) for value in [*consumed_digits, digit]
                ))
                mask[token_id] = candidate <= int(max_value)
        total = total + _masked_logp(decision.quantity_logits[step], mask, token)
        if token not in {OMIT_ID, END_ID}:
            consumed_digits.append(int(token) - DIGIT_OFFSET)
    return total


def _torch_row_logp(model, row):
    total = row.farmer.op_logits.new_zeros(())
    unit_decisions = [row.farmer, *row.hands]
    for decision in unit_decisions:
        action = decision.chosen_action
        op = str(action.get("op", "PASS"))
        total = total + _masked_logp(decision.op_logits, decision.legal_op_mask, model.unit_op_to_id[op])
        if op in {"PICKUP", "PLACE", "PLANT"}:
            total = total + _masked_logp(
                decision.item_logits, decision.legal_item_mask, ITEM_TO_ID[action["item"]]
            )
        if op in {"PICKUP", "PLACE"}:
            total = total + _quantity_logp(decision, positive=False)
    for decision in row.market:
        action = decision.chosen_action
        kind = str(action.get("kind", "ORDER"))
        op = kind if kind in {"STOP_QUEUE", "NOP_SLOT"} else str(action.get("op"))
        total = total + _masked_logp(decision.op_logits, decision.legal_op_mask, model.market_op_to_id[op])
        if op in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}:
            total = total + _masked_logp(
                decision.item_logits, decision.legal_item_mask, ITEM_TO_ID[action["item"]]
            )
            total = total + _quantity_logp(decision, positive=True)
    return total


def test_numpy_deterministic_joint_action_value_and_state_match_torch(tmp_path):
    torch.manual_seed(23)
    model = RecurrentIntentPolicy().eval()
    batch = collate_transitions([_row(8)])
    with torch.no_grad():
        torch_out = model.sample_step(batch, None, np.random.default_rng(7), deterministic=True)
    path = tmp_path / "v2_policy.npz"; export_v2_numpy(model, path)
    runtime = V2NumpyPolicy.load(path)
    got = runtime.step(
        batch.structured_states[0], {"money_delta": -20, "hand_count_delta": 1},
        batch.previous_actions[0], None, np.random.default_rng(7), deterministic=True,
    )
    assert _strip_raw(got.canonical_action) == _strip_raw(_torch_canonical(torch_out.rows[0]))
    assert np.allclose(got.recurrent_state.h, torch_out.recurrent_state[0][0].numpy(), atol=3e-5, rtol=3e-5)
    assert np.allclose(got.recurrent_state.c, torch_out.recurrent_state[1][0].numpy(), atol=3e-5, rtol=3e-5)
    assert abs(got.terminal_money - float(torch_out.aux.terminal_money[0])) < 3e-5
    assert abs(got.terminal_margin - float(torch_out.aux.terminal_margin[0])) < 3e-5


def test_numpy_evaluates_stochastic_torch_sample_logp_with_quantity_digits(tmp_path):
    torch.manual_seed(29)
    model = RecurrentIntentPolicy().eval()
    batch = collate_transitions([_row(17)])
    with torch.no_grad():
        torch_out = model.sample_step(batch, None, np.random.default_rng(991), deterministic=False)
        expected_logp = float(_torch_row_logp(model, torch_out.rows[0]))
    sampled = _torch_canonical(torch_out.rows[0])
    path = tmp_path / "v2_policy.npz"; export_v2_numpy(model, path)
    runtime = V2NumpyPolicy.load(path)
    got = runtime.evaluate_action(
        batch.structured_states[0], {"money_delta": -20, "hand_count_delta": 1},
        batch.previous_actions[0], None, sampled,
    )
    assert abs(got.logp - expected_logp) < 5e-4
    assert got.quantity_token_count == sum(
        len(d.quantity_tokens or ())
        for d in [torch_out.rows[0].farmer, *torch_out.rows[0].hands, *torch_out.rows[0].market]
    )


def test_numpy_runtime_archive_import_graph_is_torch_free():
    import inspect
    import kaggrl.v2_numpy_runtime as runtime_module
    source = inspect.getsource(runtime_module)
    assert "import torch" not in source
    assert "from torch" not in source


def test_numpy_loader_rejects_tampered_parameter_archive(tmp_path):
    torch.manual_seed(31)
    model = RecurrentIntentPolicy().eval()
    path = tmp_path / "v2_policy.npz"
    export_v2_numpy(model, path)
    data = np.load(path, allow_pickle=False)
    arrays = {key: data[key].copy() for key in data.files}
    weight_key = next(key for key in sorted(arrays) if key.startswith("p__"))
    flat = arrays[weight_key].reshape(-1)
    flat[0] = flat[0] + np.float32(0.125)
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="parameter hash"):
        V2NumpyPolicy.load(tampered)
