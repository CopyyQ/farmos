from dataclasses import asdict

import torch

from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_tensorize import collate_transitions
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_tensor_decoder import teacher_step_tensor, teacher_step_tensor_mixed
from kaggrl.v3_tensor_ledger import TensorLedger
from kaggrl.v3_tensor_targets import TensorActionTargets


def _structured(hands: int, money: int = 3000, step: int = 123):
    own_tiles = [[None for _ in range(10)] for _ in range(10)]
    rival_tiles = [[None for _ in range(10)] for _ in range(10)]
    hand_pos = [[3 - i, 4] for i in range(hands)]
    obs = {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [
            {
                "money": money,
                "farmer": [4, 4],
                "hands": hand_pos,
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": own_tiles,
            },
            {
                "money": 2500,
                "farmer": [7, 7],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": rival_tiles,
            },
        ],
        "private": {
            "shed": {"WHEAT": 5},
            "seeds": {"WHEAT": 5},
            "inventories": [{} for _ in range(hands + 1)],
        },
        "market": {
            "inventory": {"WHEAT": 10000, "FERTILIZER": 10000},
            "prices": {"WHEAT": 10, "FERTILIZER": 20},
        },
        "town": {"unlocked_shops": []},
    }
    return asdict(normalize_observation(obs))


def _unit(op="PASS", item=None, quantity=None):
    raw = [op]
    if item is not None:
        raw.append(item)
    if quantity is not None:
        raw.append(quantity)
    return {
        "op": op,
        "item": item,
        "quantity": quantity,
        "raw": raw,
    }


def _order(op, item=None, quantity=None):
    raw = [op]
    if item is not None:
        raw.extend([item, quantity])
    return {
        "kind": "ORDER",
        "op": op,
        "item": item,
        "quantity": quantity,
        "raw": raw,
    }


def _stop():
    return {
        "kind": "STOP_QUEUE",
        "op": None,
        "item": None,
        "quantity": None,
        "raw": [],
    }


def _row(hands, farmer, hand_actions, market):
    return {
        "state": _structured(hands),
        "previous_effect": {},
        "canonical_action": {
            "farmer": farmer,
            "hands": hand_actions,
            "market": market,
        },
        "effects": {},
        "terminal_result": 0,
        "final_margin": 0,
    }


def test_tensor_teacher_decoder_matches_legacy_logits_for_batch():
    torch.manual_seed(37)
    rows = [
        _row(
            0,
            _unit("PASS"),
            [],
            [_order("BUY_SEED", "WHEAT", 1), _stop()],
        ),
        _row(
            1,
            _unit("EAST"),
            [_unit("PASS")],
            [_order("HIRE"), _stop()],
        ),
    ]
    batch = collate_transitions(rows)
    model = TemporalIntentPolicyV32(strategy_count=0).eval()

    with torch.no_grad():
        legacy = model.teacher_step(
            batch,
            batch.canonical_actions,
            state=None,
        )
        targets = TensorActionTargets.from_actions(
            batch.canonical_actions,
            max_units=batch.own_units.shape[1],
        )
        ledger = TensorLedger.from_states(batch.structured_states)
        tensor = teacher_step_tensor(
            model,
            batch,
            targets,
            ledger,
            state=None,
        )

    assert torch.allclose(
        tensor.fused_temporal,
        legacy.fused_temporal,
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.allclose(
        tensor.intent,
        legacy.intent,
        atol=1e-6,
        rtol=0.0,
    )

    for row_index, row in enumerate(legacy.rows):
        decisions = [row.farmer, *row.hands]
        for actor_index, decision in enumerate(decisions):
            assert torch.allclose(
                tensor.unit_op_logits[row_index, actor_index],
                decision.op_logits,
                atol=1e-6,
                rtol=0.0,
            )
            assert torch.allclose(
                tensor.unit_item_logits[row_index, actor_index],
                decision.item_logits,
                atol=1e-6,
                rtol=0.0,
            )
            assert torch.equal(
                tensor.unit_legal_op_mask[row_index, actor_index],
                decision.legal_op_mask,
            )
            if decision.quantity_logits is not None:
                length = decision.quantity_logits.shape[0]
                assert torch.allclose(
                    tensor.unit_quantity_logits[
                        row_index, actor_index, :length
                    ],
                    decision.quantity_logits,
                    atol=1e-6,
                    rtol=0.0,
                )

        for slot, decision in enumerate(row.market):
            assert torch.allclose(
                tensor.market_continue_logits[row_index, slot],
                decision.continue_logits,
                atol=1e-6,
                rtol=0.0,
            )
            assert torch.allclose(
                tensor.market_active_logits[row_index, slot],
                decision.op_logits,
                atol=1e-6,
                rtol=0.0,
            )
            assert torch.allclose(
                tensor.market_item_logits[row_index, slot],
                decision.item_logits,
                atol=1e-6,
                rtol=0.0,
            )
            assert torch.equal(
                tensor.market_continue_legal_mask[row_index, slot],
                decision.legal_continue_mask,
            )
            assert torch.equal(
                tensor.market_active_legal_mask[row_index, slot],
                decision.legal_op_mask,
            )
            if decision.quantity_logits is not None:
                length = decision.quantity_logits.shape[0]
                assert torch.allclose(
                    tensor.market_quantity_logits[
                        row_index, slot, :length
                    ],
                    decision.quantity_logits,
                    atol=1e-6,
                    rtol=0.0,
                )


def test_tensor_teacher_decoder_matches_opening_strategy_residual():
    torch.manual_seed(41)
    row = _row(
        0,
        _unit("PASS"),
        [],
        [_order("BUY_SEED", "WHEAT", 1), _stop()],
    )
    row["state"] = _structured(0, step=0)
    batch = collate_transitions([row])
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    slots = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        legacy = model.teacher_step(
            batch,
            batch.canonical_actions,
            state=None,
            strategy_slots=slots,
        )
        targets = TensorActionTargets.from_actions(
            batch.canonical_actions,
            max_units=batch.own_units.shape[1],
        )
        tensor = teacher_step_tensor(
            model,
            batch,
            targets,
            TensorLedger.from_states(batch.structured_states),
            state=None,
            strategy_slots=slots,
        )

    assert torch.allclose(
        tensor.market_active_logits[0, 0],
        legacy.rows[0].market[0].op_logits,
        atol=1e-6,
        rtol=0.0,
    )


def test_tensor_mixed_full_teacher_skips_sampling_path(monkeypatch):
    import kaggrl.v3_tensor_decoder as decoder_module

    row = _row(
        1,
        _unit("EAST"),
        [_unit("PICKUP", "WHEAT", 1)],
        [_order("BUY_SEED", "WHEAT", 1), _stop()],
    )
    batch = collate_transitions([row])
    model = TemporalIntentPolicyV32(strategy_count=0).eval()
    targets = TensorActionTargets.from_actions(
        batch.canonical_actions,
        max_units=batch.own_units.shape[1],
    )
    ledger = TensorLedger.from_states(batch.structured_states)

    def fail_sampling(*args, **kwargs):
        raise AssertionError("full teacher forcing must not sample quantity")

    monkeypatch.setattr(
        decoder_module,
        "_sample_quantity_argmax_tensor",
        fail_sampling,
    )
    with torch.no_grad():
        output = teacher_step_tensor_mixed(
            model,
            batch,
            targets,
            ledger,
            state=None,
            teacher_mix_probability=1.0,
        )
    assert output.unit_op_logits.shape[0] == 1


def test_tensor_mixed_zero_matches_legacy_model_conditioning():
    torch.manual_seed(67)
    rows = [
        _row(
            1,
            _unit("EAST"),
            [_unit("PASS")],
            [_order("BUY_SEED", "WHEAT", 1), _stop()],
        ),
        _row(
            0,
            _unit("PASS"),
            [],
            [_order("HIRE"), _stop()],
        ),
    ]
    batch = collate_transitions(rows)
    model = TemporalIntentPolicyV32(strategy_count=0).eval()
    targets = TensorActionTargets.from_actions(
        batch.canonical_actions,
        max_units=batch.own_units.shape[1],
    )
    ledger = TensorLedger.from_states(batch.structured_states)

    with torch.no_grad():
        legacy = model.teacher_step(
            batch,
            batch.canonical_actions,
            state=None,
            teacher_mix_probability=0.0,
            conditioning_rng=None,
        )
        tensor = teacher_step_tensor_mixed(
            model,
            batch,
            targets,
            ledger,
            state=None,
            teacher_mix_probability=0.0,
        )

    for row_index, row in enumerate(legacy.rows):
        decisions = [row.farmer, *row.hands]
        for actor_index, decision in enumerate(decisions):
            assert torch.allclose(
                tensor.unit_op_logits[row_index, actor_index],
                decision.op_logits,
                atol=1e-5,
                rtol=1e-5,
            )
            assert torch.allclose(
                tensor.unit_item_logits[row_index, actor_index],
                decision.item_logits,
                atol=1e-5,
                rtol=1e-5,
            )
        for slot, decision in enumerate(row.market):
            assert torch.allclose(
                tensor.market_continue_logits[row_index, slot],
                decision.continue_logits,
                atol=1e-5,
                rtol=1e-5,
            )
            assert torch.allclose(
                tensor.market_active_logits[row_index, slot],
                decision.op_logits,
                atol=1e-5,
                rtol=1e-5,
            )
            assert torch.allclose(
                tensor.market_item_logits[row_index, slot],
                decision.item_logits,
                atol=1e-5,
                rtol=1e-5,
            )
