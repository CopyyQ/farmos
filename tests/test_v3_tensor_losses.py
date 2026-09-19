import torch

from kaggrl.v2_losses import total_pretrain_loss
from kaggrl.v2_tensorize import EFFECT_FEATURES
from kaggrl.v3_2_model import TemporalIntentPolicyV32
from kaggrl.v3_tensor_decoder import teacher_step_tensor
from kaggrl.v3_tensor_ledger import TensorLedger
from kaggrl.v3_tensor_losses import tensor_total_pretrain_loss
from kaggrl.v3_tensor_targets import TensorActionTargets

from tests.test_v3_tensor_decoder import _order, _row, _stop, _unit
from kaggrl.v2_tensorize import collate_transitions


def _attach_zero_auxiliary(batch):
    b = batch.own_grid.shape[0]
    u = batch.own_units.shape[1]
    batch.auxiliary_targets = {
        "effect": torch.zeros((b, len(EFFECT_FEATURES))),
        "future_resource": torch.zeros((b, 32)),
        "unit_task": torch.zeros((b, u, 16)),
        "opponent_effect": torch.zeros((b, 16)),
        "terminal_money": torch.zeros((b,)),
        "terminal_margin": torch.zeros((b,)),
    }
    batch.sample_weight = torch.ones((b,), dtype=torch.float32)


def _make_batch(step=123):
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
    for row in rows:
        row["state"]["step"] = step
        row["state"]["day"] = step // 24
        row["state"]["hour"] = step % 24
    batch = collate_transitions(rows)
    _attach_zero_auxiliary(batch)
    return batch


def test_tensor_total_pretrain_loss_matches_legacy_default_objective():
    torch.manual_seed(53)
    batch = _make_batch(step=123)
    model = TemporalIntentPolicyV32(strategy_count=0).eval()
    targets = TensorActionTargets.from_actions(
        batch.canonical_actions,
        max_units=batch.own_units.shape[1],
    )
    ledger = TensorLedger.from_states(batch.structured_states)
    with torch.no_grad():
        legacy_output = model.teacher_step(
            batch,
            batch.canonical_actions,
            state=None,
        )
        tensor_output = teacher_step_tensor(
            model,
            batch,
            targets,
            ledger,
            state=None,
        )
        legacy = total_pretrain_loss(
            legacy_output,
            batch,
        )
        tensor = tensor_total_pretrain_loss(
            tensor_output,
            batch,
            targets,
            step=ledger.step,
        )
    for key in (
        "action",
        "farmer",
        "hands",
        "market",
        "effect",
        "future_resource",
        "unit_task",
        "opponent_effect",
        "value",
        "total",
    ):
        assert torch.allclose(
            tensor[key],
            legacy[key],
            atol=1e-5,
            rtol=1e-5,
        ), (key, tensor[key], legacy[key])



def test_tensor_loss_ignores_nan_in_inactive_semantic_heads():
    torch.manual_seed(61)
    batch = _make_batch(step=123)
    model = TemporalIntentPolicyV32(strategy_count=0).eval()
    targets = TensorActionTargets.from_actions(
        batch.canonical_actions,
        max_units=batch.own_units.shape[1],
    )
    ledger = TensorLedger.from_states(batch.structured_states)
    with torch.no_grad():
        output = teacher_step_tensor(
            model,
            batch,
            targets,
            ledger,
            state=None,
        )

        # STOP_QUEUE never uses the active-op head.
        output.market_active_logits[:, 1, :] = float("nan")
        # HIRE/STOP_QUEUE do not use item or quantity heads.
        output.market_item_logits[1, 0, :] = float("nan")
        output.market_item_logits[:, 1, :] = float("nan")
        output.market_quantity_logits[1, 0, :, :] = float("nan")
        output.market_quantity_logits[:, 1, :, :] = float("nan")

        # PASS/EAST do not use unit item/quantity heads.
        output.unit_item_logits[:, 0, :] = float("nan")
        output.unit_quantity_logits[:, 0, :, :] = float("nan")

        losses = tensor_total_pretrain_loss(
            output,
            batch,
            targets,
            step=ledger.step,
        )

    assert torch.isfinite(losses["market"])
    assert torch.isfinite(losses["action"])
    assert torch.isfinite(losses["total"])


def test_tensor_total_pretrain_loss_matches_step0_market_weighting():
    torch.manual_seed(59)
    batch = _make_batch(step=0)
    model = TemporalIntentPolicyV32(strategy_count=2).eval()
    slots = torch.tensor([0, 1], dtype=torch.long)
    targets = TensorActionTargets.from_actions(
        batch.canonical_actions,
        max_units=batch.own_units.shape[1],
    )
    ledger = TensorLedger.from_states(batch.structured_states)
    family_weights = {
        "unit": {
            "WAIT": 1.0,
            "MOVEMENT": 1.5,
            "ACQUISITION": 2.0,
            "PRODUCTION": 1.2,
            "MAINTENANCE": 1.1,
            "HARVEST": 1.4,
            "DEPOSIT": 1.3,
        },
        "market": {
            "WAIT": 1.0,
            "ACQUISITION": 2.2,
            "SALE": 1.7,
            "HIRE": 2.5,
            "EXPANSION": 2.8,
        },
    }
    active_weights = {
        "NOP_SLOT": 1.0,
        "BUY_SEED": 3.0,
        "BUY_PRODUCT": 2.0,
        "BUY_ANIMAL": 4.0,
        "SELL": 2.5,
        "HIRE": 3.5,
        "BUY_LAND": 5.0,
    }
    with torch.no_grad():
        legacy_output = model.teacher_step(
            batch,
            batch.canonical_actions,
            state=None,
            strategy_slots=slots,
        )
        tensor_output = teacher_step_tensor(
            model,
            batch,
            targets,
            ledger,
            state=None,
            strategy_slots=slots,
        )
        legacy = total_pretrain_loss(
            legacy_output,
            batch,
            family_weights=family_weights,
            market_active_op_weights=active_weights,
        )
        tensor = tensor_total_pretrain_loss(
            tensor_output,
            batch,
            targets,
            step=ledger.step,
            family_weights=family_weights,
            market_active_op_weights=active_weights,
        )
    for key in ("farmer", "hands", "market", "action", "total"):
        assert torch.allclose(
            tensor[key],
            legacy[key],
            atol=1e-5,
            rtol=1e-5,
        ), (key, tensor[key], legacy[key])
