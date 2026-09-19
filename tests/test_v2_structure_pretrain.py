import random
from dataclasses import asdict

import torch

from kaggrl.v2_observation import normalize_observation
from kaggrl.v2_structure_pretrain import mask_joint_action, structure_reconstruction_loss
from kaggrl.v2_model import RecurrentIntentPolicy
from kaggrl.v2_tensorize import collate_transitions


def _unit(op="PASS", item=None, quantity=None):
    return {"op": op, "item": item, "quantity": quantity, "raw": [op]}


def _order(op, item=None, quantity=None):
    if op in {"STOP_QUEUE", "NOP_SLOT"}:
        return {"kind": op, "op": None, "item": None, "quantity": None, "raw": []}
    raw = [op] if item is None else [op, item, quantity]
    return {"kind": "ORDER", "op": op, "item": item, "quantity": quantity, "raw": raw}


def _row():
    tiles = [[None for _ in range(10)] for _ in range(10)]
    obs = {
        "player": 0, "step": 8, "day": 0, "hour": 8,
        "farms": [
            {"money": 3000, "farmer": [4, 4], "hands": [[4, 4]],
             "hires_today": 0, "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"money": 3000, "farmer": [7, 7], "hands": [],
             "hires_today": 0, "unlocked_quadrants": ["NW"],
             "tiles": [[None for _ in range(10)] for _ in range(10)]},
        ],
        "private": {
            "shed": {"WHEAT": 20}, "seeds": {"WHEAT": 5},
            "inventories": [{}, {"WHEAT": 3}],
        },
        "market": {"inventory": {"WHEAT": 10000}, "prices": {"WHEAT": 25}},
        "town": {"unlocked_shops": []},
    }
    state = asdict(normalize_observation(obs))
    action = {
        "farmer": _unit("PICKUP", "WHEAT", 10),
        "hands": [_unit("PLACE", "WHEAT", 3)],
        "market": [_order("SELL", "WHEAT", 7), _order("STOP_QUEUE")],
    }
    return {"state": state, "previous_effect": {}, "canonical_action": action,
            "effects": {}, "terminal_result": 0, "final_margin": 0}


def test_masking_is_seed_deterministic_and_only_marks_active_semantic_fields():
    action = _row()["canonical_action"]
    first = mask_joint_action(action, random.Random(123), rate=0.5)
    second = mask_joint_action(action, random.Random(123), rate=0.5)
    assert first.masked_fields == second.masked_fields
    assert first.conditioned_action == second.conditioned_action
    assert all(field in {"op", "item", "quantity"} for _, _, field in first.masked_fields)


def test_full_mask_covers_active_fields_but_not_inactive_arguments():
    masked = mask_joint_action(_row()["canonical_action"], random.Random(0), rate=1.0)
    expected = {
        ("farmer", 0, "op"), ("farmer", 0, "item"), ("farmer", 0, "quantity"),
        ("hand", 0, "op"), ("hand", 0, "item"), ("hand", 0, "quantity"),
        ("market", 0, "op"), ("market", 0, "item"), ("market", 0, "quantity"),
        ("market", 1, "op"),
    }
    assert set(masked.masked_fields) == expected
    assert "item" not in masked.conditioned_action["market"][1].get("_mask_fields", [])
    assert "quantity" not in masked.conditioned_action["market"][1].get("_mask_fields", [])


def test_zero_mask_rate_has_zero_reconstruction_loss():
    torch.manual_seed(0)
    batch = collate_transitions([_row()])
    model = RecurrentIntentPolicy()
    loss = structure_reconstruction_loss(model, batch, mask_rate=0.0, rng=random.Random(1))
    assert loss.item() == 0.0


def test_structure_loss_backpropagates_into_runtime_encoder_and_action_embeddings():
    torch.manual_seed(1)
    batch = collate_transitions([_row()])
    model = RecurrentIntentPolicy()
    loss = structure_reconstruction_loss(model, batch, mask_rate=1.0, rng=random.Random(2))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    unit_grad = model.encoder.unit_encoder.input_proj[0].weight.grad
    action_grad = model.action_proj[0].weight.grad
    assert unit_grad is not None and torch.isfinite(unit_grad).all() and unit_grad.abs().sum() > 0
    assert action_grad is not None and torch.isfinite(action_grad).all() and action_grad.abs().sum() > 0


def test_batched_structure_loss_matches_reference_loss_and_gradients():
    from copy import deepcopy
    from kaggrl.v2_structure_pretrain import structure_reconstruction_loss_reference

    torch.manual_seed(42)
    batch = collate_transitions([_row(), _row(), _row()])
    reference = RecurrentIntentPolicy()
    batched = deepcopy(reference)
    reference_loss = structure_reconstruction_loss_reference(
        reference, batch, mask_rate=1.0, rng=random.Random(77),
    )
    batched_loss = structure_reconstruction_loss(
        batched, batch, mask_rate=1.0, rng=random.Random(77),
    )
    assert torch.allclose(reference_loss, batched_loss, rtol=1e-5, atol=1e-6)
    reference_loss.backward()
    batched_loss.backward()
    names = (
        "encoder.tile_encoder.net.0.weight",
        "decoder_cell.weight_ih",
        "action_proj.0.weight",
        "quantity_decoder.output.weight",
    )
    ref_params = dict(reference.named_parameters())
    batched_params = dict(batched.named_parameters())
    for name in names:
        assert ref_params[name].grad is not None and batched_params[name].grad is not None
        assert torch.allclose(ref_params[name].grad, batched_params[name].grad, rtol=2e-4, atol=2e-6), name


def test_batched_structure_loss_matches_reference_with_partial_mask_and_ledger_updates():
    from copy import deepcopy
    from kaggrl.v2_structure_pretrain import structure_reconstruction_loss_reference

    torch.manual_seed(123)
    batch = collate_transitions([_row(), _row()])
    reference = RecurrentIntentPolicy()
    batched = deepcopy(reference)
    reference_loss = structure_reconstruction_loss_reference(
        reference, batch, mask_rate=0.35, rng=random.Random(19),
    )
    batched_loss = structure_reconstruction_loss(
        batched, batch, mask_rate=0.35, rng=random.Random(19),
    )
    assert reference_loss.item() > 0
    assert torch.allclose(reference_loss, batched_loss, rtol=1e-5, atol=1e-6)
