import torch

from kaggrl.v4_option_model import (
    V4OptionPolicy,
    option_metrics,
    v4_option_loss,
)


def _targets(batch=2, steps=4):
    return {
        "route_target": torch.zeros(batch, steps, dtype=torch.long),
        "route_confidence": torch.ones(batch, steps),
        "route_gate_target": torch.ones(batch, steps),
        "market_target": torch.zeros(batch, steps, dtype=torch.long),
        "phase_target": torch.zeros(batch, steps, dtype=torch.long),
        "clock_target": torch.zeros(batch, steps, 2),
    }


def test_v4_option_model_outputs_one_strategy_decision_per_step():
    torch.manual_seed(7)
    model = V4OptionPolicy(
        input_dim=16,
        route_count=5,
        market_mode_count=3,
        hidden_dim=24,
    )
    obs = torch.randn(2, 6, 16)
    clock_context = torch.rand(2, 6, model.clock_dim)
    out, state = model.forward_sequence(obs, clock_context)
    assert out["route"].shape == (2, 6, 5)
    assert out["market"].shape == (2, 6, 3)
    assert out["phase"].shape == (2, 6, 5)
    assert out["clock"].shape == (2, 6, 2)
    assert out["value"].shape == (2, 6)
    assert state[0].shape == (1, 2, 24)
    assert torch.all((out["clock"] >= 0.0) & (out["clock"] <= 1.0))


def test_zero_route_confidence_removes_ambiguous_route_supervision():
    outputs = {
        "route": torch.randn(1, 3, 4, requires_grad=True),
        "route_gate_logits": torch.zeros(
            1, 3, requires_grad=True
        ),
        "market": torch.randn(1, 3, 3, requires_grad=True),
        "phase": torch.randn(1, 3, 5, requires_grad=True),
        "clock": torch.sigmoid(torch.randn(1, 3, 2, requires_grad=True)),
    }
    target = _targets(1, 3)
    target["route_confidence"].zero_()
    loss = v4_option_loss(outputs, **target)
    assert float(loss.route.detach()) == 0.0
    assert torch.isfinite(loss.total)


def test_clock_reconstruction_penalizes_wrong_absolute_time():
    target = _targets(1, 2)
    target["clock_target"] = torch.tensor(
        [[[0.10, 0.90], [0.90, 0.10]]],
        dtype=torch.float32,
    )
    common = {
        "route": torch.zeros(1, 2, 2),
        "route_gate_logits": torch.zeros(1, 2),
        "market": torch.zeros(1, 2, 3),
        "phase": torch.zeros(1, 2, 5),
    }
    perfect = {
        **common,
        "clock": target["clock_target"].clone(),
    }
    reversed_clock = {
        **common,
        "clock": target["clock_target"].flip(1),
    }
    good = v4_option_loss(perfect, **target)
    bad = v4_option_loss(reversed_clock, **target)
    assert float(good.clock) == 0.0
    assert float(bad.clock) > float(good.clock)


def test_metrics_report_clock_error_in_environment_turns():
    target = _targets(1, 2)
    target["clock_target"] = torch.tensor(
        [[[0.25, 0.75], [0.50, 0.50]]],
        dtype=torch.float32,
    )
    outputs = {
        "route": torch.tensor([[[5.0, 0.0], [5.0, 0.0]]]),
        "route_gate": torch.ones(1, 2),
        "market": torch.tensor([[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]),
        "phase": torch.tensor(
            [[[5.0, 0.0, 0.0, 0.0, 0.0],
              [5.0, 0.0, 0.0, 0.0, 0.0]]]
        ),
        "clock": target["clock_target"] + 0.01,
    }
    metrics = option_metrics(outputs, **target, last_step=719)
    assert metrics["route_acc_confident"] == 1.0
    assert metrics["market_acc"] == 1.0
    assert metrics["phase_acc"] == 1.0
    assert 7.18 < metrics["step_mae_turns"] < 7.20
    assert 7.18 < metrics["remaining_mae_turns"] < 7.20


def test_v4_option_model_refuses_missing_or_misaligned_clock_context():
    model = V4OptionPolicy(
        input_dim=8,
        route_count=2,
        market_mode_count=3,
        hidden_dim=12,
    )
    obs = torch.zeros(1, 4, 8)
    bad_clock = torch.zeros(1, 3, model.clock_dim)
    try:
        model.forward_sequence(obs, bad_clock)
    except ValueError as error:
        assert "clock_context" in str(error)
    else:
        raise AssertionError("misaligned clock context must fail")


def test_route_mask_removes_incompatible_logits_from_loss_and_metrics():
    target = _targets(1, 1)
    target["route_target"] = torch.tensor([[1]], dtype=torch.long)
    target["route_confidence"] = torch.ones(1, 1)
    target["route_mask"] = torch.tensor(
        [[[False, True, False]]],
        dtype=torch.bool,
    )
    outputs = {
        # Incompatible route 0 intentionally has the largest raw logit.
        "route": torch.tensor([[[100.0, 1.0, 50.0]]]),
        "route_gate_logits": torch.zeros(1, 1),
        "route_gate": torch.ones(1, 1),
        "market": torch.tensor([[[5.0, 0.0, 0.0]]]),
        "phase": torch.tensor(
            [[[5.0, 0.0, 0.0, 0.0, 0.0]]]
        ),
        "clock": target["clock_target"].clone(),
    }
    loss = v4_option_loss(outputs, **target)
    metrics = option_metrics(outputs, **target)
    assert torch.isfinite(loss.total)
    assert metrics["route_acc_confident"] == 1.0
