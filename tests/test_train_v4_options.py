import pandas as pd

from training.train_v4_options import (
    _window_indices,
    promotion_gate,
)


def _metrics(**overrides):
    base = {
        "route_acc_confident": 0.80,
        "route_gate_balanced_acc": 0.80,
        "route_gate_positive_recall": 0.80,
        "route_gate_negative_recall": 0.90,
        "route_gate_precision": 0.50,
        "market_acc": 0.90,
        "market_balanced_acc": 0.80,
        "market_min_recall": 0.70,
        "phase_acc": 1.00,
        "step_mae_turns": 0.50,
        "remaining_mae_turns": 0.50,
        "route_clock_logit_std": 0.05,
        "market_clock_logit_std": 0.05,
        "terminal_margin_mae": 5000.0,
        "terminal_margin_win_acc": 0.70,
        "terminal_margin_corr": 0.50,
        "route_q_margin_corr": 0.50,
        "market_q_margin_corr": 0.50,
        "route_q_margin_win_acc": 0.70,
        "market_q_margin_win_acc": 0.70,
    }
    base.update(overrides)
    return base


def test_promotion_gate_requires_actual_clock_learning():
    assert promotion_gate(_metrics())["passed"] is True
    assert promotion_gate(
        _metrics(step_mae_turns=10.0)
    )["passed"] is False
    assert promotion_gate(
        _metrics(remaining_mae_turns=10.0)
    )["passed"] is False
    assert promotion_gate(
        _metrics(phase_acc=0.90)
    )["passed"] is False


def test_window_indices_never_cross_episode_or_seat():
    rows = []
    for episode_id, seat in ((1, 0), (1, 1), (2, 0)):
        for step in range(40):
            rows.append({
                "episode_id": episode_id,
                "seat": seat,
                "step": step,
                "split": "train",
            })
    frame = pd.DataFrame(rows)
    windows = _window_indices(frame, "train", 32)
    assert len(windows) == 6
    for idx in windows:
        part = frame.loc[idx]
        assert part["episode_id"].nunique() == 1
        assert part["seat"].nunique() == 1
        assert part["step"].astype(int).tolist() == list(
            range(
                int(part["step"].iloc[0]),
                int(part["step"].iloc[0]) + 32,
            )
        )


def test_window_indices_include_last_step_of_episode():
    frame = pd.DataFrame({
        "episode_id": [1] * 719,
        "seat": [0] * 719,
        "step": list(range(719)),
        "split": ["train"] * 719,
    })
    windows = _window_indices(frame, "train", 32)
    tail = frame.loc[windows[-1], "step"].astype(int).tolist()
    assert tail[0] == 687
    assert tail[-1] == 718


def test_promotion_gate_requires_margin_learning():
    assert promotion_gate(
        _metrics(terminal_margin_mae=15000.0)
    )["passed"] is False
    assert promotion_gate(
        _metrics(terminal_margin_win_acc=0.50)
    )["passed"] is False
    assert promotion_gate(
        _metrics(route_q_margin_corr=0.05)
    )["passed"] is False
    assert promotion_gate(
        _metrics(market_q_margin_corr=0.05)
    )["passed"] is False
