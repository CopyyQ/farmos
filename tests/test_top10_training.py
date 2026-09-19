import pandas as pd
import pytest

from kaggrl.top10_training import make_episode_chunks, validation_score


def test_chunks_never_cross_episode_boundary():
    frame = pd.DataFrame({
        "episode_id": [1] * 55 + [2] * 61,
        "step": list(range(55)) + list(range(61)),
        "split": ["train"] * 116,
    })
    chunks = make_episode_chunks(frame, seq_len=48)
    assert [c.length for c in chunks] == [48, 7, 48, 13]
    assert [c.episode_id for c in chunks] == [1, 1, 2, 2]
    for chunk in chunks:
        assert frame.loc[chunk.indices, "episode_id"].nunique() == 1


def test_validation_score_rewards_farmer_and_market_without_hand_domination():
    metrics = {"farmer_op_acc": 0.8, "hand_op_acc": 0.2, "market_op_acc": 0.7}
    assert validation_score(metrics) == pytest.approx(0.45 * 0.8 + 0.20 * 0.2 + 0.35 * 0.7)


def test_training_phase_uses_farmer_warmup_before_joint_training():
    from kaggrl.top10_training import training_phase

    assert [training_phase(epoch, 8) for epoch in range(1, 9)] == [
        "farmer", "farmer", "farmer", "farmer",
        "joint", "joint", "joint", "joint",
    ]
    assert [training_phase(epoch, 2) for epoch in range(1, 3)] == ["farmer", "joint"]
