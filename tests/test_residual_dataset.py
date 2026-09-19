import json

from kaggrl.residual_dataset import derive_market_residual, participant_seat
from kaggrl.residual_actions import apply_residual


def test_participant_seat_matches_exact_team_name():
    raw = json.dumps(["Alpha", "Top Team"])
    assert participant_seat(raw, "Top Team") == 1


def test_participant_seat_rejects_missing_or_ambiguous_team():
    for raw, team in [(json.dumps(["A", "B"]), "C"), (json.dumps(["A", "A"]), "A")]:
        try:
            participant_seat(raw, team)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_market_residual_reconstructs_first_two_effective_orders():
    base = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 3], ["SELL", "WOOL", 2], ["HIRE"]]}
    teacher = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WOOL", 1], ["BUY_SEED", "WHEAT", 4], ["HIRE"]]}
    residual = derive_market_residual(base, teacher)
    out = apply_residual(base, residual)
    assert out["market"][:2] == teacher["market"][:2]
    assert residual.changed


def test_episode_split_is_stable_and_episode_scoped():
    from kaggrl.residual_dataset import episode_split
    a = episode_split(108715272)
    b = episode_split(108715272)
    assert a == b
    assert a in {"train", "val", "test"}


def test_deterministic_sampling_is_stable():
    from kaggrl.residual_dataset import sample_residual_row
    args = (108715272, "Top Team", 313)
    assert sample_residual_row(*args, changed=True) == sample_residual_row(*args, changed=True)
    assert sample_residual_row(*args, changed=False) == sample_residual_row(*args, changed=False)
