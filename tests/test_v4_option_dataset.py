import pytest

from kaggrl.macro_policy import MacroPolicy
from kaggrl.v4_option_dataset import (
    action_distance,
    build_route_signature_table,
    contiguous_window_starts,
    route_label_for_horizon,
    route_labels_for_episode,
)
from kaggrl.v45_macro_data import load_v45_macro_data


def _obs(step):
    day, hour = divmod(step, 24)
    return {
        "player": 0,
        "step": step,
        "day": day,
        "hour": hour,
        "town": {"unlocked_shops": []},
    }


def test_action_distance_is_zero_only_for_same_semantics():
    left = {
        "farmer": ["NORTH"],
        "hands": [["PASS"], ["WATER"]],
        "market": [["SELL", "WHEAT", 2]],
    }
    assert action_distance(left, left) == 0.0
    changed = {
        "farmer": ["WEST"],
        "hands": [["PASS"], ["WATER"]],
        "market": [["SELL", "WHEAT", 2]],
    }
    assert action_distance(left, changed) == 3.0


def test_horizon_route_label_distinguishes_real_v45_routes():
    routes, new_routes, old_routes = load_v45_macro_data()
    macro = MacroPolicy(routes, new_routes, old_routes)
    observations = [_obs(step) for step in range(144, 152)]
    teacher = [
        macro.action_for_route(obs, 9)
        for obs in observations
    ]
    label = route_label_for_horizon(
        observations,
        teacher,
        route_ids=(9, 100),
    )
    assert label.route_id == 9
    assert label.best_score == 0.0
    assert label.second_score > 0.0
    assert label.confidence > 0.0


def test_contiguous_windows_cover_the_late_game_tail():
    starts = contiguous_window_starts(range(719), 32)
    assert starts[0] == 0
    assert starts[-1] == 687
    covered = set()
    for start in starts:
        covered.update(range(start, start + 32))
    assert covered == set(range(719))


def test_contiguous_windows_reject_missing_environment_step():
    steps = list(range(64))
    steps[31] = 32
    with pytest.raises(ValueError, match="non-contiguous"):
        contiguous_window_starts(steps, 32)


def test_short_episode_does_not_make_fake_padded_window():
    assert contiguous_window_starts(range(7), 32) == ()


def test_vectorized_episode_route_labels_match_single_horizon_label():
    routes, new_routes, old_routes = load_v45_macro_data()
    macro = MacroPolicy(routes, new_routes, old_routes)
    observations = [_obs(step) for step in range(144, 160)]
    teacher = [
        macro.action_for_route(obs, 9)
        for obs in observations
    ]
    table = build_route_signature_table(route_ids=(9, 100))
    vectorized = route_labels_for_episode(
        observations,
        teacher,
        horizon=8,
        route_ids=(9, 100),
        signature_table=table,
    )
    single = route_label_for_horizon(
        observations[:8],
        teacher[:8],
        route_ids=(9, 100),
        signature_table=table,
    )
    assert vectorized[0].route_id == single.route_id == 9
    assert vectorized[0].best_score == pytest.approx(single.best_score)
    assert vectorized[0].second_score == pytest.approx(single.second_score)
    assert vectorized[0].confidence == pytest.approx(single.confidence)
