import pytest

from kaggrl.clock import CLOCK_FEATURES, PHASE_NAMES, resolve_clock, resolve_step


def test_missing_or_none_step_reconstructs_absolute_turn():
    missing = {"day": 29, "hour": 23}
    explicit_none = {"step": None, "day": 29, "hour": 23}
    for obs in (missing, explicit_none):
        clock = resolve_clock(obs)
        assert clock.step == 719
        assert clock.day == 29
        assert clock.hour == 23
        assert clock.remaining_steps == 0
        assert resolve_step(obs) == 719


def test_configuration_controls_clock_geometry():
    clock = resolve_clock(
        {"day": 2, "hour": 3},
        {"turnsPerDay": 10, "episodeSteps": 100},
    )
    assert (clock.step, clock.day, clock.hour) == (23, 2, 3)
    assert clock.remaining_steps == 76


@pytest.mark.parametrize(
    ("step", "phase"),
    [(0, "early"), (143, "early"), (144, "growth"), (359, "growth"),
     (360, "mid"), (575, "mid"), (576, "harvest"), (647, "harvest"),
     (648, "liquidation"), (719, "liquidation")],
)
def test_clock_phase_boundaries(step, phase):
    clock = resolve_clock({"step": step})
    assert PHASE_NAMES[clock.phase_index] == phase


def test_clock_features_expose_remaining_time_and_phase():
    early = dict(zip(CLOCK_FEATURES, resolve_clock({"step": 0}).features()))
    late = dict(zip(CLOCK_FEATURES, resolve_clock({"step": 719}).features()))
    assert early["step_norm"] == 0.0
    assert early["remaining_steps_norm"] == 1.0
    assert early["phase:early"] == 1.0
    assert late["step_norm"] == 1.0
    assert late["remaining_steps_norm"] == 0.0
    assert late["phase:liquidation"] == 1.0
