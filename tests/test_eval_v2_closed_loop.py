from pathlib import Path

from evaluation.eval_v2_closed_loop import (
    GameSpec,
    artifact_name,
    build_game_specs,
    summarize_closed_loop,
)


def test_game_scheduler_is_both_seat_complete_and_deterministic():
    specs = build_game_specs([12, 11], ["v17", "starter"])
    assert len(specs) == 8
    expected = {
        (seed, seat, opponent)
        for seed in (11, 12) for opponent in ("starter", "v17") for seat in (0, 1)
    }
    assert {(s.seed, s.learner_seat, s.opponent) for s in specs} == expected
    assert specs == sorted(specs, key=lambda s: (s.seed, s.opponent, s.learner_seat))
    names = [artifact_name(spec, "candidate") for spec in specs]
    assert len(names) == len(set(names))
    assert all(name.endswith(".json") for name in names)


def test_game_spec_rejects_invalid_seat():
    try:
        GameSpec(seed=1, learner_seat=2, opponent="starter")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid learner seat must be rejected")

def test_closed_loop_summary_uses_observed_economic_events_and_money_trajectory():
    games = [
        {
            "seed": 11, "learner_seat": 0, "opponent": "starter",
            "final_money": 1200, "margin": 200, "min_money": 700,
            "median_money": 900, "max_money": 1200, "max_hands": 4,
            "effective_family_counts": {"movement": 2, "acquisition": 1,
                "production": 1, "deposit": 1, "sale": 1, "hire": 1},
            "land_unlocks": 1, "longest_effectless_streak": 8,
            "statuses": ["DONE", "DONE"], "finite": True,
        },
        {
            "seed": 11, "learner_seat": 1, "opponent": "starter",
            "final_money": 800, "margin": -100, "min_money": 600,
            "median_money": 750, "max_money": 950, "max_hands": 2,
            "effective_family_counts": {"movement": 1, "service": 2, "harvest": 1,
                "deposit": 1, "sale": 1},
            "land_unlocks": 0, "longest_effectless_streak": 12,
            "statuses": ["DONE", "DONE"], "finite": True,
        },
    ]
    summary = summarize_closed_loop(games)
    assert summary["games"] == 2
    assert summary["median_final_money"] == 1000.0
    assert summary["effective_family_counts"]["sale"] == 2
    assert summary["effective_family_counts"]["movement"] == 3
    assert summary["land_unlocks"] == 1
    assert summary["max_hands"] == 4
    assert summary["max_effectless_streak"] == 12
    assert summary["all_done"] is True and summary["all_finite"] is True


def test_closed_loop_summary_fails_closed_on_explicit_integrity_fields():
    base = {
        "seed": 21, "learner_seat": 0, "opponent": "starter",
        "final_money": 1000, "margin": 0, "max_hands": 1,
        "effective_family_counts": {"movement": 1, "sale": 1},
        "land_unlocks": 0, "longest_effectless_streak": 1,
        "statuses": ["DONE", "DONE"], "finite": True,
        "schema_valid": True, "timeout": False,
    }
    summary = summarize_closed_loop([base])
    assert summary["all_schema_valid"] is True
    assert summary["all_no_timeout"] is True

    missing_schema = dict(base)
    missing_schema.pop("schema_valid")
    assert summarize_closed_loop([missing_schema])["all_schema_valid"] is False

    missing_timeout = dict(base)
    missing_timeout.pop("timeout")
    assert summarize_closed_loop([missing_timeout])["all_no_timeout"] is False
