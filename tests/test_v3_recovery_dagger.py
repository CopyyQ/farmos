from __future__ import annotations

from dataclasses import asdict

from kaggrl.v2_observation import normalize_observation
from training.build_v3_recovery_dataset import (
    _RecoveryCollectingAgent,
    _terminal_outcome,
    canonicalize_teacher_action,
    project_teacher_action_to_executable,
    teacher_id_from_path,
    validate_recovery_row,
)


def test_terminal_outcome_reads_player_relative_money():
    obs = _observation()
    obs["farms"][0]["money"] = 12345
    obs["farms"][1]["money"] = 10000
    assert _terminal_outcome(obs) == (12345, 2345, 1)
    obs["player"] = 1
    assert _terminal_outcome(obs) == (10000, -2345, -1)


def test_teacher_id_from_extracted_directory():
    from pathlib import Path

    assert teacher_id_from_path(
        Path("/content/source_public/extracted_v50/main.py")
    ) == "v50"


def _observation():
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            {
                "money": 3000,
                "farmer": [4, 4],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": tiles,
            },
            {
                "money": 3000,
                "farmer": [8, 8],
                "hands": [],
                "hires_today": 0,
                "unlocked_quadrants": ["NW"],
                "tiles": [[None for _ in range(10)] for _ in range(10)],
            },
        ],
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {
            "inventory": {"WHEAT": 1000},
            "prices": {"WHEAT": 25},
        },
        "town": {"unlocked_shops": []},
    }


class _Candidate:
    def __init__(self):
        self.diagnostic_fixtures = []

    def __call__(self, observation, configuration=None):
        del configuration
        self.diagnostic_fixtures.append({
            "step": int(observation["step"]),
            "structured_state": asdict(normalize_observation(observation)),
            "previous_action": {},
            "previous_effect": {},
        })
        return {
            "farmer": ["NORTH"],
            "hands": [],
            "market": [],
        }




def test_v45_wheat_wash_is_legal_after_projection():
    obs = _observation()
    requested = canonicalize_teacher_action(
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [
                ["BUY_PRODUCT", "WHEAT", 70],
                ["SELL", "WHEAT", 70],
            ],
        },
        obs,
    )
    state = asdict(normalize_observation(obs))
    projected, corrections = project_teacher_action_to_executable(
        requested,
        state,
    )
    assert corrections == 0
    assert projected["market"][0]["op"] == "BUY_PRODUCT"
    assert projected["market"][0]["quantity"] == 70
    assert projected["market"][1]["op"] == "SELL"
    assert projected["market"][1]["quantity"] == 70

    row = {
        "episode_id": 1,
        "seat": 0,
        "step": 0,
        "state": state,
        "canonical_action": projected,
        "previous_action": {},
        "previous_effect": {},
        "effects": {},
        "final_own_money": 0,
        "final_margin": 0,
        "terminal_result": 0,
        "teacher_id": "v45",
        "teacher_version": "v45-test",
        "supervision_kind": "accepted_policy",
        "learner_model_sha256": "c" * 64,
        "strategy_slot": 0,
    }
    assert validate_recovery_row(row) is True


def test_dagger_projects_invalid_teacher_requests_to_effective_noops():
    obs = _observation()
    # WEST from x=0 is an engine no-op on this learner-visited state.
    obs["farms"][0]["farmer"] = [0, 0]
    candidate = _Candidate()

    def teacher(observation, configuration=None):
        del observation, configuration
        return {
            "farmer": ["WEST"],
            "hands": [],
            "market": [["SELL", "WHEAT", 100]],
        }

    collector = _RecoveryCollectingAgent(
        candidate,
        teacher,
        seat=0,
        episode_id=789,
        teacher_version="v45-test",
        model_sha="d" * 64,
        teacher_id="v45",
        supervision_kind="accepted_policy",
        strategy_slot=0,
    )
    learner_action = collector(obs, configuration={})
    assert learner_action["farmer"] == ["NORTH"]
    assert len(collector.rows) == 1
    label = collector.rows[0]["canonical_action"]
    assert label["farmer"]["op"] == "PASS"
    assert label["market"][0]["kind"] == "NOP_SLOT"
    assert collector.projected_rows == 1
    assert collector.projection_corrections >= 2



def test_dagger_can_drop_projected_teacher_labels():
    obs = _observation()
    obs["farms"][0]["farmer"] = [0, 0]
    candidate = _Candidate()

    def teacher(observation, configuration=None):
        del observation, configuration
        return {
            "farmer": ["WEST"],
            "hands": [],
            "market": [["SELL", "WHEAT", 100]],
        }

    collector = _RecoveryCollectingAgent(
        candidate,
        teacher,
        seat=0,
        episode_id=790,
        teacher_version="v45-test",
        model_sha="e" * 64,
        teacher_id="v45",
        supervision_kind="accepted_policy",
        strategy_slot=0,
        drop_projected_labels=True,
    )
    learner_action = collector(obs, configuration={})
    assert learner_action["farmer"] == ["NORTH"]
    assert collector.rows == []
    assert collector.projected_rows == 1
    assert collector.dropped_projected_rows == 1
    assert collector.projection_corrections >= 2


def test_dagger_teacher_failure_does_not_interrupt_learner_rollout():
    candidate = _Candidate()

    def broken_teacher(observation, configuration=None):
        del observation, configuration
        raise RuntimeError("teacher exploded")

    collector = _RecoveryCollectingAgent(
        candidate,
        broken_teacher,
        seat=0,
        episode_id=456,
        teacher_version="v45-test",
        model_sha="b" * 64,
        teacher_id="v45",
        supervision_kind="accepted_policy",
        strategy_slot=0,
    )
    executed = collector(_observation(), configuration={})
    assert executed["farmer"] == ["NORTH"]
    assert collector.rows == []
    assert len(collector.label_errors) == 1
    assert collector.label_errors[0]["stage"] == "teacher_call"
    assert "teacher exploded" in collector.label_errors[0]["error"]


def test_dagger_teacher_labels_state_but_candidate_action_executes():
    candidate = _Candidate()

    def teacher(observation, configuration=None):
        del observation, configuration
        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": [],
        }

    collector = _RecoveryCollectingAgent(
        candidate,
        teacher,
        seat=0,
        episode_id=123,
        teacher_version="v45-test",
        model_sha="a" * 64,
        teacher_id="v45",
        supervision_kind="accepted_policy",
        strategy_slot=0,
    )
    executed = collector(_observation(), configuration={})
    assert executed["farmer"] == ["NORTH"]
    assert len(collector.rows) == 1
    row = collector.rows[0]
    assert row["canonical_action"]["farmer"]["op"] == "PASS"
    assert row["teacher_id"] == "v45"
    assert row["supervision_kind"] == "accepted_policy"
    assert row["strategy_slot"] == 0
