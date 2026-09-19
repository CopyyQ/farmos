from __future__ import annotations

from dataclasses import asdict

from kaggrl.v2_observation import normalize_observation
from training.build_v3_recovery_dataset import _RecoveryCollectingAgent


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
