import pytest

from kaggrl.live_top10_manifest import (
    ManifestResolutionError,
    resolve_active_submission,
    select_latest_candidate,
)


def test_active_submission_matches_team_and_exact_date():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03Z"}
    submissions = [
        {"id": 10, "teamId": 7, "dateSubmitted": "2026-09-16T23:00:00Z", "publicScore": "3100"},
        {"id": 11, "teamId": 7, "dateSubmitted": "2026-09-17T01:02:03Z", "publicScore": "3090"},
    ]
    assert resolve_active_submission(row, submissions)["id"] == 11


def test_active_submission_does_not_use_nearest_score():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03Z", "score": "3090"}
    submissions = [
        {"id": 10, "teamId": 7, "dateSubmitted": "2026-09-16T23:00:00Z", "publicScore": "3090"},
        {"id": 11, "teamId": 7, "dateSubmitted": "2026-09-17T01:02:03Z", "publicScore": "3000"},
    ]
    assert resolve_active_submission(row, submissions)["id"] == 11


def test_active_submission_aborts_when_missing():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03Z"}
    with pytest.raises(ManifestResolutionError):
        resolve_active_submission(row, [])


def test_active_submission_aborts_when_duplicate_exact_match():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03Z"}
    submissions = [
        {"id": 11, "teamId": 7, "dateSubmitted": "2026-09-17T01:02:03Z"},
        {"id": 12, "teamId": 7, "dateSubmitted": "2026-09-17T01:02:03+00:00"},
    ]
    with pytest.raises(ManifestResolutionError):
        resolve_active_submission(row, submissions)


def test_latest_candidate_is_newest_newer_qualified_submission():
    active = {"id": 11, "teamId": 7, "dateSubmitted": "2026-09-17T01:00:00Z", "publicScore": "3000"}
    submissions = [
        active,
        {"id": 12, "teamId": 7, "dateSubmitted": "2026-09-17T02:00:00Z", "publicScore": "2800"},
        {"id": 13, "teamId": 7, "dateSubmitted": "2026-09-17T03:00:00Z", "publicScore": "2860"},
        {"id": 14, "teamId": 7, "dateSubmitted": "2026-09-17T04:00:00Z", "publicScore": None},
    ]
    assert select_latest_candidate(active, submissions)["id"] == 13


from datetime import datetime, timezone
from types import SimpleNamespace


def test_resolver_accepts_real_kaggle_sdk_object_shapes():
    row = SimpleNamespace(
        team_id=7,
        team_name="Team Seven",
        submission_date=datetime(2026, 9, 17, 1, 2, 3, tzinfo=timezone.utc),
        score="3090",
    )
    submissions = [
        SimpleNamespace(id=10, date_submitted=datetime(2026, 9, 16, 23, tzinfo=timezone.utc), public_score="3090"),
        SimpleNamespace(id=11, date_submitted=datetime(2026, 9, 17, 1, 2, 3, tzinfo=timezone.utc), public_score="3000"),
    ]
    assert resolve_active_submission(row, submissions).id == 11


def test_build_snapshot_freezes_one_leaderboard_response():
    from kaggrl.live_top10_manifest import build_snapshot

    class FakeApi:
        def __init__(self):
            self.leaderboard_calls = 0
        def competition_leaderboard_view(self, competition, page_size=20):
            self.leaderboard_calls += 1
            return [SimpleNamespace(team_id=7, team_name="A", submission_date=datetime(2026,9,17,1,tzinfo=timezone.utc), score="3000")]
        def competition_team_submissions(self, team_id):
            assert team_id == 7
            return [SimpleNamespace(id=11, date_submitted=datetime(2026,9,17,1,tzinfo=timezone.utc), public_score="3000")]

    api = FakeApi()
    snapshot = build_snapshot(api, "kaggriculture", top_k=1)
    assert api.leaderboard_calls == 1
    assert snapshot["teams"][0]["roles"]["active_best"]["id"] == 11
    assert snapshot["teams"][0]["team_id"] == 7
    assert len(snapshot["leaderboard_sha256"]) == 64


def test_collect_episode_sources_filters_public_completed_and_dedupes():
    from kaggrl.live_top10_manifest import collect_episode_sources

    agent_a = SimpleNamespace(submission_id=11, index=0, reward=100, team_name="A", team_id=7)
    agent_b = SimpleNamespace(submission_id=21, index=1, reward=90, team_name="B", team_id=8)
    common = SimpleNamespace(
        id=1001,
        create_time=datetime(2026,9,17,4,tzinfo=timezone.utc),
        end_time=datetime(2026,9,17,4,10,tzinfo=timezone.utc),
        state="COMPLETED",
        type="EPISODE_TYPE_PUBLIC",
        agents=[agent_a, agent_b],
    )
    private = SimpleNamespace(
        id=1002,
        create_time=datetime(2026,9,17,3,tzinfo=timezone.utc),
        end_time=datetime(2026,9,17,3,10,tzinfo=timezone.utc),
        state="COMPLETED",
        type="EPISODE_TYPE_PRIVATE",
        agents=[agent_a, agent_b],
    )

    snapshot = {
        "teams": [
            {"rank":1,"team_id":7,"team_name":"A","leaderboard_score":3000,"roles":{"active_best":{"id":11,"dateSubmitted":"2026-09-17T01:00:00+00:00","publicScore":"3000"},"latest_candidate":None}},
            {"rank":2,"team_id":8,"team_name":"B","leaderboard_score":2990,"roles":{"active_best":{"id":21,"dateSubmitted":"2026-09-17T01:00:00+00:00","publicScore":"2990"},"latest_candidate":None}},
        ]
    }

    class FakeEpisodeApi:
        def competition_list_episodes(self, submission_id):
            return [private, common]

    result = collect_episode_sources(FakeEpisodeApi(), snapshot, active_limit=2, candidate_limit=1)
    assert len(result["episodes"]) == 1
    assert result["episodes"][0]["episode"]["id"] == 1001
    assert {s["submission_id"] for s in result["episodes"][0]["sources"]} == {11, 21}
    assert {s["seat"] for s in result["episodes"][0]["sources"]} == {0, 1}


def test_active_submission_accepts_one_millisecond_sdk_timestamp_skew():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03.966Z"}
    submissions = [
        {"id": 11, "dateSubmitted": "2026-09-17T01:02:03.967Z", "publicScore": "3000"},
        {"id": 10, "dateSubmitted": "2026-09-16T23:00:00Z", "publicScore": "3100"},
    ]
    assert resolve_active_submission(row, submissions)["id"] == 11


def test_active_submission_rejects_more_than_one_millisecond_skew():
    row = {"teamId": 7, "submissionDate": "2026-09-17T01:02:03.966Z"}
    submissions = [
        {"id": 11, "dateSubmitted": "2026-09-17T01:02:03.968Z", "publicScore": "3000"},
    ]
    with pytest.raises(ManifestResolutionError):
        resolve_active_submission(row, submissions)
