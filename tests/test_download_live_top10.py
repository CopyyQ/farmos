from datetime import datetime, timezone
from types import SimpleNamespace

from training.download_live_top10 import build_live_manifest


def _episode(submission_id):
    agent = SimpleNamespace(
        submission_id=submission_id,
        index=0,
        reward=123.0,
        team_name="A",
        team_id=7,
    )
    return SimpleNamespace(
        id=1001,
        create_time=datetime(2026, 9, 17, 4, tzinfo=timezone.utc),
        end_time=datetime(2026, 9, 17, 4, 5, tzinfo=timezone.utc),
        state="COMPLETED",
        type="EPISODE_TYPE_PUBLIC",
        agents=[agent],
    )


class FakeApi:
    def competition_leaderboard_view(self, competition, page_size=20):
        return [SimpleNamespace(
            team_id=7,
            team_name="A",
            submission_date=datetime(2026, 9, 17, 1, tzinfo=timezone.utc),
            score="3000",
        )]
    def competition_team_submissions(self, team_id):
        return [SimpleNamespace(
            id=11,
            date_submitted=datetime(2026, 9, 17, 1, tzinfo=timezone.utc),
            public_score="3000",
        )]

    def competition_list_episodes(self, submission_id):
        return [_episode(submission_id)]


def test_build_live_manifest_writes_v2_snapshot(tmp_path):
    manifest_path, manifest = build_live_manifest(
        FakeApi(),
        tmp_path,
        competition="kaggriculture",
        top_k=1,
        active_episodes=12,
        candidate_episodes=3,
    )
    assert manifest_path == tmp_path / "manifests" / "live_top10_snapshot.json"
    assert manifest_path.exists()
    assert manifest["teams"][0]["roles"]["active_best"]["id"] == 11
    assert manifest["episodes"][0]["episode"]["id"] == 1001
    assert manifest["episodes"][0]["sources"][0]["seat"] == 0
    assert "live_top10_bootstrap" not in manifest_path.read_text()


def test_load_live_manifest_reuses_exact_frozen_snapshot(tmp_path):
    from training.download_live_top10 import load_live_manifest

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    expected = {"snapshot_time_utc": "2026-09-17T02:24:27+00:00", "teams": [], "episodes": []}
    path = manifest_dir / "live_top10_snapshot.json"
    path.write_text(__import__("json").dumps(expected), encoding="utf-8")

    loaded_path, loaded = load_live_manifest(tmp_path)
    assert loaded_path == path
    assert loaded == expected


def test_prepare_manifest_reuse_does_not_refresh_leaderboard(tmp_path):
    from training.download_live_top10 import prepare_manifest

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    frozen = {
        "snapshot_time_utc": "2026-09-17T02:24:27+00:00",
        "leaderboard_sha256": "abc",
        "teams": [],
        "episodes": [],
    }
    (manifest_dir / "live_top10_snapshot.json").write_text(
        __import__("json").dumps(frozen), encoding="utf-8"
    )

    class NoRefreshApi:
        def competition_leaderboard_view(self, *args, **kwargs):
            raise AssertionError("leaderboard must not be refreshed")

    _, loaded = prepare_manifest(
        NoRefreshApi(), tmp_path, reuse_manifest=True, competition="kaggriculture"
    )
    assert loaded == frozen


def test_select_episode_slice_is_deterministic_and_non_overlapping():
    from training.download_live_top10 import select_episode_slice

    manifest = {"episodes": [{"episode": {"id": i}} for i in range(10)]}
    a = select_episode_slice(manifest, offset=0, limit=4)
    b = select_episode_slice(manifest, offset=4, limit=4)
    c = select_episode_slice(manifest, offset=8, limit=0)
    assert [x["episode"]["id"] for x in a] == [0, 1, 2, 3]
    assert [x["episode"]["id"] for x in b] == [4, 5, 6, 7]
    assert [x["episode"]["id"] for x in c] == [8, 9]
    assert not ({x["episode"]["id"] for x in a} & {x["episode"]["id"] for x in b})
