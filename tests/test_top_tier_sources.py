from kaggrl.top_tier_sources import choose_submission_roles, sample_weight


def test_choose_active_best_and_recent_good_update():
    submissions = [
        {"id": 20, "dateSubmitted": "2026-09-16T17:00:00Z", "publicScore": "3020.0"},
        {"id": 10, "dateSubmitted": "2026-09-15T10:00:00Z", "publicScore": "3100.0"},
    ]
    roles = choose_submission_roles("3100.0", submissions)
    assert roles["active_best"]["id"] == 10
    assert roles["latest_qualified"]["id"] == 20


def test_rejects_obviously_regressive_latest_update():
    submissions = [
        {"id": 20, "dateSubmitted": "2026-09-16T17:00:00Z", "publicScore": "1400.0"},
        {"id": 10, "dateSubmitted": "2026-09-15T10:00:00Z", "publicScore": "3100.0"},
    ]
    roles = choose_submission_roles("3100.0", submissions)
    assert roles["active_best"]["id"] == 10
    assert roles["latest_qualified"] is None


def test_recency_weight_prioritizes_rank_and_freshness():
    newest_rank1 = sample_weight(rank=1, age_days=0.0, role="active_best", score_ratio=1.0)
    older_rank1 = sample_weight(rank=1, age_days=7.0, role="active_best", score_ratio=1.0)
    newest_rank10 = sample_weight(rank=10, age_days=0.0, role="active_best", score_ratio=1.0)
    recent_update = sample_weight(rank=1, age_days=0.0, role="latest_qualified", score_ratio=0.98)
    assert newest_rank1 > older_rank1
    assert newest_rank1 > newest_rank10
    assert recent_update > older_rank1
    assert 0.0 < recent_update <= newest_rank1


def test_eligible_recent_episodes_filters_and_sorts():
    from kaggrl.top_tier_sources import eligible_recent_episodes

    rows = [
        {"id": 1, "createTime": "2026-09-16T10:00:00Z", "state": "COMPLETED", "type": "EPISODE_TYPE_PUBLIC"},
        {"id": 2, "createTime": "2026-09-16T12:00:00Z", "state": "COMPLETED", "type": "EPISODE_TYPE_PUBLIC"},
        {"id": 3, "createTime": "2026-09-16T13:00:00Z", "state": "ERROR", "type": "EPISODE_TYPE_PUBLIC"},
        {"id": 4, "createTime": "2026-09-16T14:00:00Z", "state": "COMPLETED", "type": "EPISODE_TYPE_PRIVATE"},
    ]
    selected = eligible_recent_episodes(rows, limit=2)
    assert [row["id"] for row in selected] == [2, 1]
