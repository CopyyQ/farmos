from kaggrl.v3_strategy import build_strategy_manifest, strategy_slot_for_team


def test_strategy_manifest_is_contiguous_and_deterministic():
    left = build_strategy_manifest([42, 7, 42, 9])
    right = build_strategy_manifest([9, 42, 7, 42])
    assert left.team_to_slot == {7: 0, 9: 1, 42: 2}
    assert left.slot_to_team == (7, 9, 42)
    assert left.sha256 == right.sha256
    assert strategy_slot_for_team(9, left) == 1


def test_strategy_slot_rejects_unknown_team():
    manifest = build_strategy_manifest([7, 9])
    try:
        strategy_slot_for_team(42, manifest)
    except KeyError as exc:
        assert "42" in str(exc)
    else:
        raise AssertionError("unknown team id must fail closed")
