import pytest

from kaggrl.v2_action_schema import (
    JointAction,
    MarketSlot,
    UnitCommand,
    audit_replay_actions,
    parse_raw_action,
    raw_equal,
    semantic_equivalent,
    to_engine_action,
)


@pytest.mark.parametrize("hands", [0, 1, 16, 17, 32, 64, 128])
def test_dynamic_hand_count_round_trip(hands):
    raw = {"farmer": ["PASS"], "hands": [["PASS"]] * hands, "market": []}
    joint = parse_raw_action(raw, hands)
    assert len(joint.hands) == hands
    assert to_engine_action(joint) == raw


@pytest.mark.parametrize("q", [0, 1, 100, 101, 1000, 100000, -1, -1000])
def test_market_quantity_is_never_clamped_and_raw_round_trips(q):
    raw = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", q]]}
    joint = parse_raw_action(raw, 0)
    assert joint.market[0].quantity == q
    assert to_engine_action(joint) == raw


def test_stop_queue_and_nop_slot_are_distinct():
    stopped = parse_raw_action({"farmer": ["PASS"], "hands": [], "market": []}, 0)
    noop = parse_raw_action(
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 0]]}, 0
    )
    assert stopped.market[0].kind == "STOP_QUEUE"
    assert noop.market[0].kind == "NOP_SLOT"
    assert noop.market[1].kind == "STOP_QUEUE"
    assert not semantic_equivalent(to_engine_action(stopped), to_engine_action(noop), None)


def test_pickup_omitted_quantity_is_semantically_one_but_raw_syntax_stays_distinct():
    omitted = {"farmer": ["PICKUP", "WHEAT"], "hands": [], "market": []}
    explicit = {"farmer": ["PICKUP", "WHEAT", 1], "hands": [], "market": []}
    assert semantic_equivalent(omitted, explicit, None)
    assert not raw_equal(omitted, explicit)
    assert to_engine_action(parse_raw_action(omitted, 0)) == omitted
    assert to_engine_action(parse_raw_action(explicit, 0)) == explicit


def test_model_constructed_joint_action_emits_canonical_engine_action():
    joint = JointAction(
        farmer=UnitCommand("NORTH"),
        hands=(UnitCommand("PICKUP", "WHEAT", 7),),
        market=(MarketSlot("ORDER", "SELL", "MILK", 1000, ()), MarketSlot.stop()),
    )
    assert to_engine_action(joint) == {
        "farmer": ["NORTH"],
        "hands": [["PICKUP", "WHEAT", 7]],
        "market": [["SELL", "MILK", 1000]],
    }


def test_replay_action_audit_reports_no_loss_for_dynamic_and_large_quantity():
    raw = {
        "farmer": ["PASS"],
        "hands": [["PICKUP", "WHEAT", 1000]] * 17,
        "market": [["SELL", "MILK", 1000], ["SELL", "WHEAT", 0]],
    }
    replay = {
        "steps": [
            [{"observation": {"player": 0, "farms": [{"hands": [[0, 0]] * 17}]}, "action": {}}],
            [{"observation": {"player": 0, "farms": [{"hands": [[0, 0]] * 17}]}, "action": raw}],
        ]
    }
    failures = audit_replay_actions(replay, 0)
    assert failures == []
