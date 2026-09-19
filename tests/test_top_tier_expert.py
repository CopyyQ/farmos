import torch

from kaggrl.actions import ActionCodec
from kaggrl.constants import ITEM_TO_ID, MARKET_TO_ID
from kaggrl.top_tier_expert import TopTierExpertPolicy, decode_action, expert_bc_loss


def test_expert_output_shapes():
    model = TopTierExpertPolicy(1024, 128)
    codec = ActionCodec()
    obs = torch.zeros(2, 5, 1024)
    teacher = torch.zeros(2, 5, codec.width, dtype=torch.long)
    out, state = model.forward_sequence(obs, teacher_actions=teacher)
    assert out["farmer_op"].shape == (2, 5, 18)
    assert out["hand_op"].shape == (2, 5, 16, 18)
    assert out["market_op"].shape == (2, 5, 10, 7)
    assert out["market_item"].shape[-1] == max(ITEM_TO_ID.values()) + 1
    assert state[0].shape == (2, 128)


def test_bc_loss_is_finite_and_backpropagates():
    model = TopTierExpertPolicy(1024, 64)
    codec = ActionCodec()
    obs = torch.randn(2, 3, 1024)
    actions = torch.zeros(2, 3, codec.width, dtype=torch.long)
    masks = torch.ones(2, 3, codec.width)
    weights = torch.ones(2, 3)
    out, _ = model.forward_sequence(obs, teacher_actions=actions)
    loss, parts = expert_bc_loss(out, actions, masks, weights)
    assert torch.isfinite(loss)
    assert set(parts) == {"farmer", "hands", "market"}
    loss.backward()
    assert model.input_proj.weight.grad is not None

def test_decode_action_stops_market_at_none():
    model = TopTierExpertPolicy(1024, 32)
    out, _ = model.forward_sequence(torch.zeros(1, 1, 1024))
    for value in out.values():
        value.zero_()
    out["market_op"][0, 0, 0, MARKET_TO_ID["BUY_PRODUCT"]] = 10
    out["market_item"][0, 0, 0, ITEM_TO_ID["WHEAT"]] = 10
    out["market_qty"][0, 0, 0, 3] = 10
    out["market_op"][0, 0, 1, MARKET_TO_ID["SELL"]] = 10
    out["market_item"][0, 0, 1, ITEM_TO_ID["WHEAT"]] = 10
    out["market_qty"][0, 0, 1, 2] = 10
    out["market_op"][0, 0, 2, MARKET_TO_ID["NONE"]] = 10
    observation = {"player": 0, "farms": [{"hands": []}]}
    action = decode_action(out, observation)
    assert action["market"] == [["BUY_PRODUCT", "WHEAT", 3], ["SELL", "WHEAT", 2]]
