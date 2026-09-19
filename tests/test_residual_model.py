import torch


def test_residual_policy_head_shapes():
    from kaggrl.residual_model import ResidualPolicy
    model = ResidualPolicy(input_dim=1024, hidden_dim=64, order_vocab_size=137)
    x = torch.randn(3, 11, 1024)
    out, state = model.forward_sequence(x)
    assert out["edit0"].shape == (3, 11, 3)
    assert out["edit1"].shape == (3, 11, 3)
    assert out["order0"].shape == (3, 11, 137)
    assert out["order1"].shape == (3, 11, 137)
    assert out["value"].shape == (3, 11)
    assert state[0].shape == (1, 3, 64)
    assert state[1].shape == (1, 3, 64)


def test_bc_loss_masks_order_head_unless_replace():
    from kaggrl.residual_model import residual_bc_loss
    logits = {
        "edit0": torch.zeros(2, 3), "edit1": torch.zeros(2, 3),
        "order0": torch.zeros(2, 5), "order1": torch.zeros(2, 5),
    }
    labels = {
        "edit0": torch.tensor([0, 2]), "edit1": torch.tensor([1, 0]),
        "order0": torch.tensor([4, 3]), "order1": torch.tensor([2, 1]),
    }
    loss, metrics = residual_bc_loss(logits, labels)
    assert torch.isfinite(loss)
    assert metrics["replace_slots"] == 1
