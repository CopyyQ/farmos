import torch

from kaggrl.v2_quantity import QuantityDecoder, encode_quantity


def test_tensor_teacher_quantity_logits_match_legacy_path():
    torch.manual_seed(23)
    decoder = QuantityDecoder(96, hidden_dim=64, token_dim=16, max_digits=5).eval()
    context = torch.randn(4, 96)
    targets = [
        encode_quantity(None),
        encode_quantity(1),
        encode_quantity(12),
        encode_quantity(999),
    ]
    legacy_logits, legacy_mask = decoder.teacher_logits(context, targets)
    width = legacy_mask.shape[1]
    tensor_targets = torch.zeros((len(targets), width), dtype=torch.long)
    tensor_mask = torch.zeros((len(targets), width), dtype=torch.bool)
    for row, tokens in enumerate(targets):
        tensor_targets[row, : len(tokens)] = torch.tensor(tokens)
        tensor_mask[row, : len(tokens)] = True

    tensor_logits, got_mask = decoder.teacher_logits_tensor(
        context, tensor_targets, tensor_mask
    )
    assert torch.equal(got_mask, legacy_mask)
    assert torch.allclose(tensor_logits, legacy_logits, atol=1e-6, rtol=0.0)


def test_tensor_teacher_quantity_path_has_no_cpu_target_dependency():
    decoder = QuantityDecoder(96, hidden_dim=32, token_dim=8, max_digits=5).eval()
    context = torch.zeros(2, 96)
    targets = torch.tensor([[0, 0, 0], [2, 11, 0]], dtype=torch.long)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    logits, out_mask = decoder.teacher_logits_tensor(context, targets, mask)
    assert logits.shape == (2, 3, 12)
    assert torch.equal(out_mask, mask)
    assert torch.isfinite(logits).all()
