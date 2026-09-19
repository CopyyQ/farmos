import pytest
import torch

from kaggrl.v2_quantity import (
    END_ID,
    OMIT_ID,
    QuantityDecoder,
    decode_quantity,
    encode_quantity,
)


@pytest.mark.parametrize("value", [None, 0, 1, 100, 101, 1000, 100000, 10**12])
def test_quantity_round_trip_without_clamp(value):
    tokens = encode_quantity(value)
    assert decode_quantity(tokens) == value


def test_negative_quantity_is_not_a_neural_canonical_value():
    with pytest.raises(ValueError, match="nonnegative"):
        encode_quantity(-1)


def test_unterminated_digit_stream_is_rejected():
    with pytest.raises(ValueError, match="unterminated"):
        decode_quantity(encode_quantity(123)[:-1])


def test_decode_respects_safety_digit_bound_without_training_clamp():
    tokens = encode_quantity(10**20)
    assert decode_quantity(tokens, max_digits=21) == 10**20
    with pytest.raises(ValueError, match="maximum digit"):
        decode_quantity(tokens, max_digits=20)


def test_omit_and_end_have_distinct_semantics():
    assert decode_quantity([OMIT_ID]) is None
    with pytest.raises(ValueError):
        decode_quantity([END_ID])


def test_quantity_decoder_teacher_logits_cover_each_target_token():
    torch.manual_seed(7)
    decoder = QuantityDecoder(context_dim=16, hidden_dim=12, token_dim=8)
    context = torch.randn(2, 16, requires_grad=True)
    targets = [encode_quantity(1000), encode_quantity(None)]
    logits, mask = decoder.teacher_logits(context, targets)
    assert logits.shape[:2] == mask.shape
    assert logits.shape[-1] == decoder.vocab_size
    assert mask.sum(dim=1).tolist() == [len(targets[0]), len(targets[1])]
    loss = logits[mask].square().mean()
    loss.backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
