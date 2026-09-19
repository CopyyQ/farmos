import torch

from kaggrl.v2_tensorize import (
    ECONOMY_FEATURES,
    EFFECT_FEATURES,
    PREV_ACTION_GLOBAL_FEATURES,
)
from kaggrl.v3_temporal import TemporalCore


def _context(ref):
    batch = ref.shape[0]
    return (
        ref.new_zeros((batch, len(PREV_ACTION_GLOBAL_FEATURES))),
        ref.new_zeros((batch, len(EFFECT_FEATURES))),
        ref.new_zeros((batch, len(ECONOMY_FEATURES))),
    )


def _step(core, fused, state=None, active_mask=None):
    previous, effect, economy = _context(fused)
    return core.step(
        fused, previous, effect, economy, state,
        active_mask=active_mask,
    )

def _roll(core, sequence):
    state = None
    outputs = []
    for time_index in range(sequence.shape[1]):
        fused, _, state, _ = _step(
            core, sequence[:, time_index], state,
        )
        outputs.append(fused.detach().clone())
    return outputs, state


def test_earlier_step_changes_later_fused_state():
    torch.manual_seed(17)
    core = TemporalCore().eval()
    base = torch.zeros(1, 6, 256)
    changed = base.clone()
    changed[:, 1, 0] = 1.0
    left, _ = _roll(core, base)
    right, _ = _roll(core, changed)
    assert not torch.allclose(left[4], right[4])


def test_permuting_time_order_changes_later_output():
    torch.manual_seed(17)
    core = TemporalCore().eval()
    sequence = torch.zeros(1, 5, 256)
    sequence[:, 0, 0] = 1.0
    sequence[:, 1, 1] = 1.0
    permuted = sequence.clone()
    permuted[:, [0, 1]] = permuted[:, [1, 0]]
    left, _ = _roll(core, sequence)
    right, _ = _roll(core, permuted)
    assert not torch.allclose(left[-1], right[-1])


def test_future_perturbation_cannot_change_earlier_output():
    torch.manual_seed(17)
    core = TemporalCore().eval()
    base = torch.zeros(1, 6, 256)
    changed = base.clone()
    changed[:, 5, 0] = 1.0
    left, _ = _roll(core, base)
    right, _ = _roll(core, changed)
    assert torch.equal(left[4], right[4])


def test_inactive_row_does_not_mutate_temporal_state():
    torch.manual_seed(17)
    core = TemporalCore().eval()
    fused = torch.zeros(2, 256)
    _, _, state, _ = _step(core, fused)
    before = (
        state.h[1].clone(), state.c[1].clone(),
        state.memory[1].clone(), state.valid_length[1].clone(),
        state.write_pos[1].clone(),
    )
    changed = fused.clone()
    changed[1, 0] = 9.0
    _, _, after, _ = _step(
        core, changed, state,
        active_mask=torch.tensor([True, False]),
    )
    got = (
        after.h[1], after.c[1], after.memory[1],
        after.valid_length[1], after.write_pos[1],
    )
    for expected, value in zip(before, got):
        assert torch.equal(expected, value)


def test_ring_buffer_keeps_last_32_tokens_after_33_steps():
    torch.manual_seed(17)
    core = TemporalCore().eval()
    state = None
    for step in range(33):
        fused = torch.zeros(1, 256)
        fused[:, 0] = float(step)
        _, _, state, _ = _step(core, fused, state)
    assert int(state.valid_length.item()) == 32
    assert int(state.write_pos.item()) == 1
    ordered = core.ordered_memory(state)[0]
    assert ordered.shape == (32, 128)
    assert torch.allclose(ordered[-1], state.memory[0, 0])
    assert torch.isfinite(ordered).all()


def _sequence_reference(core, fused, previous, effect, economy, state=None):
    outputs = []
    intents = []
    weights = []
    entropy = []
    mean_age = []
    current = state
    for time_index in range(fused.shape[1]):
        out, intent, current, diagnostics = core.step(
            fused[:, time_index],
            previous[:, time_index],
            effect[:, time_index],
            economy[:, time_index],
            current,
        )
        outputs.append(out)
        intents.append(intent)
        weights.append(diagnostics.attention_weights)
        entropy.append(diagnostics.attention_entropy)
        mean_age.append(diagnostics.mean_attended_age)
    return (
        torch.stack(outputs, dim=1),
        torch.stack(intents, dim=1),
        current,
        torch.stack(weights, dim=1),
        torch.stack(entropy, dim=1),
        torch.stack(mean_age, dim=1),
    )


def test_temporal_sequence_matches_step_loop_from_zero_state():
    from kaggrl.v2_tensorize import (
        ECONOMY_FEATURES,
        EFFECT_FEATURES,
        PREV_ACTION_GLOBAL_FEATURES,
    )

    torch.manual_seed(29)
    core = TemporalCore().eval()
    batch, steps = 2, 5
    fused = torch.randn(batch, steps, 256)
    previous = torch.randn(batch, steps, len(PREV_ACTION_GLOBAL_FEATURES))
    effect = torch.randn(batch, steps, len(EFFECT_FEATURES))
    economy = torch.randn(batch, steps, len(ECONOMY_FEATURES))

    reference = _sequence_reference(
        core, fused, previous, effect, economy
    )
    output, intent, state, diagnostics = core.sequence(
        fused, previous, effect, economy
    )
    assert torch.allclose(output, reference[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(intent, reference[1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(state.h, reference[2].h, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state.c, reference[2].c, atol=1e-5, rtol=1e-5)
    assert torch.allclose(state.memory, reference[2].memory, atol=1e-5, rtol=1e-5)
    assert torch.equal(state.valid_length, reference[2].valid_length)
    assert torch.equal(state.write_pos, reference[2].write_pos)
    assert torch.allclose(
        diagnostics.attention_weights, reference[3], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        diagnostics.attention_entropy, reference[4], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        diagnostics.mean_attended_age, reference[5], atol=1e-5, rtol=1e-5
    )


def test_temporal_sequence_matches_step_loop_with_carried_ring_state():
    from kaggrl.v2_tensorize import (
        ECONOMY_FEATURES,
        EFFECT_FEATURES,
        PREV_ACTION_GLOBAL_FEATURES,
    )

    torch.manual_seed(31)
    core = TemporalCore().eval()
    batch = 2
    state = None
    for _ in range(27):
        fused0 = torch.randn(batch, 256)
        previous0 = torch.randn(batch, len(PREV_ACTION_GLOBAL_FEATURES))
        effect0 = torch.randn(batch, len(EFFECT_FEATURES))
        economy0 = torch.randn(batch, len(ECONOMY_FEATURES))
        _, _, state, _ = core.step(
            fused0, previous0, effect0, economy0, state
        )
    carried = type(state)(
        h=state.h.clone(),
        c=state.c.clone(),
        memory=state.memory.clone(),
        valid_length=state.valid_length.clone(),
        write_pos=state.write_pos.clone(),
    )

    steps = 5
    fused = torch.randn(batch, steps, 256)
    previous = torch.randn(batch, steps, len(PREV_ACTION_GLOBAL_FEATURES))
    effect = torch.randn(batch, steps, len(EFFECT_FEATURES))
    economy = torch.randn(batch, steps, len(ECONOMY_FEATURES))
    reference = _sequence_reference(
        core, fused, previous, effect, economy, carried
    )
    output, intent, final_state, diagnostics = core.sequence(
        fused, previous, effect, economy, state
    )
    assert torch.allclose(output, reference[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(intent, reference[1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        final_state.memory, reference[2].memory, atol=1e-5, rtol=1e-5
    )
    assert torch.equal(final_state.valid_length, reference[2].valid_length)
    assert torch.equal(final_state.write_pos, reference[2].write_pos)
    assert torch.allclose(
        diagnostics.attention_weights, reference[3], atol=1e-5, rtol=1e-5
    )


def test_attention_fp16_masking_and_diagnostics_are_finite():
    torch.manual_seed(17)
    core = TemporalCore(window=4).eval().half()
    state = core.zero_state(2, dtype=torch.float16)
    state.valid_length[1] = 1
    state.write_pos[1] = 1
    state.memory[1, 0, 0] = 1.0
    token = torch.zeros(2, core.attention_dim, dtype=torch.float16)
    context, diagnostics = core._attend(token, state)
    assert context.dtype == torch.float16
    assert torch.isfinite(context).all()
    assert diagnostics.attention_weights.dtype == torch.float32
    assert torch.isfinite(diagnostics.attention_weights).all()
    assert torch.isfinite(diagnostics.attention_entropy).all()
    assert torch.isfinite(diagnostics.mean_attended_age).all()
