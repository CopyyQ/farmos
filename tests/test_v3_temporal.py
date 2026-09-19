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
