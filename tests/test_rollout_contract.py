import numpy as np

from kaggrl.actions import ActionCodec
from kaggrl.rollout import EpisodeBuffer, potential, transition_reward


def _obs(player, a, b):
    return {"player": player, "farms": [{"money": a}, {"money": b}], "step": 10}


def test_potential_and_reward_are_seat_symmetric():
    a = _obs(0, 12000, 9000)
    b = _obs(1, 12000, 9000)
    assert np.isclose(potential(a), -potential(b))
    a2 = _obs(0, 12500, 9000)
    b2 = _obs(1, 12500, 9000)
    ra = transition_reward(a, a2, 0.0, gamma=0.995)
    rb = transition_reward(b, b2, 0.0, gamma=0.995)
    assert np.isclose(ra, -rb, atol=1e-6)


def test_terminal_result_is_primary_reward_component():
    obs = _obs(0, 10000, 10000)
    assert transition_reward(obs, obs, 1.0, gamma=1.0) == 1.0
    assert transition_reward(obs, obs, -1.0, gamma=1.0) == -1.0

def test_episode_buffer_saves_exact_rollout_dtypes(tmp_path):
    codec = ActionCodec()
    buf = EpisodeBuffer(model_sha256="abc", opponent="v17", seat=0, seed=301)
    buf.append(
        step=0,
        obs=np.zeros(1024, np.float32),
        action=np.zeros(codec.width, np.int16),
        mask=np.zeros(codec.width, np.uint8),
        old_logp=-1.25,
        old_value=0.5,
        reward=0.01,
        done=False,
    )
    path = tmp_path / "episode.npz"
    buf.save_npz(path)
    data = np.load(path, allow_pickle=False)
    assert data["obs_f16"].shape == (1, 1024)
    assert data["action_i16"].shape == (1, codec.width)
    assert data["mask_u8"].dtype == np.uint8
    assert data["old_logp_f32"].dtype == np.float32
    assert data["old_value_f32"].dtype == np.float32
    assert data["reward_f32"].dtype == np.float32
    assert data["done_u8"].dtype == np.uint8
    assert data["step_i16"].tolist() == [0]
    assert str(data["model_sha256"].item()) == "abc"