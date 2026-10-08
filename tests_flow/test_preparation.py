import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from flow_jepa import environment
from flow_jepa.common import digest
from flow_jepa.data import encode_episode, encode_goal_batch, preparation_batches, summarize_episode, write_episode, _encode_list
from flow_jepa.vision import history_clip


@pytest.mark.parametrize("successful", [True, False])
@pytest.mark.parametrize("mode", ["expert", "random"])
def test_goal_replay_preserves_full_trace_and_last_success_image(monkeypatch, successful, mode):
    class Env:
        def reset(self):
            self.step_index, self.state = 0, np.zeros(4, np.float32)
            return self.state.copy(), {}
        def render(self):
            return np.full((4, 4, 3), self.step_index, np.uint8)
        def step(self, action):
            self.state += action
            self.step_index += 1
            return self.state.copy(), float(self.state.sum()), self.step_index == 7, False, {
                "success": successful and self.step_index in (2, 4)}
        def close(self):
            pass
    class Policy:
        def get_action(self, obs):
            return np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    monkeypatch.setattr(environment, "make_env", lambda *args: Env())
    monkeypatch.setitem(sys.modules, "metaworld.policies", SimpleNamespace(ENV_POLICY_MAP={"reach-v3": Policy}))
    c = {"primitive_budget": 10}
    old = environment.collect_episode("reach", 123, c, mode)
    new = environment.collect_goal_episode("reach", 123, c, mode)
    for name in ("actions", "rewards", "successes", "goal_index", "expert_success"):
        np.testing.assert_array_equal(old[name], new[name])
    np.testing.assert_array_equal(new["initial_rgb"], old["frames"][0])
    np.testing.assert_array_equal(new["goal_rgb"], old["frames"][old["goal_index"]])
    assert new["goal_index"] == (4 if successful else 7)


def test_streaming_encode_write_and_summary_preserve_arrays(small_config, tmp_path):
    c = small_config
    c["data"].update(history_frames=4, action_repeat=2)
    c["encoder"]["batch_size"] = 2
    class Encoder:
        def __call__(self, clips):
            values = torch.from_numpy(clips.copy()).float().mean((1, 2, 3, 4))
            return values[:, None, None].expand(-1, 2, 8).clone()
    encoder = Encoder()
    frames = np.random.default_rng(42).integers(0, 255, (11, 5, 5, 3), dtype=np.uint8)
    episode = dict(frames=frames, actions=np.arange(40, dtype=np.float32).reshape(10, 4),
                   successes=np.array([False] * 8 + [True, False]), goal_index=9, expert_success=True)
    row = dict(id="train/reach/00000", task="reach", split="train", seed=1, mode="expert", index=0)
    indexes = list(range(0, len(frames), 2))
    z = _encode_list(encoder, [history_clip(frames, i, 4) for i in indexes], 2)
    gz = _encode_list(encoder, [np.repeat(frames[i][None], 4, axis=0) for i in indexes], 2)
    arrays = encode_episode(episode, row, c, encoder)
    np.testing.assert_array_equal(z, arrays["z"])
    np.testing.assert_array_equal(gz, arrays["image_goals"])
    write_episode(tmp_path, row, episode, arrays, c, "fixture")
    path = tmp_path / "episodes/train/reach/00000.h5"
    with h5py.File(path) as f:
        assert f.attrs["protocol"] == digest(c)
        for key, value in arrays.items():
            np.testing.assert_array_equal(f[key][:], value)
    assert not path.with_suffix(".partial").exists()
    item, total, squared, count = summarize_episode(tmp_path, row, c)
    reference = z.astype(np.float64).reshape(-1, 8)
    np.testing.assert_array_equal(total, reference.sum(0))
    np.testing.assert_array_equal(squared, np.square(reference).sum(0))
    assert count == len(reference) and item["first_success_action"] == 8


def test_goal_batches_flush_tail_and_preserve_individual_outputs(small_config):
    c = small_config
    class Encoder:
        def __call__(self, clips):
            return torch.from_numpy(clips.copy()).float().mean((1, 2, 3, 4))[:, None, None].expand(-1, 2, 8)
    items = []
    for i in range(5):
        e = dict(initial_rgb=np.full((4, 4, 3), i, np.uint8),
                 goal_rgb=np.full((4, 4, 3), i + 10, np.uint8),
                 successes=np.array([False, True]))
        items.append((dict(split="test", id=str(i)), e, 0.0))
    batches = list(preparation_batches(iter(items), 3))
    assert [len(b) for b in batches] == [3, 2]
    encoded = [arrays for batch in batches for arrays in encode_goal_batch(batch, c, Encoder())]
    for item, arrays in zip(items, encoded):
        reference = encode_goal_batch([item], c, Encoder())[0]
        for key in arrays:
            np.testing.assert_array_equal(reference[key], arrays[key])
