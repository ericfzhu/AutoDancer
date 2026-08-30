from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autodancer.constants import ACTION_COUNT
from autodancer.observation import observation_space
from autodancer.training.imitation_sequences import (
    RecurrentDemonstration,
    load_recurrent_demonstrations,
    manifest_path,
    write_recurrent_demonstrations,
)
from autodancer.training.model import START_ACTION


def demonstration(*, trace_id: str = "trace-a", seed: int = 7) -> RecurrentDemonstration:
    length = 3
    observations = {
        name: np.zeros((length, *space.shape), dtype=space.dtype)
        for name, space in observation_space().spaces.items()
    }
    observations["action_mask"].fill(1)
    actions = np.asarray([0, 3, ACTION_COUNT - 1], dtype=np.int64)
    return RecurrentDemonstration(
        trace_id=trace_id,
        seed=seed,
        observations=observations,
        actions=actions,
        previous_actions=np.asarray([START_ACTION, 0, 3], dtype=np.int64),
        previous_rewards=np.asarray([0.0, 0.25, -0.5], dtype=np.float32),
        episode_starts=np.asarray([True, False, False], dtype=np.bool_),
    )


def test_recurrent_demonstration_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sequences.npz"
    written = write_recurrent_demonstrations(
        path,
        [demonstration(), demonstration(trace_id="trace-b", seed=8)],
        provenance={"bank_sha256": "bank", "qualification_sha256": "qualification"},
    )

    manifest, loaded = load_recurrent_demonstrations(path)

    assert manifest == written
    assert manifest_path(path).is_file()
    assert [trace.trace_id for trace in loaded] == ["trace-a", "trace-b"]
    assert [trace.seed for trace in loaded] == [7, 8]
    assert np.array_equal(loaded[0].actions, np.asarray([0, 3, ACTION_COUNT - 1]))
    assert np.array_equal(loaded[0].previous_actions, np.asarray([START_ACTION, 0, 3]))
    assert np.allclose(loaded[0].previous_rewards, [0.0, 0.25, -0.5])


def test_recurrent_demonstration_rejects_noncontiguous_context() -> None:
    value = demonstration()
    value.previous_actions[2] = 1

    with pytest.raises(ValueError, match="not contiguous"):
        value.validate()


def test_recurrent_demonstration_rejects_masked_target() -> None:
    value = demonstration()
    value.observations["action_mask"][1, 3] = 0

    with pytest.raises(ValueError, match="masked action"):
        value.validate()


def test_recurrent_demonstration_rejects_artifact_tampering(tmp_path: Path) -> None:
    path = tmp_path / "sequences.npz"
    write_recurrent_demonstrations(path, [demonstration()], provenance={"source": "test"})
    with path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_recurrent_demonstrations(path)
