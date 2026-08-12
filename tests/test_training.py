from __future__ import annotations

import numpy as np
import torch

from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.training.curriculum import (
    SEED_RANGES,
    AdaptiveTaskMixer,
    CurriculumEnv,
    fixed_seed,
)
from autodancer.training.model import AutoDancerEncoder
from autodancer.training.train import parse_arguments


def test_seed_splits_do_not_overlap() -> None:
    assert set(SEED_RANGES["validation"]).isdisjoint(SEED_RANGES["test"])
    assert fixed_seed("train", "worker", 10) in SEED_RANGES["train"]
    assert fixed_seed("validation", "worker", 10) in SEED_RANGES["validation"]


def test_curriculum_has_smooth_nonzero_probabilities() -> None:
    mixer = AdaptiveTaskMixer()
    initial = mixer.probabilities()
    assert np.all(initial > 0)
    mixer.update("navigation", True)
    updated = mixer.probabilities()
    assert np.all(updated > 0)
    np.testing.assert_allclose(updated.sum(), 1.0)


def test_curriculum_environment_reports_selected_task() -> None:
    environment = CurriculumEnv(stream="test-stream")
    observation, info = environment.reset()
    assert environment.observation_space.contains(observation)
    assert info["curriculum_task"] in environment.mixer.success_rates


def test_symbolic_encoder_output_shape() -> None:
    cfg = parse_arguments(["--experiment=encoder_test"])
    environment = AutoDancerSimEnv(task="navigation")
    observation, _ = environment.reset(seed=1)
    encoder = AutoDancerEncoder(cfg, environment.observation_space)
    batch = {
        key: torch.from_numpy(np.stack((value, value)))
        for key, value in observation.items()
    }
    assert encoder(batch).shape == (2, 256)


def test_training_defaults_request_64_environments_and_gru() -> None:
    cfg = parse_arguments(["--experiment=defaults_test"])
    assert cfg.num_workers * cfg.num_envs_per_worker == 64
    assert cfg.use_rnn
    assert cfg.rnn_type == "gru"
    assert cfg.rnn_size == 256
