from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from autodancer.live.run import RecurrentTorchPolicy


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("initial_rnn_state", torch.zeros(1, 3))

    def forward(
        self,
        grid: Tensor,
        player: Tensor,
        inventory: Tensor,
        action_mask: Tensor,
        rnn_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del grid, player, inventory, action_mask
        logits = torch.arange(11, dtype=torch.float32).unsqueeze(0)
        return logits, rnn_state + 1


def observation() -> dict[str, np.ndarray]:
    mask = np.zeros(11, dtype=np.int8)
    mask[0] = 1
    mask[3] = 1
    return {
        "grid": np.zeros((21, 21, 11), dtype=np.int16),
        "player": np.zeros(16, dtype=np.int32),
        "inventory": np.zeros((8, 4), dtype=np.int16),
        "action_mask": mask,
    }


def test_recurrent_policy_applies_mask_and_resets_state() -> None:
    policy = RecurrentTorchPolicy(FakePolicy(), device="cpu")
    assert policy(observation()) == 3
    torch.testing.assert_close(policy.rnn_state, torch.ones(1, 3))
    policy.reset()
    torch.testing.assert_close(policy.rnn_state, torch.zeros(1, 3))
