from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from autodancer.live.run import RecurrentTorchPolicy
from autodancer.training.export import RecurrentPolicyExport


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
        "grid": np.zeros((21, 21, 7), dtype=np.int16),
        "player": np.zeros(16, dtype=np.int32),
        "inventory": np.zeros((8, 3), dtype=np.int16),
        "action_mask": mask,
    }


def test_recurrent_policy_applies_mask_and_resets_state() -> None:
    policy = RecurrentTorchPolicy(FakePolicy(), device="cpu")
    assert policy(observation()) == 3
    torch.testing.assert_close(policy.rnn_state, torch.ones(1, 3))
    policy.reset()
    torch.testing.assert_close(policy.rnn_state, torch.zeros(1, 3))


class FakeActorCritic(nn.Module):
    def forward_head(self, observations: dict[str, Tensor]) -> Tensor:
        return observations["player"].float()[:, :3]

    def forward_core(self, head: Tensor, rnn_state: Tensor) -> tuple[Tensor, Tensor]:
        return head, rnn_state + 2

    def forward_tail(
        self, core: Tensor, values_only: bool, sample_actions: bool
    ) -> dict[str, Tensor]:
        del values_only, sample_actions
        logits = torch.zeros(core.shape[0], 11)
        logits[:, 2] = core[:, 0] + 1
        return {"action_logits": logits}


def test_export_wrapper_exposes_logits_and_explicit_recurrent_state() -> None:
    wrapper = RecurrentPolicyExport(FakeActorCritic(), rnn_size=3)
    obs = observation()
    logits, state = wrapper(
        torch.from_numpy(obs["grid"]).unsqueeze(0),
        torch.from_numpy(obs["player"]).unsqueeze(0),
        torch.from_numpy(obs["inventory"]).unsqueeze(0),
        torch.from_numpy(obs["action_mask"]).unsqueeze(0),
        torch.zeros(1, 3),
    )
    assert logits.shape == (1, 11)
    torch.testing.assert_close(state, torch.full((1, 3), 2.0))
