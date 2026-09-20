from types import SimpleNamespace

import torch

from tools.audit_saved_batch_stability import observed_terminal_returns


def test_observed_returns_do_not_cross_truncation_or_unfinished_suffix():
    rollout = SimpleNamespace(
        actions=torch.zeros((6, 1)),
        rewards=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        dones=torch.tensor([[False], [True], [False], [True], [False], [False]]),
        terminations=torch.tensor([[False], [False], [False], [True], [False], [False]]),
    )
    values, mask = observed_terminal_returns(rollout, 0.5)
    assert mask[:, 0].tolist() == [False, False, True, True, False, False]
    torch.testing.assert_close(values[:, 0], torch.tensor([0.0, 0.0, 5.0, 4.0, 0.0, 0.0]))


def test_observed_returns_separate_true_terminal_episodes_and_workers():
    rollout = SimpleNamespace(
        actions=torch.zeros((3, 2)),
        rewards=torch.tensor([[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]]),
        dones=torch.tensor([[True, False], [False, False], [True, False]]),
        terminations=torch.tensor([[True, False], [False, False], [True, False]]),
    )
    values, mask = observed_terminal_returns(rollout, 0.5)
    torch.testing.assert_close(values[:, 0], torch.tensor([1.0, 3.5, 3.0]))
    assert mask[:, 0].all() and not mask[:, 1].any()
