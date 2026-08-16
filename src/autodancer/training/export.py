"""Export a Sample Factory GRU checkpoint for dependency-light live inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from autodancer.envs.sim import AutoDancerSimEnv
from autodancer.training.train import parse_arguments, register_components


class RecurrentPolicyExport(nn.Module):
    """Expose logits and explicit recurrent state without Sample Factory sampling."""

    def __init__(self, actor_critic: nn.Module, rnn_size: int) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.register_buffer("initial_rnn_state", torch.zeros(1, rnn_size))

    def forward(
        self,
        grid: Tensor,
        player: Tensor,
        inventory: Tensor,
        action_mask: Tensor,
        rnn_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        observations = {
            "grid": grid,
            "player": player,
            "inventory": inventory,
            "action_mask": action_mask,
        }
        rnn_states = rnn_states + self.initial_rnn_state.to(rnn_states) * 0.0
        head = self.actor_critic.forward_head(observations)
        core, new_rnn_states = self.actor_critic.forward_core(head, rnn_states)
        result = self.actor_critic.forward_tail(
            core, values_only=False, sample_actions=False
        )
        return result["action_logits"], new_rnn_states


def _load_actor_critic(
    experiment: str,
    train_dir: Path,
    checkpoint_kind: str,
    policy_index: int,
):
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.model.model_utils import get_rnn_size

    register_components()
    cfg = parse_arguments(
        [
            f"--experiment={experiment}",
            f"--train_dir={train_dir}",
            "--device=cpu",
        ]
    )
    cfg = load_from_checkpoint(cfg)
    cfg.device = "cpu"
    environment = AutoDancerSimEnv(task="all_zones")
    actor_critic = create_actor_critic(
        cfg, environment.observation_space, environment.action_space
    )
    actor_critic.eval()
    actor_critic.model_to_device(torch.device("cpu"))

    prefix = {"latest": "checkpoint", "best": "best"}[checkpoint_kind]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, policy_index), f"{prefix}_*"
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No {checkpoint_kind} checkpoint was found for experiment {experiment!r}"
        )
    checkpoint = Learner.load_checkpoint(checkpoints, torch.device("cpu"))
    actor_critic.load_state_dict(checkpoint["model"])
    return actor_critic, cfg, get_rnn_size(cfg), Path(checkpoints[-1])


def _example_inputs(rnn_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    environment = AutoDancerSimEnv(task="all_zones")
    observation, _ = environment.reset(seed=0)
    return (
        torch.from_numpy(observation["grid"]).unsqueeze(0),
        torch.from_numpy(observation["player"]).unsqueeze(0),
        torch.from_numpy(observation["inventory"]).unsqueeze(0),
        torch.from_numpy(observation["action_mask"]).unsqueeze(0),
        torch.zeros(1, rnn_size),
    )


def export_policy(
    *,
    experiment: str,
    train_dir: Path,
    checkpoint_kind: str,
    policy_index: int,
    output: Path,
) -> dict[str, Any]:
    if output.suffix.lower() not in {".pt", ".ts"}:
        raise ValueError("TorchScript policy exports must use a .pt or .ts extension")
    actor_critic, cfg, rnn_size, checkpoint_path = _load_actor_critic(
        experiment, train_dir, checkpoint_kind, policy_index
    )
    wrapper = RecurrentPolicyExport(actor_critic, rnn_size).eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            _example_inputs(rnn_size),
            strict=False,
            check_trace=False,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, output)
    metadata = {
        "format": "torchscript",
        "experiment": experiment,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint": str(checkpoint_path),
        "policy_index": policy_index,
        "rnn_size": rnn_size,
        "action_count": 11,
        "use_rnn": bool(cfg.use_rnn),
        "rnn_type": str(cfg.rnn_type),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a recurrent AutoDancer policy")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-kind", choices=("latest", "best"), default="best")
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    metadata = export_policy(
        experiment=arguments.experiment,
        train_dir=arguments.train_dir,
        checkpoint_kind=arguments.checkpoint_kind,
        policy_index=arguments.policy_index,
        output=arguments.output,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
