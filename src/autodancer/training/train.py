"""End-to-end arbitrary-N live recurrent PPO training CLI."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.model import RecurrentActorCritic
from autodancer.training.ppo import PPOConfig, RecurrentPPO, RolloutBatch


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def default_mod_dir() -> Path | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    return Path(local_app_data) / "NecroDancer" / "mods" / "AutoDancer"


def tensor_observation(
    observation: dict[str, np.ndarray], device: torch.device
) -> dict[str, Tensor]:
    return {key: torch.from_numpy(value).to(device) for key, value in observation.items()}


def replace_observation_rows(
    target: dict[str, np.ndarray],
    indices: list[int],
    replacements: list[dict[str, np.ndarray]],
) -> None:
    for index, replacement in zip(indices, replacements, strict=True):
        for key in target:
            target[key][index] = replacement[key]


class RolloutCollector:
    def __init__(
        self,
        environment: AutoDancerVectorEnv,
        model: RecurrentActorCritic,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.environment = environment
        self.model = model
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.observation, self.infos = environment.reset(self._seeds())
        self.hidden = model.initial_state(environment.num_envs, device=device)
        self.episode_starts = torch.ones(environment.num_envs, dtype=torch.bool, device=device)
        self.episode_returns = np.zeros(environment.num_envs, dtype=np.float64)
        self.episode_events: list[list[dict[str, Any]]] = [
            [] for _ in range(environment.num_envs)
        ]
        self.furthest_zone = np.zeros(environment.num_envs, dtype=np.int32)
        self.furthest_floor = np.zeros(environment.num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []

    def _seeds(self, count: int | None = None) -> list[int]:
        count = self.environment.num_envs if count is None else count
        return self.rng.integers(0, 2**31, size=count, dtype=np.int64).tolist()

    def collect(self, length: int) -> RolloutBatch:
        observations: dict[str, list[Tensor]] = {key: [] for key in self.observation}
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []
        episode_starts: list[Tensor] = []
        values: list[Tensor] = []
        hiddens: list[Tensor] = []
        self.model.eval()
        for _ in range(length):
            for key, value in self.observation.items():
                observations[key].append(torch.from_numpy(value.copy()))
            episode_starts.append(self.episode_starts.detach().cpu())
            hiddens.append(self.hidden.detach().cpu())
            with torch.inference_mode():
                action, log_prob, _, value, next_hidden = self.model.act(
                    tensor_observation(self.observation, self.device), self.hidden
                )
            next_observation, reward, terminated, truncated, infos = self.environment.step(
                action.cpu().numpy()
            )
            done = terminated | truncated
            self.episode_returns += reward
            for index, info in enumerate(infos):
                self.episode_events[index].extend(info.get("raw_events", []))
                self.furthest_zone[index] = max(
                    self.furthest_zone[index], int(info.get("zone") or 0)
                )
                self.furthest_floor[index] = max(
                    self.furthest_floor[index], int(info.get("floor") or 0)
                )
            actions.append(action.cpu())
            log_probs.append(log_prob.cpu())
            rewards.append(torch.from_numpy(reward.copy()))
            dones.append(torch.from_numpy(done.copy()))
            values.append(value.cpu())
            done_indices = np.flatnonzero(done).tolist()
            if done_indices:
                reset_results = self.environment.reset_at(
                    done_indices, self._seeds(len(done_indices))
                )
                replace_observation_rows(
                    next_observation,
                    done_indices,
                    [result[0] for result in reset_results],
                )
                for index in done_indices:
                    info = infos[index]
                    self.completed_episodes.append(
                        {
                            "worker_id": self.environment.worker_ids[index],
                            "return": float(self.episode_returns[index]),
                            "status": info.get("episode_status"),
                            "zone": int(self.furthest_zone[index]),
                            "floor": int(self.furthest_floor[index]),
                            "turns": info.get("turns"),
                            "events": self.episode_events[index],
                        }
                    )
                    self.episode_returns[index] = 0.0
                    self.episode_events[index] = []
                    self.furthest_zone[index] = 0
                    self.furthest_floor[index] = 0
            alive = torch.from_numpy(~done).to(self.device).float().unsqueeze(-1)
            self.hidden = next_hidden * alive
            self.episode_starts = torch.from_numpy(done).to(self.device)
            self.observation = next_observation
            self.infos = infos
        with torch.inference_mode():
            current = tensor_observation(self.observation, self.device)
            _, next_value, _ = self.model.step(
                current["grid"],
                current["player"],
                current["inventory"],
                current["action_mask"],
                self.hidden,
            )
        return RolloutBatch(
            observations={key: torch.stack(value) for key, value in observations.items()},
            actions=torch.stack(actions),
            old_log_probs=torch.stack(log_probs),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            episode_starts=torch.stack(episode_starts),
            values=torch.stack(values),
            hiddens=torch.stack(hiddens),
            next_value=next_value.cpu(),
        )


def episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {"episodes": 0.0}
    events = [event for episode in episodes for event in episode.get("events", [])]
    return {
        "episodes": float(len(episodes)),
        "mean_return": float(np.mean([episode["return"] for episode in episodes])),
        "deaths": float(sum(episode["status"] == "dead" for episode in episodes)),
        "completions": float(sum(episode["status"] == "won" for episode in episodes)),
        "enemy_kills": float(sum(event.get("kind") == "enemy_kill" for event in events)),
        "items_collected": float(sum(event.get("kind") == "item_collected" for event in events)),
        "furthest_zone": float(max(int(episode.get("zone") or 0) for episode in episodes)),
        "furthest_floor": float(max(int(episode.get("floor") or 0) for episode in episodes)),
    }


def evaluate_policy(
    environment: AutoDancerVectorEnv,
    model: RecurrentActorCritic,
    *,
    device: torch.device,
    seed: int,
    steps: int,
) -> dict[str, float]:
    """Evaluate deterministically on every worker, leaving all workers reset."""
    rng = np.random.default_rng(seed)
    observation, _ = environment.reset(
        rng.integers(0, 2**31, size=environment.num_envs, dtype=np.int64).tolist()
    )
    hidden = model.initial_state(environment.num_envs, device=device)
    returns = np.zeros(environment.num_envs, dtype=np.float64)
    completed: list[float] = []
    model.eval()
    for _ in range(steps):
        with torch.inference_mode():
            action, _, _, _, next_hidden = model.act(
                tensor_observation(observation, device), hidden, deterministic=True
            )
        observation, reward, terminated, truncated, _ = environment.step(
            action.cpu().numpy()
        )
        returns += reward
        done = terminated | truncated
        done_indices = np.flatnonzero(done).tolist()
        if done_indices:
            completed.extend(returns[done_indices].tolist())
            resets = environment.reset_at(
                done_indices,
                rng.integers(0, 2**31, size=len(done_indices), dtype=np.int64).tolist(),
            )
            replace_observation_rows(observation, done_indices, [item[0] for item in resets])
            returns[done_indices] = 0.0
        hidden = next_hidden * torch.from_numpy(~done).to(device).float().unsqueeze(-1)
    environment.reset(
        rng.integers(0, 2**31, size=environment.num_envs, dtype=np.int64).tolist()
    )
    scores = completed if completed else returns.tolist()
    return {
        "evaluation_episodes": float(len(completed)),
        "evaluation_mean_return": float(np.mean(scores)),
    }


def train(arguments: argparse.Namespace) -> None:
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = resolve_device(arguments.device)
    ppo_config = PPOConfig(
        rollout_length=arguments.rollout_length,
        sequence_length=arguments.sequence_length,
    )
    supervisor_config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=arguments.num_instances,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        max_turns=arguments.max_turns,
    )
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.run_dir / "metrics.jsonl"
    with AutoDancerSupervisor(supervisor_config) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            model = RecurrentActorCritic()
            algorithm = RecurrentPPO(model, ppo_config, device=device)
            if arguments.resume:
                algorithm.load(arguments.resume)
            collector = RolloutCollector(
                environment, algorithm.model, device=device, seed=arguments.seed
            )
            started = time.monotonic()
            metrics: dict[str, Any] = {
                "global_step": algorithm.global_step,
                "updates": algorithm.updates,
            }
            next_evaluation = (
                algorithm.global_step + arguments.evaluation_interval
                if arguments.evaluation_interval > 0
                else None
            )
            while algorithm.global_step < arguments.total_steps:
                rollout = collector.collect(ppo_config.rollout_length)
                update_metrics = algorithm.update(rollout)
                elapsed = max(time.monotonic() - started, 1.0e-6)
                metrics = {
                    "global_step": algorithm.global_step,
                    "updates": algorithm.updates,
                    "steps_per_second": algorithm.global_step / elapsed,
                    **update_metrics,
                    **episode_metrics(collector.completed_episodes),
                    "worker_restarts": sum(
                        handle.restart_count for handle in supervisor.workers.values()
                    ),
                }
                if next_evaluation is not None and algorithm.global_step >= next_evaluation:
                    metrics.update(
                        evaluate_policy(
                            environment,
                            algorithm.model,
                            device=device,
                            seed=arguments.seed + algorithm.updates,
                            steps=arguments.evaluation_steps,
                        )
                    )
                    collector = RolloutCollector(
                        environment,
                        algorithm.model,
                        device=device,
                        seed=arguments.seed + algorithm.global_step,
                    )
                    next_evaluation += arguments.evaluation_interval
                collector.completed_episodes.clear()
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                print(json.dumps(metrics, sort_keys=True))
                if algorithm.global_step % arguments.checkpoint_interval < (
                    ppo_config.rollout_length * arguments.num_instances
                ):
                    algorithm.save(arguments.run_dir / "latest.pt", metrics=metrics)
            algorithm.save(arguments.run_dir / "final.pt", metrics=metrics)
            (arguments.run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "ppo": asdict(ppo_config),
                        "supervisor": {
                            "num_instances": arguments.num_instances,
                            "game_dir": str(arguments.game_dir),
                            "mod_dir": str(arguments.mod_dir),
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            environment.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train recurrent PPO in live NecroDancer")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--evaluation-interval", type=int, default=50000)
    parser.add_argument("--evaluation-steps", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--turn-timeout", type=float, default=10.0)
    parser.add_argument("--max-turns", type=int, default=10000)
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    if arguments.total_steps <= 0 or arguments.num_instances <= 0:
        parser.error("--total-steps and --num-instances must be positive")
    train(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
