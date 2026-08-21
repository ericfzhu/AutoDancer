"""Reproducible live-game baseline evaluation for Bard policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.constants import PlayerFeature
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.protocol import SUPPORTED_GAME_VERSION, SUPPORTED_STEAM_BUILD
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.model import START_ACTION, ModelConfig, RecurrentActorCritic
from autodancer.training.train import default_mod_dir, replace_observation_rows, resolve_device


@dataclass(slots=True)
class EpisodeAccumulator:
    seed: int
    worker_id: str
    run_id: str
    episode_return: float = 0.0
    turns: int = 0
    furthest_zone: int = 0
    furthest_floor: int = 0
    max_gold: int = 0
    enemy_kills: int = 0
    item_pickups: int = 0
    item_value: int = 0
    enemy_damage: int = 0
    player_damage: int = 0

    def observe(
        self,
        observation: dict[str, np.ndarray],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        self.episode_return += float(reward)
        self.turns += 1
        self.furthest_zone = max(self.furthest_zone, int(info.get("zone") or 0))
        self.furthest_floor = max(self.furthest_floor, int(info.get("floor") or 0))
        self.max_gold = max(
            self.max_gold,
            int(observation["player"][PlayerFeature.GOLD]),
        )
        for event in info.get("raw_events", []):
            kind = str(event.get("kind", ""))
            amount = int(event.get("amount", 0) or 0)
            if kind == "enemy_kill":
                self.enemy_kills += max(amount, 1)
            elif kind == "item_collected":
                self.item_pickups += 1
                self.item_value += amount
            elif kind == "enemy_damage":
                self.enemy_damage += amount
            elif kind == "player_damage":
                self.player_damage += amount

    def finish(self, status: str) -> dict[str, Any]:
        return {**asdict(self), "status": status}


def masked_random_actions(action_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions: list[int] = []
    for mask in action_mask:
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError("Live observation contains no legal action")
        actions.append(int(rng.choice(legal)))
    return np.asarray(actions, dtype=np.int64)


def summarize_episodes(episodes: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if not episodes:
        raise ValueError("At least one episode is required")
    count = len(episodes)
    progress = [
        max(int(episode["furthest_zone"]) - 1, 0) * 4 + int(episode["furthest_floor"])
        for episode in episodes
    ]
    return {
        "policy": policy,
        "episodes": count,
        "completion_rate": sum(episode["status"] == "won" for episode in episodes) / count,
        "death_rate": sum(episode["status"] == "dead" for episode in episodes) / count,
        "abort_rate": sum(episode["status"] == "aborted" for episode in episodes) / count,
        "step_limit_rate": sum(episode["status"] == "step_limit" for episode in episodes) / count,
        "mean_return": float(np.mean([episode["episode_return"] for episode in episodes])),
        "mean_turns": float(np.mean([episode["turns"] for episode in episodes])),
        "mean_progress": float(np.mean(progress)),
        "furthest_zone": max(int(episode["furthest_zone"]) for episode in episodes),
        "furthest_floor": max(progress),
        "mean_max_gold": float(np.mean([episode["max_gold"] for episode in episodes])),
        "enemy_kills": sum(int(episode["enemy_kills"]) for episode in episodes),
        "item_pickups": sum(int(episode["item_pickups"]) for episode in episodes),
        "enemy_damage": sum(int(episode["enemy_damage"]) for episode in episodes),
        "player_damage": sum(int(episode["player_damage"]) for episode in episodes),
        "results": episodes,
    }


def compare_summaries(reference: dict[str, Any], trained: dict[str, Any]) -> dict[str, float]:
    fields = (
        "completion_rate",
        "death_rate",
        "mean_return",
        "mean_turns",
        "mean_progress",
        "mean_max_gold",
        "enemy_kills",
        "item_pickups",
        "enemy_damage",
        "player_damage",
    )
    return {f"{field}_delta": float(trained[field]) - float(reference[field]) for field in fields}


def _model_actions(
    model: RecurrentActorCritic,
    observation: dict[str, np.ndarray],
    hidden: Tensor,
    device: torch.device,
    previous_actions: np.ndarray,
    previous_rewards: np.ndarray,
) -> tuple[np.ndarray, Tensor]:
    tensors = {key: torch.from_numpy(value).to(device) for key, value in observation.items()}
    tensors["previous_action"] = torch.from_numpy(previous_actions).to(device)
    tensors["previous_reward"] = torch.from_numpy(previous_rewards).to(device)
    with torch.inference_mode():
        actions, _, _, _, next_hidden = model.act(tensors, hidden, deterministic=True)
    return actions.cpu().numpy(), next_hidden


def zero_hidden_rows(hidden: Tensor, indices: list[int]) -> Tensor:
    """Reset selected recurrent slots without mutating inference tensors."""
    keep = torch.ones(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
    keep[indices] = 0
    return hidden * keep.reshape(hidden.shape[0], *([1] * (hidden.ndim - 1)))


def evaluate_live_policy(
    environment: AutoDancerVectorEnv,
    *,
    seeds: list[int],
    max_steps: int,
    policy_seed: int,
    device: torch.device,
    model: RecurrentActorCritic | None = None,
) -> list[dict[str, Any]]:
    if len(seeds) % environment.num_envs:
        raise ValueError("The evaluation seed count must be divisible by num_instances")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    rng = np.random.default_rng(policy_seed)
    results: list[dict[str, Any]] = []
    parking_seed = 2_000_000_000

    for start in range(0, len(seeds), environment.num_envs):
        wave_seeds = seeds[start : start + environment.num_envs]
        observation, infos = environment.reset(wave_seeds)
        accumulators: list[EpisodeAccumulator | None] = [
            EpisodeAccumulator(
                seed=int(info.get("seed") if info.get("seed") is not None else seed),
                worker_id=environment.worker_ids[index],
                run_id=str(info.get("run_id", "")),
                furthest_zone=int(info.get("zone") or 0),
                furthest_floor=int(info.get("floor") or 0),
                max_gold=int(observation["player"][index, PlayerFeature.GOLD]),
            )
            for index, (seed, info) in enumerate(zip(wave_seeds, infos, strict=True))
        ]
        hidden = (
            model.initial_state(environment.num_envs, device=device) if model is not None else None
        )
        previous_actions = np.full(environment.num_envs, START_ACTION, dtype=np.int64)
        previous_rewards = np.zeros(environment.num_envs, dtype=np.float32)

        while any(accumulator is not None for accumulator in accumulators):
            if model is None:
                actions = masked_random_actions(observation["action_mask"], rng)
                next_hidden = None
            else:
                assert hidden is not None
                actions, next_hidden = _model_actions(
                    model, observation, hidden, device, previous_actions, previous_rewards
                )
            next_observation, rewards, terminated, truncated, step_infos = environment.step(actions)
            reset_indices: list[int] = []
            for index, accumulator in enumerate(accumulators):
                done = bool(terminated[index] or truncated[index])
                if accumulator is not None:
                    accumulator.observe(
                        {key: value[index] for key, value in next_observation.items()},
                        float(rewards[index]),
                        step_infos[index],
                    )
                    reached_limit = accumulator.turns >= max_steps and not done
                    if done or reached_limit:
                        status = (
                            str(step_infos[index].get("episode_status", "aborted"))
                            if done
                            else "step_limit"
                        )
                        results.append(accumulator.finish(status))
                        accumulators[index] = None
                        reset_indices.append(index)
                elif done:
                    reset_indices.append(index)

            if not any(accumulator is not None for accumulator in accumulators):
                break
            if reset_indices:
                reset_seeds = list(range(parking_seed, parking_seed + len(reset_indices)))
                parking_seed += len(reset_indices)
                reset_results = environment.reset_at(reset_indices, reset_seeds)
                replace_observation_rows(
                    next_observation,
                    reset_indices,
                    [result[0] for result in reset_results],
                )
                if next_hidden is not None:
                    next_hidden = zero_hidden_rows(next_hidden, reset_indices)
            observation = next_observation
            hidden = next_hidden
            previous_actions = actions.astype(np.int64, copy=True)
            previous_rewards = rewards.astype(np.float32, copy=True)
            previous_actions[reset_indices] = START_ACTION
            previous_rewards[reset_indices] = 0.0
    return results


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_baseline(arguments: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(arguments.device)
    payload = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
    expected = RecurrentActorCritic(ModelConfig())
    if payload.get("architecture") != expected.architecture_spec():
        raise ValueError("Checkpoint model architecture is incompatible with the schema-5 policy")
    model = expected.to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=arguments.num_instances,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(arguments.max_steps + 1, 2),
    )
    with AutoDancerSupervisor(config) as supervisor:
        environment = AutoDancerVectorEnv(supervisor)
        try:
            random_results = evaluate_live_policy(
                environment,
                seeds=arguments.seeds,
                max_steps=arguments.max_steps,
                policy_seed=arguments.policy_seed,
                device=device,
            )
            trained_results = evaluate_live_policy(
                environment,
                seeds=arguments.seeds,
                max_steps=arguments.max_steps,
                policy_seed=arguments.policy_seed,
                device=device,
                model=model,
            )
            restarts = sum(handle.restart_count for handle in supervisor.workers.values())
        finally:
            environment.close()

    reference = summarize_episodes(random_results, "masked_random")
    trained = summarize_episodes(trained_results, "checkpoint_deterministic")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "game_version": SUPPORTED_GAME_VERSION,
        "steam_build": SUPPORTED_STEAM_BUILD,
        "character": "Bard",
        "mode": "AllZonesSeeded",
        "checkpoint": str(arguments.checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_hash(arguments.checkpoint),
        "checkpoint_global_step": int(payload.get("global_step", 0)),
        "checkpoint_updates": int(payload.get("updates", 0)),
        "num_instances": arguments.num_instances,
        "max_steps_per_episode": arguments.max_steps,
        "seeds": arguments.seeds,
        "policy_seed": arguments.policy_seed,
        "worker_restarts": restarts,
        "reference": reference,
        "trained": trained,
        "trained_minus_reference": compare_summaries(reference, trained),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(arguments.output)
    return report


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 or seed >= 2**31 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be integers in [0, 2^31)")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("evaluation seeds must be unique")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a trained Bard checkpoint with a masked-random live baseline"
    )
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-instances", type=int, default=4)
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--policy-seed", type=int, default=8675309)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--turn-timeout", type=float, default=10.0)
    parser.add_argument("--reset-timeout", type=float, default=30.0)
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    if arguments.num_instances <= 0 or arguments.max_steps <= 0:
        parser.error("--num-instances and --max-steps must be positive")
    if len(arguments.seeds) % arguments.num_instances:
        parser.error("--seeds count must be divisible by --num-instances")
    report = run_baseline(arguments)
    summary = {
        "checkpoint_global_step": report["checkpoint_global_step"],
        "reference": {key: value for key, value in report["reference"].items() if key != "results"},
        "trained": {key: value for key, value in report["trained"].items() if key != "results"},
        "trained_minus_reference": report["trained_minus_reference"],
        "output": str(arguments.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
