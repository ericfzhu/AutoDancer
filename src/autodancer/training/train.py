"""End-to-end arbitrary-N live recurrent PPO training CLI."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.rewards import load_reward_config
from autodancer.training.action_contract import ACTION_CONTRACTS
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.dashboard import DashboardServer, DashboardState
from autodancer.training.model import (
    START_ACTION,
    AdapterActorCritic,
    ModelConfig,
    PolicyModel,
    ProjectedAdapterActorCritic,
    RecurrentActorCritic,
)
from autodancer.training.ppo import PPOConfig, RecurrentPPO, RolloutBatch

TelemetryCallback = Callable[
    [dict[str, np.ndarray], list[dict[str, Any]], np.ndarray | None, np.ndarray | None],
    None,
]


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


def archive_checkpoint(source: Path, target: Path) -> None:
    """Archive an atomic checkpoint without serializing the model twice."""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)


class RolloutCollector:
    def __init__(
        self,
        environment: AutoDancerVectorEnv,
        model: PolicyModel,
        *,
        device: torch.device,
        seed: int,
        telemetry_callback: TelemetryCallback | None = None,
    ) -> None:
        self.environment = environment
        self.model = model
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.observation, self.infos = environment.reset(self._seeds())
        self.hidden = model.initial_state(environment.num_envs, device=device)
        self.previous_actions = np.full(environment.num_envs, START_ACTION, dtype=np.int64)
        self.previous_rewards = np.zeros(environment.num_envs, dtype=np.float32)
        self.episode_starts = torch.ones(environment.num_envs, dtype=torch.bool, device=device)
        self.episode_returns = np.zeros(environment.num_envs, dtype=np.float64)
        self.episode_events: list[list[dict[str, Any]]] = [[] for _ in range(environment.num_envs)]
        self.furthest_zone = np.zeros(environment.num_envs, dtype=np.int32)
        self.furthest_floor = np.zeros(environment.num_envs, dtype=np.int32)
        self.completed_episodes: list[dict[str, Any]] = []
        self.last_reward_components: dict[str, float] = {}
        self.telemetry_callback = telemetry_callback
        if self.telemetry_callback is not None:
            self.telemetry_callback(self.observation, self.infos, None, None)

    def _seeds(self, count: int | None = None) -> list[int]:
        count = self.environment.num_envs if count is None else count
        return self.rng.integers(0, 2**31, size=count, dtype=np.int64).tolist()

    def collect(self, length: int) -> RolloutBatch:
        reward_components: dict[str, float] = {}
        observations: dict[str, list[Tensor]] = {
            **{key: [] for key in self.observation},
            "previous_action": [],
            "previous_reward": [],
        }
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
            observations["previous_action"].append(torch.from_numpy(self.previous_actions.copy()))
            observations["previous_reward"].append(torch.from_numpy(self.previous_rewards.copy()))
            episode_starts.append(self.episode_starts.detach().cpu())
            hiddens.append(self.hidden.detach().cpu())
            with torch.inference_mode():
                policy_observation = tensor_observation(self.observation, self.device)
                policy_observation["previous_action"] = torch.from_numpy(self.previous_actions).to(
                    self.device
                )
                policy_observation["previous_reward"] = torch.from_numpy(self.previous_rewards).to(
                    self.device
                )
                action, log_prob, _, value, next_hidden = self.model.act(
                    policy_observation, self.hidden
                )
            next_observation, reward, terminated, truncated, infos = self.environment.step(
                action.cpu().numpy()
            )
            if self.telemetry_callback is not None:
                self.telemetry_callback(
                    next_observation,
                    infos,
                    action.cpu().numpy(),
                    reward,
                )
            done = terminated | truncated
            self.episode_returns += reward
            for index, info in enumerate(infos):
                for name, component_value in info.get("reward_components", {}).items():
                    reward_components[name] = reward_components.get(name, 0.0) + float(
                        component_value
                    )
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
            self.previous_actions = action.cpu().numpy().astype(np.int64, copy=True)
            self.previous_rewards = reward.astype(np.float32, copy=True)
            self.previous_actions[done] = START_ACTION
            self.previous_rewards[done] = 0.0
            alive = torch.from_numpy(~done).to(self.device).float().reshape(-1, 1, 1)
            self.hidden = next_hidden * alive
            self.episode_starts = torch.from_numpy(done).to(self.device)
            self.observation = next_observation
            self.infos = infos
        self.last_reward_components = reward_components
        with torch.inference_mode():
            current = tensor_observation(self.observation, self.device)
            current["previous_action"] = torch.from_numpy(self.previous_actions).to(self.device)
            current["previous_reward"] = torch.from_numpy(self.previous_rewards).to(self.device)
            _, next_value, _ = self.model.step(current, self.hidden)
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
        "mean_extrinsic_return": float(
            np.mean([episode.get("extrinsic_return", 0.0) for episode in episodes])
        ),
        "mean_shaping_return": float(
            np.mean([episode.get("shaping_return", 0.0) for episode in episodes])
        ),
        "deaths": float(sum(episode["status"] == "dead" for episode in episodes)),
        "completions": float(sum(episode["status"] == "won" for episode in episodes)),
        "enemy_kills": float(sum(event.get("kind") == "enemy_kill" for event in events)),
        "items_collected": float(sum(event.get("kind") == "item_collected" for event in events)),
        "furthest_zone": float(max(int(episode.get("zone") or 0) for episode in episodes)),
        "furthest_floor": float(max(int(episode.get("floor") or 0) for episode in episodes)),
    }


def evaluate_policy(
    environment: AutoDancerVectorEnv,
    model: PolicyModel,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    action_contract: str = "current",
) -> dict[str, float]:
    """Evaluate deterministically on every worker, leaving all workers reset."""
    from autodancer.training.baseline import _evaluate_deterministic_async

    rng = np.random.default_rng(seed)
    episodes = _evaluate_deterministic_async(
        environment,
        model,
        seeds=rng.integers(
            0, 2**31, size=environment.num_envs, dtype=np.int64
        ).tolist(),
        max_steps=steps,
        device=device,
        dashboard_state=None,
        action_contract=action_contract,
    )
    scores = [float(episode["episode_return"]) for episode in episodes]
    return {
        "evaluation_episodes": float(len(episodes)),
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
    reward_config = load_reward_config(arguments.reward_config)
    if reward_config.discount != ppo_config.gamma:
        raise ValueError("Reward potential discount must match PPO gamma")
    supervisor_config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=arguments.num_instances,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=arguments.max_turns,
        reward_config=reward_config,
        telemetry_transport=arguments.telemetry_transport,
        worker_profile=arguments.worker_profile,
        affinity_policy=arguments.affinity,
        diagnostic_root=arguments.run_dir / "controller-diagnostics",
    )
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.run_dir / "metrics.jsonl"
    dashboard_state = DashboardState() if arguments.dashboard is not None else None
    dashboard_server = (
        DashboardServer(dashboard_state, host=arguments.dashboard_host, port=arguments.dashboard)
        if dashboard_state is not None
        else None
    )
    if dashboard_server is not None:
        dashboard_server.start()
        print(json.dumps({"dashboard_url": dashboard_server.url}, sort_keys=True))
    try:
        with AutoDancerSupervisor(supervisor_config) as supervisor:
            environment = AutoDancerVectorEnv(supervisor)
            if dashboard_state is not None:
                dashboard_state.set_status("training")

            def publish_telemetry(
                index: int,
                observation: dict[str, np.ndarray],
                info: dict[str, Any],
                action: int | None,
                reward: float | None,
            ) -> None:
                assert dashboard_state is not None
                dashboard_state.update_worker(
                    index,
                    environment.worker_ids[index],
                    observation,
                    info,
                    action=action,
                    reward=reward,
                )

            telemetry_callback = publish_telemetry if dashboard_state is not None else None
            try:
                # Resume replaces every tensor, but a warm start can intentionally
                # leave new architecture modules and the critic at fresh values.
                initialize = arguments.resume is None
                if arguments.architecture == 2:
                    model: PolicyModel = RecurrentActorCritic(
                        ModelConfig(map_size=0), initialize=initialize
                    )
                elif arguments.architecture == 7:
                    model = AdapterActorCritic(initialize=initialize)
                elif arguments.architecture == 8:
                    model = ProjectedAdapterActorCritic(initialize=initialize)
                else:
                    model = RecurrentActorCritic(ModelConfig(), initialize=initialize)
                algorithm = RecurrentPPO(
                    model,
                    ppo_config,
                    device=device,
                    checkpoint_metadata={
                        "reward": reward_config.specification(),
                        "action_contract": arguments.action_contract,
                        "freeze_base_updates": arguments.freeze_base_updates,
                    },
                )
                if arguments.resume:
                    algorithm.load(arguments.resume)
                elif arguments.initialize_from:
                    algorithm.initialize_from(arguments.initialize_from)
                elif arguments.fine_tune_from:
                    algorithm.initialize_for_finetune(arguments.fine_tune_from)
                collector = VersionedAsyncRolloutCollector(
                    environment,
                    algorithm.model,
                    device=device,
                    seed=arguments.seed,
                    telemetry_callback=telemetry_callback,
                    batch_delay=arguments.inference_batch_delay_ms / 1000.0,
                    action_contract=arguments.action_contract,
                    initial_policy_version=algorithm.updates,
                )
                started = time.monotonic()
                process_start_step = algorithm.global_step
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
                    base_frozen = False
                    if isinstance(model, ProjectedAdapterActorCritic):
                        base_frozen = algorithm.updates < arguments.freeze_base_updates
                        model.set_base_trainable(not base_frozen)
                    update_metrics = algorithm.update(rollout)
                    elapsed = max(time.monotonic() - started, 1.0e-6)
                    metrics = {
                        "global_step": algorithm.global_step,
                        "updates": algorithm.updates,
                        "base_frozen": float(base_frozen),
                        "steps_per_second": (
                            algorithm.global_step - process_start_step
                        ) / elapsed,
                        **update_metrics,
                        **collector.last_runtime_metrics,
                        **episode_metrics(collector.completed_episodes),
                        **{
                            f"reward_{name}": value
                            for name, value in collector.last_reward_components.items()
                        },
                        "worker_restarts": sum(
                            handle.restart_count for handle in supervisor.workers.values()
                        ),
                        **(
                            model.architecture_metrics()
                            if isinstance(
                                model, (AdapterActorCritic, ProjectedAdapterActorCritic)
                            )
                            else {}
                        ),
                    }
                    if next_evaluation is not None and algorithm.global_step >= next_evaluation:
                        if dashboard_state is not None:
                            dashboard_state.set_status("evaluating")
                        metrics.update(
                            evaluate_policy(
                                environment,
                                algorithm.model,
                                device=device,
                                seed=arguments.seed + algorithm.updates,
                                steps=arguments.evaluation_steps,
                                action_contract=arguments.action_contract,
                            )
                        )
                        if dashboard_state is not None:
                            dashboard_state.set_status("training")
                        collector.close()
                        collector = VersionedAsyncRolloutCollector(
                            environment,
                            algorithm.model,
                            device=device,
                            seed=arguments.seed + algorithm.global_step,
                            telemetry_callback=telemetry_callback,
                            batch_delay=arguments.inference_batch_delay_ms / 1000.0,
                            action_contract=arguments.action_contract,
                            initial_policy_version=algorithm.updates,
                        )
                        next_evaluation += arguments.evaluation_interval
                    collector.completed_episodes.clear()
                    if dashboard_state is not None:
                        dashboard_state.update_health(supervisor.health())
                        dashboard_state.update_training(metrics)
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                    print(json.dumps(metrics, sort_keys=True))
                    if algorithm.global_step % arguments.checkpoint_interval < (
                        ppo_config.rollout_length * arguments.num_instances
                    ):
                        latest = arguments.run_dir / "latest.pt"
                        algorithm.save(latest, metrics=metrics)
                        archive_checkpoint(
                            latest,
                            arguments.run_dir / f"checkpoint-{algorithm.global_step:08d}.pt",
                        )
                algorithm.save(arguments.run_dir / "final.pt", metrics=metrics)
                collector.close()
                if dashboard_state is not None:
                    dashboard_state.set_status("complete")
                (arguments.run_dir / "config.json").write_text(
                    json.dumps(
                        {
                            "ppo": asdict(ppo_config),
                            "architecture": model.architecture_spec(),
                            "reward": reward_config.specification(),
                            "initialized_from": algorithm.checkpoint_metadata.get(
                                "initialization"
                            ),
                            "action_contract": arguments.action_contract,
                            "freeze_base_updates": arguments.freeze_base_updates,
                            "supervisor": {
                                "num_instances": arguments.num_instances,
                                "game_dir": str(arguments.game_dir),
                                "mod_dir": str(arguments.mod_dir),
                                "telemetry_transport": arguments.telemetry_transport,
                                "worker_profile": arguments.worker_profile,
                                "affinity": arguments.affinity,
                                "startup_timeout": arguments.startup_timeout,
                                "turn_timeout": arguments.turn_timeout,
                                "reset_timeout": arguments.reset_timeout,
                                "collector": "versioned-async",
                                "inference_batch_delay_ms": arguments.inference_batch_delay_ms,
                            },
                            "dashboard_url": (
                                dashboard_server.url if dashboard_server is not None else None
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            finally:
                environment.close()
    finally:
        if dashboard_server is not None:
            dashboard_server.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train recurrent PPO in live NecroDancer")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--fine-tune-from",
        type=Path,
        help="preserve the complete source function while resetting optimizer state",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="warm-start policy/representation weights while keeping a fresh critic and optimizer",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--architecture",
        type=int,
        choices=(2, 6, 7, 8),
        default=6,
        help="policy architecture (A8 is the zero-projection staged adapter)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--evaluation-interval", type=int, default=50000)
    parser.add_argument("--evaluation-steps", type=int, default=512)
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    parser.add_argument("--turn-timeout", type=float, default=10.0)
    parser.add_argument("--reset-timeout", type=float, default=30.0)
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--telemetry-transport", choices=("native-pipe",), default="native-pipe")
    parser.add_argument("--worker-profile", choices=("symbolic",), default="symbolic")
    parser.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    parser.add_argument("--inference-batch-delay-ms", type=float, default=2.0)
    parser.add_argument("--action-contract", choices=ACTION_CONTRACTS, default="current")
    parser.add_argument("--freeze-base-updates", type=int, default=0)
    parser.add_argument(
        "--reward-config",
        type=Path,
        help="JSON object overriding the versioned default reward weights",
    )
    parser.add_argument(
        "--dashboard",
        nargs="?",
        type=int,
        const=8765,
        metavar="PORT",
        help="serve the live symbolic worker dashboard (default port: 8765)",
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    if arguments.total_steps <= 0 or arguments.num_instances <= 0:
        parser.error("--total-steps and --num-instances must be positive")
    sources = (arguments.resume, arguments.initialize_from, arguments.fine_tune_from)
    if sum(value is not None for value in sources) > 1:
        parser.error("--resume, --initialize-from, and --fine-tune-from are mutually exclusive")
    if arguments.dashboard is not None and not 0 <= arguments.dashboard <= 65535:
        parser.error("--dashboard port must be in [0, 65535]")
    if arguments.inference_batch_delay_ms < 0:
        parser.error("--inference-batch-delay-ms cannot be negative")
    if arguments.freeze_base_updates < 0:
        parser.error("--freeze-base-updates cannot be negative")
    if arguments.freeze_base_updates and arguments.architecture != 8:
        parser.error("--freeze-base-updates is only valid for Architecture 8")
    train(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
