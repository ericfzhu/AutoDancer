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

from autodancer.curriculum import fixed_reset_spec, load_curriculum_mixture
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.live.bridge import CURRICULUM_PROFILES
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.progress import deeper_level, level_progress
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
    model_from_spec,
)
from autodancer.training.natural_prefix import (
    NATURAL_PREFIX_RECURRENT_MODES,
    NaturalPrefixConfig,
)
from autodancer.training.ppo import PPOConfig, RecurrentPPO, RolloutBatch
from autodancer.training.seed_schedule import parse_training_seed_pool

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
        terminations: list[Tensor] = []
        truncation_values: list[Tensor] = []
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
            terminal_bootstrap = torch.zeros(self.environment.num_envs, dtype=torch.float32)
            bootstrap_indices = np.flatnonzero(truncated & ~terminated).tolist()
            if bootstrap_indices:
                with torch.inference_mode():
                    terminal_observation = tensor_observation(next_observation, self.device)
                    terminal_observation["previous_action"] = action.to(self.device)
                    terminal_observation["previous_reward"] = torch.from_numpy(
                        reward.astype(np.float32, copy=False)
                    ).to(self.device)
                    _, terminal_values, _ = self.model.step(terminal_observation, next_hidden)
                terminal_bootstrap[bootstrap_indices] = terminal_values[bootstrap_indices].cpu()
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
            terminations.append(torch.from_numpy(terminated.copy()))
            truncation_values.append(terminal_bootstrap)
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
            terminations=torch.stack(terminations),
            truncation_values=torch.stack(truncation_values),
            episode_starts=torch.stack(episode_starts),
            values=torch.stack(values),
            hiddens=torch.stack(hiddens),
            next_value=next_value.cpu(),
        )


def episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {
            "episodes": 0.0,
            "curriculum_completions": 0.0,
            "time_limits": 0.0,
            "episode_seeds": [],
        }
    events = [event for episode in episodes for event in episode.get("events", [])]
    boss_events = [event for event in events if bool((event.get("data") or {}).get("boss", False))]
    boss_add_events = [
        event for event in events if bool((event.get("data") or {}).get("boss_add", False))
    ]
    statuses = [str(episode.get("status", "")) for episode in episodes]
    prefixes = [
        dict(episode.get("natural_prefix") or {})
        for episode in episodes
        if episode.get("natural_prefix")
    ]
    boss_types = sorted({int(episode.get("boss_type", 0)) for episode in episodes})
    deepest_level = (0, 0)
    for episode in episodes:
        deepest_level = deeper_level(
            deepest_level,
            (int(episode.get("zone") or 0), int(episode.get("floor") or 0)),
        )
    return {
        "episodes": float(len(episodes)),
        "mean_return": float(np.mean([episode["return"] for episode in episodes])),
        "mean_extrinsic_return": float(
            np.mean([episode.get("extrinsic_return", 0.0) for episode in episodes])
        ),
        "mean_shaping_return": float(
            np.mean([episode.get("shaping_return", 0.0) for episode in episodes])
        ),
        "deaths": float(sum(status == "dead" for status in statuses)),
        "completions": float(sum(status == "won" for status in statuses)),
        "curriculum_completions": float(
            sum(status == "curriculum_complete" for status in statuses)
        ),
        "time_limits": float(sum(status == "time_limit" for status in statuses)),
        "natural_prefix_episodes": float(len(prefixes)),
        "natural_prefix_acquisition_rate": (
            float(np.mean([bool(prefix.get("acquired", False)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_mean_guide_turns": (
            float(np.mean([int(prefix.get("guide_turns", 0)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_mean_attempts": (
            float(np.mean([int(prefix.get("attempts", 0)) for prefix in prefixes]))
            if prefixes
            else 0.0
        ),
        "natural_prefix_boundaries_valid": float(
            bool(prefixes)
            and all(
                not bool(prefix.get("acquired", False))
                or bool((prefix.get("boundary") or {}).get("reached"))
                for prefix in prefixes
            )
        ),
        "episode_seeds": sorted(
            {int(episode["seed"]) for episode in episodes if "seed" in episode}
        ),
        "boss_type_counts": {
            str(boss_type): float(
                sum(int(episode.get("boss_type", 0)) == boss_type for episode in episodes)
            )
            for boss_type in boss_types
        },
        "enemy_kills": float(sum(event.get("kind") == "enemy_kill" for event in events)),
        "boss_damage": float(
            sum(
                int(event.get("amount", 0) or 0)
                for event in boss_events
                if event.get("kind") == "enemy_damage"
            )
        ),
        "boss_add_damage": float(
            sum(
                int(event.get("amount", 0) or 0)
                for event in boss_add_events
                if event.get("kind") == "enemy_damage"
            )
        ),
        "boss_kills": float(sum(event.get("kind") == "enemy_kill" for event in boss_events)),
        "boss_add_kills": float(
            sum(event.get("kind") == "enemy_kill" for event in boss_add_events)
        ),
        "items_collected": float(sum(event.get("kind") == "item_collected" for event in events)),
        "furthest_zone": float(deepest_level[0]),
        "furthest_floor": float(deepest_level[1]),
        "furthest_level": float(level_progress(*deepest_level)),
    }


def evaluate_policy(
    environment: AutoDancerVectorEnv,
    model: PolicyModel,
    *,
    device: torch.device,
    seed: int,
    steps: int,
    action_contract: str = "current",
    guide_model: PolicyModel | None = None,
    natural_prefix: NaturalPrefixConfig | None = None,
) -> dict[str, float]:
    """Evaluate deterministically on every worker, leaving all workers reset."""
    from autodancer.training.baseline import _evaluate_deterministic_async

    rng = np.random.default_rng(seed)
    episodes = _evaluate_deterministic_async(
        environment,
        model,
        seeds=rng.integers(0, 2**31, size=environment.num_envs, dtype=np.int64).tolist(),
        max_steps=steps,
        device=device,
        dashboard_state=None,
        action_contract=action_contract,
        guide_model=guide_model,
        natural_prefix=natural_prefix,
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
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
    )
    reward_config = load_reward_config(arguments.reward_config)
    natural_prefix = (
        None
        if arguments.natural_prefix_guide is None
        else NaturalPrefixConfig(
            target_phase=arguments.natural_prefix_target_phase,
            max_guide_turns=arguments.natural_prefix_max_turns,
            max_attempts=arguments.natural_prefix_max_attempts,
            max_failed_seeds_per_fragment=arguments.natural_prefix_max_failed_seeds,
            deterministic_guide=arguments.natural_prefix_guide_mode == "deterministic",
            guide_policy_seed=arguments.natural_prefix_policy_seed,
            recurrent_state_mode=arguments.natural_prefix_recurrent_state,
        )
    )
    training_seed_pool = (
        ()
        if arguments.training_seed_pool is None
        else parse_training_seed_pool(arguments.training_seed_pool)
    )
    curriculum_entries = (
        load_curriculum_mixture(arguments.curriculum_mixture)
        if arguments.curriculum_mixture is not None
        else fixed_reset_spec(
            arguments.curriculum_start_level,
            arguments.curriculum_target_level,
            arguments.curriculum_profile,
        )
    )
    curriculum_distribution = {
        "schema_version": 1,
        "mode": "weighted-per-episode-v1",
        "entries": [entry.as_dict() for entry in curriculum_entries],
    }
    seed_checkpoint_metadata = (
        {
            "training_seed_schedule": "uniform-pool-v1",
            "training_seed_pool": list(training_seed_pool),
        }
        if training_seed_pool
        else {}
    )
    curriculum_metadata = (
        {"curriculum_distribution": curriculum_distribution}
        if arguments.curriculum_mixture is not None
        else {
            "curriculum": {
                "start_level": arguments.curriculum_start_level,
                "target_level": arguments.curriculum_target_level,
                "profile": arguments.curriculum_profile,
                "reset_semantics": "normal-reset-sequential-goto-reward-reset-v1",
            }
        }
        if arguments.curriculum_start_level != 1 or arguments.curriculum_target_level is not None
        else {}
    )
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
        curriculum_start_level=arguments.curriculum_start_level,
        curriculum_target_level=arguments.curriculum_target_level,
        curriculum_profile=arguments.curriculum_profile,
        curriculum_commands_enabled=any(entry.spec.start_level > 1 for entry in curriculum_entries),
    )
    arguments.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.run_dir / "metrics.jsonl"
    tracker: Any | None = None
    experiment_id = getattr(arguments, "experiment_id", None)
    if experiment_id is not None:
        from autodancer.experiments.provenance import sha256_file
        from autodancer.experiments.tracking import ExperimentTracker, LineageConfig

        source_checkpoint = (
            arguments.resume or arguments.initialize_from or arguments.fine_tune_from
        )
        tracker = ExperimentTracker(
            LineageConfig(
                experiment_id=experiment_id,
                arm=arguments.experiment_arm,
                trial=arguments.trial_id or f"seed-{arguments.seed}",
                stage="training",
                run_dir=arguments.run_dir,
                store_root=arguments.experiment_root,
                tracking_uri=arguments.mlflow_tracking_uri,
                qualification_report=arguments.controller_qualification,
            ),
            game_dir=arguments.game_dir,
            mod_dir=arguments.mod_dir,
            device=str(device),
            parameters={
                "seed": arguments.seed,
                "total_steps": arguments.total_steps,
                "num_instances": arguments.num_instances,
                "max_turns": arguments.max_turns,
                "architecture": arguments.architecture,
                "ppo": asdict(ppo_config),
                "reward": reward_config.specification(),
                "reward_lineage_version": arguments.reward_lineage_version,
                "reward_config": (
                    None if arguments.reward_config is None else str(arguments.reward_config)
                ),
                "reward_config_sha256": sha256_file(arguments.reward_config),
                "action_contract": arguments.action_contract,
                "training_seed_schedule": (
                    "uniform-pool-v1" if training_seed_pool else "unbounded-random-v1"
                ),
                "training_seed_pool": list(training_seed_pool),
                "curriculum_start_level": arguments.curriculum_start_level,
                "curriculum_target_level": arguments.curriculum_target_level,
                "curriculum_profile": arguments.curriculum_profile,
                "curriculum_mixture": curriculum_distribution,
                "freeze_base_updates": arguments.freeze_base_updates,
                "telemetry_transport": arguments.telemetry_transport,
                "worker_profile": arguments.worker_profile,
                "affinity": arguments.affinity,
            },
            source_checkpoint=source_checkpoint,
        )
        try:
            if arguments.reward_lineage_version is None:
                raise ValueError("Tracked training requires --reward-lineage-version")
            if not arguments.reward_lineage_version.startswith(f"V{reward_config.profile_version}"):
                raise ValueError("Reward lineage version disagrees with the loaded reward profile")
            tracker.validate_component_versions(
                {
                    "architecture": f"A{arguments.architecture}",
                    "reward": arguments.reward_lineage_version,
                },
                config_hashes={"reward": sha256_file(arguments.reward_config)},
            )
            if training_seed_pool:
                tracker.validate_component_versions(
                    {
                        "training-level-distribution": (
                            "mixed-curriculum-replay-v1"
                            if arguments.curriculum_mixture is not None
                            else "reverse-curriculum-sequential-goto-v1"
                            if arguments.curriculum_start_level != 1
                            else "uniform-finite-pool-v1"
                        )
                    },
                    require_declared=True,
                )
        except BaseException as error:
            tracker.fail(error)
            raise
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
                        "max_turns": arguments.max_turns,
                        **seed_checkpoint_metadata,
                        **curriculum_metadata,
                        **(
                            {
                                "natural_prefix": {
                                    **natural_prefix.specification(),
                                    "guide_checkpoint": str(
                                        arguments.natural_prefix_guide.resolve()
                                    ),
                                }
                            }
                            if natural_prefix is not None
                            else {}
                        ),
                        "freeze_base_updates": arguments.freeze_base_updates,
                    },
                )
                resume_metrics: dict[str, Any] = {}
                if arguments.resume:
                    resume_metrics = algorithm.load(arguments.resume)
                elif arguments.initialize_from:
                    algorithm.initialize_from(arguments.initialize_from)
                elif arguments.fine_tune_from:
                    algorithm.initialize_for_finetune(arguments.fine_tune_from)
                guide_model: PolicyModel | None = None
                if arguments.natural_prefix_guide is not None:
                    guide_payload = torch.load(
                        arguments.natural_prefix_guide,
                        map_location=device,
                        weights_only=False,
                    )
                    guide_model = model_from_spec(
                        guide_payload.get("architecture", {}), initialize=False
                    ).to(device)
                    guide_model.load_state_dict(guide_payload["model"])
                    guide_model.eval()
                    for parameter in guide_model.parameters():
                        parameter.requires_grad_(False)
                if (
                    arguments.resume is not None
                    and arguments.curriculum_mixture is not None
                    and algorithm.global_step > 0
                    and "curriculum_schedule_state" not in resume_metrics
                ):
                    raise ValueError("The resume checkpoint has no exact curriculum schedule state")
                if tracker is not None:
                    tracker.set_resolved(
                        {
                            "device": str(device),
                            "architecture": model.architecture_spec(),
                            "reward": reward_config.specification(),
                            "initialization": algorithm.checkpoint_metadata.get("initialization"),
                            "starting_global_step": algorithm.global_step,
                            "starting_updates": algorithm.updates,
                        }
                    )
                collector = VersionedAsyncRolloutCollector(
                    environment,
                    algorithm.model,
                    device=device,
                    seed=arguments.seed,
                    telemetry_callback=telemetry_callback,
                    batch_delay=arguments.inference_batch_delay_ms / 1000.0,
                    action_contract=arguments.action_contract,
                    initial_policy_version=algorithm.updates,
                    training_seed_pool=training_seed_pool,
                    seed_schedule_state=resume_metrics.get("training_seed_schedule_state"),
                    curriculum_entries=curriculum_entries,
                    curriculum_schedule_state=resume_metrics.get("curriculum_schedule_state"),
                    guide_model=guide_model,
                    natural_prefix=natural_prefix,
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
                        "steps_per_second": (algorithm.global_step - process_start_step) / elapsed,
                        **update_metrics,
                        **collector.last_runtime_metrics,
                        "training_seed_schedule_state": collector.seed_schedule_state(),
                        "curriculum_schedule_state": collector.curriculum_schedule_state(),
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
                            if isinstance(model, (AdapterActorCritic, ProjectedAdapterActorCritic))
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
                                guide_model=guide_model,
                                natural_prefix=natural_prefix,
                            )
                        )
                        if dashboard_state is not None:
                            dashboard_state.set_status("training")
                        schedule_state = collector.seed_schedule_state()
                        curriculum_schedule_state = collector.curriculum_schedule_state()
                        collector.close()
                        collector = VersionedAsyncRolloutCollector(
                            environment,
                            algorithm.model,
                            device=device,
                            seed=arguments.seed,
                            telemetry_callback=telemetry_callback,
                            batch_delay=arguments.inference_batch_delay_ms / 1000.0,
                            action_contract=arguments.action_contract,
                            initial_policy_version=algorithm.updates,
                            training_seed_pool=training_seed_pool,
                            seed_schedule_state=schedule_state,
                            curriculum_entries=curriculum_entries,
                            curriculum_schedule_state=curriculum_schedule_state,
                            guide_model=guide_model,
                            natural_prefix=natural_prefix,
                        )
                        metrics["training_seed_schedule_state"] = collector.seed_schedule_state()
                        metrics["curriculum_schedule_state"] = collector.curriculum_schedule_state()
                        next_evaluation += arguments.evaluation_interval
                    collector.completed_episodes.clear()
                    if dashboard_state is not None:
                        dashboard_state.update_health(supervisor.health())
                        dashboard_state.update_training(metrics)
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                    if tracker is not None:
                        tracker.log_metrics(metrics, step=algorithm.global_step)
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
                            "initialized_from": algorithm.checkpoint_metadata.get("initialization"),
                            "action_contract": arguments.action_contract,
                            "max_turns": arguments.max_turns,
                            "training_seed_schedule": (
                                "uniform-pool-v1" if training_seed_pool else "unbounded-random-v1"
                            ),
                            "training_seed_pool": list(training_seed_pool),
                            "curriculum_start_level": arguments.curriculum_start_level,
                            "curriculum_target_level": arguments.curriculum_target_level,
                            "curriculum_profile": arguments.curriculum_profile,
                            "curriculum_mixture": curriculum_distribution,
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
                                "max_turns": arguments.max_turns,
                                "collector": "versioned-async",
                                "inference_batch_delay_ms": arguments.inference_batch_delay_ms,
                            },
                            "dashboard_url": (
                                dashboard_server.url if dashboard_server is not None else None
                            ),
                            "lineage": (
                                None
                                if tracker is None
                                else {
                                    "experiment_id": tracker.config.experiment_id,
                                    "arm": tracker.config.arm,
                                    "trial": tracker.config.trial,
                                    "mlflow_run_id": tracker.run_id,
                                    "spec_sha256": tracker.spec.digest,
                                }
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                if tracker is not None:
                    tracker.complete(
                        [
                            arguments.run_dir / "config.json",
                            metrics_path,
                            arguments.run_dir / "final.pt",
                        ],
                        summary=metrics,
                    )
            finally:
                environment.close()
    except BaseException as error:
        if tracker is not None:
            try:
                tracker.fail(error)
            except BaseException:
                pass
        raise
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
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor; must match the reward profile's potential discount",
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
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
    parser.add_argument(
        "--training-seed-pool",
        help="finite `start-end` or comma-separated game-seed pool sampled on resets",
    )
    parser.add_argument(
        "--curriculum-start-level",
        type=int,
        default=1,
        help="sequential All Zones level to start from after each normal seeded reset",
    )
    parser.add_argument(
        "--curriculum-target-level",
        type=int,
        help="terminate the curriculum episode successfully on entering this level",
    )
    parser.add_argument(
        "--curriculum-profile",
        choices=CURRICULUM_PROFILES,
        default="normal",
        help="qualification-only assistance applied on the curriculum start level",
    )
    parser.add_argument(
        "--curriculum-mixture",
        type=Path,
        help=(
            "schema-1 JSON weighted per-episode reset distribution; mutually exclusive "
            "with non-default fixed curriculum arguments"
        ),
    )
    parser.add_argument(
        "--natural-prefix-guide",
        type=Path,
        help=(
            "frozen guide checkpoint that reaches a legally observed Death Metal "
            "phase before learner collection begins"
        ),
    )
    parser.add_argument(
        "--natural-prefix-target-phase",
        type=int,
        choices=(2, 3, 4),
        default=4,
    )
    parser.add_argument("--natural-prefix-max-turns", type=int, default=512)
    parser.add_argument("--natural-prefix-max-attempts", type=int, default=8)
    parser.add_argument("--natural-prefix-max-failed-seeds", type=int, default=16)
    parser.add_argument(
        "--natural-prefix-guide-mode",
        choices=("deterministic", "stochastic"),
        default="stochastic",
    )
    parser.add_argument("--natural-prefix-policy-seed", type=int, default=0)
    parser.add_argument(
        "--natural-prefix-recurrent-state",
        choices=NATURAL_PREFIX_RECURRENT_MODES,
        default="fresh",
    )
    parser.add_argument("--freeze-base-updates", type=int, default=0)
    parser.add_argument(
        "--reward-config",
        type=Path,
        help="JSON object overriding the versioned default reward weights",
    )
    parser.add_argument(
        "--reward-lineage-version",
        help="component catalog label such as V2 or V4A (required for tracked runs)",
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
    parser.add_argument("--experiment-id", help="registered immutable experiment id")
    parser.add_argument("--experiment-arm", help="arm id declared in experiment.yaml")
    parser.add_argument("--trial-id", help="stable trial label (defaults to seed-N)")
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments"))
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument(
        "--controller-qualification",
        type=Path,
        default=Path("runs/controller-qualification/qualification.json"),
    )
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
    if arguments.curriculum_start_level <= 0:
        parser.error("--curriculum-start-level must be positive")
    if arguments.curriculum_profile != "normal" and arguments.curriculum_start_level <= 1:
        parser.error("assisted --curriculum-profile requires --curriculum-start-level > 1")
    if (
        arguments.curriculum_target_level is not None
        and arguments.curriculum_target_level <= arguments.curriculum_start_level
    ):
        parser.error("--curriculum-target-level must be after --curriculum-start-level")
    if arguments.curriculum_mixture is not None and (
        arguments.curriculum_start_level != 1
        or arguments.curriculum_target_level is not None
        or arguments.curriculum_profile != "normal"
    ):
        parser.error("--curriculum-mixture is mutually exclusive with fixed curriculum arguments")
    if arguments.natural_prefix_guide is not None:
        if arguments.curriculum_start_level != 4 or arguments.curriculum_target_level != 5:
            parser.error(
                "--natural-prefix-guide currently requires the Zone 1 boss-to-Zone 2 "
                "curriculum (--curriculum-start-level 4 --curriculum-target-level 5)"
            )
        if arguments.curriculum_mixture is not None:
            parser.error("natural-prefix training does not yet support curriculum mixtures")
        if arguments.natural_prefix_max_turns <= 0 or arguments.natural_prefix_max_attempts <= 0:
            parser.error("natural-prefix turn and attempt limits must be positive")
    if bool(arguments.experiment_id) != bool(arguments.experiment_arm):
        parser.error("--experiment-id and --experiment-arm must be supplied together")
    train(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
