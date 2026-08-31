"""Build and qualify successful action traces against fresh live game resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from autodancer.constants import ACTION_COUNT, GridChannel, PlayerFeature, Terrain
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.progress import deeper_level
from autodancer.rewards import RewardConfig, RewardTracker, load_reward_config
from autodancer.training.action_contract import ACTION_CONTRACTS, ActionContractMemory
from autodancer.training.demonstrations import (
    build_demonstration_bank,
    iter_trace_actions,
    normalized_observation_digest,
    validate_demonstration_bank,
    validate_demonstration_sources,
    write_demonstration_bank,
)
from autodancer.training.imitation_sequences import (
    OBSERVATION_NAMES,
    RecurrentDemonstration,
    write_recurrent_demonstrations,
)
from autodancer.training.model import START_ACTION
from autodancer.training.train import default_mod_dir


class ReplayEnvironment(Protocol):
    def reset(
        self, *, seed: int, options: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]: ...


@dataclass(slots=True)
class RecurrentReplayCapture:
    """Collect exact learner inputs while a qualified action trace is replayed."""

    action_contract: str
    feedback_config: RewardConfig
    demonstration: RecurrentDemonstration | None = field(default=None, init=False)
    _contract: ActionContractMemory = field(init=False, repr=False)
    _feedback: RewardTracker = field(init=False, repr=False)
    _observation: dict[str, np.ndarray] = field(init=False, repr=False)
    _previous_action: int = field(default=START_ACTION, init=False, repr=False)
    _previous_reward: float = field(default=0.0, init=False, repr=False)
    _observations: dict[str, list[np.ndarray]] = field(init=False, repr=False)
    _actions: list[int] = field(default_factory=list, init=False, repr=False)
    _previous_actions: list[int] = field(default_factory=list, init=False, repr=False)
    _previous_rewards: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._contract = ActionContractMemory(self.action_contract, 1)
        self._feedback = RewardTracker(self.feedback_config)
        self._observations = {name: [] for name in OBSERVATION_NAMES}

    def begin(self, observation: dict[str, np.ndarray], info: Mapping[str, Any]) -> None:
        missing = set(OBSERVATION_NAMES) - set(observation)
        if missing:
            raise ValueError(
                "recurrent replay observation is missing policy inputs: "
                + ", ".join(sorted(missing))
            )
        self._observation = self._contract.reset_slot(0, observation)
        self._feedback.reset(self._observation, dict(info))

    def before_action(self, action: int) -> None:
        if not bool(self._observation["action_mask"][action]):
            raise ValueError(f"recorded action {action} is masked by the action contract")
        for name in OBSERVATION_NAMES:
            self._observations[name].append(self._observation[name].copy())
        self._actions.append(int(action))
        self._previous_actions.append(self._previous_action)
        self._previous_rewards.append(self._previous_reward)

    def after_action(
        self,
        action: int,
        next_observation: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        next_info = dict(info)
        self._contract.observe(0, self._observation, action, next_observation, next_info)
        effective = self._contract.apply_slot(0, next_observation)
        feedback, _ = self._feedback.score(
            effective,
            next_info,
            next_info.get("raw_events", ()),
            terminated=terminated,
            truncated=truncated,
        )
        self._observation = effective
        self._previous_action = int(action)
        self._previous_reward = float(feedback if self.feedback_config is not None else reward)

    def finalize(self, trace: Mapping[str, Any], *, valid: bool) -> None:
        if not valid:
            self.demonstration = None
            return
        length = len(self._actions)
        demonstration = RecurrentDemonstration(
            trace_id=str(trace["trace_id"]),
            seed=int(trace["seed"]),
            observations={
                name: np.stack(values).astype(values[0].dtype, copy=False)
                for name, values in self._observations.items()
            },
            actions=np.asarray(self._actions, dtype=np.int64),
            previous_actions=np.asarray(self._previous_actions, dtype=np.int64),
            previous_rewards=np.asarray(self._previous_rewards, dtype=np.float32),
            episode_starts=np.asarray([True, *([False] * (length - 1))], dtype=np.bool_),
        )
        demonstration.validate()
        self.demonstration = demonstration


def _event_counts(events: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events or []:
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("kind") or event.get("type") or "")
        if kind:
            counts[kind] += 1
    return counts


def _observation_summary(observation: Mapping[str, np.ndarray]) -> dict[str, Any]:
    player = np.asarray(observation["player"])
    terrain = np.asarray(observation["grid"])[..., int(GridChannel.TERRAIN_CLASS)]
    centre = terrain.shape[0] // 2
    stairs = [
        [int(row - centre), int(column - centre)]
        for row, column in np.argwhere(terrain == int(Terrain.STAIRS))
    ]
    return {
        "zone": int(player[PlayerFeature.ZONE]),
        "floor": int(player[PlayerFeature.FLOOR]),
        "turn": int(player[PlayerFeature.TURN]),
        "health": int(player[PlayerFeature.HEALTH]),
        "on_stairs": bool(player[PlayerFeature.ON_STAIRS]),
        "dead": bool(player[PlayerFeature.DEAD]),
        "visible_enemies": int(player[PlayerFeature.VISIBLE_ENEMIES]),
        "visible_stairs_relative": stairs,
        "action_mask": [int(value) for value in observation["action_mask"]],
        "digest": normalized_observation_digest(dict(observation)),
    }


def replay_trace(
    environment: ReplayEnvironment,
    trace: Mapping[str, Any],
    *,
    recurrent_capture: RecurrentReplayCapture | None = None,
) -> dict[str, Any]:
    """Replay one trace and compare its terminal gameplay evidence exactly."""

    started = time.monotonic()
    seed = int(trace["seed"])
    reset = dict(trace["curriculum_reset"])
    observation, reset_info = environment.reset(seed=seed, options={"curriculum": reset})
    if recurrent_capture is not None:
        recurrent_capture.begin(observation, reset_info)
    turn_digests = [normalized_observation_digest(observation)]
    actual_events: Counter[str] = Counter()
    furthest = (
        int(reset_info.get("zone") or observation["player"][PlayerFeature.ZONE]),
        int(reset_info.get("floor") or observation["player"][PlayerFeature.FLOOR]),
    )
    terminal = False
    status = str(reset_info.get("episode_status", "running"))
    executed = 0
    error = ""
    for action in iter_trace_actions(trace):
        if terminal:
            error = "episode terminated before the recorded action sequence ended"
            break
        mask = observation.get("action_mask")
        if mask is None or np.asarray(mask).shape != (ACTION_COUNT,):
            error = "live replay observation has an invalid action mask"
            break
        if not bool(mask[action]):
            error = f"recorded action {action} is masked at replay turn {executed}"
            break
        if recurrent_capture is not None:
            try:
                recurrent_capture.before_action(action)
            except ValueError as capture_error:
                error = str(capture_error)
                break
        observation, _, terminated, truncated, info = environment.step(action)
        turn_digests.append(normalized_observation_digest(observation))
        executed += 1
        terminal = bool(terminated or truncated)
        status = str(info.get("episode_status", "running"))
        actual_events.update(_event_counts(info.get("raw_events")))
        if recurrent_capture is not None:
            recurrent_capture.after_action(
                action,
                observation,
                0.0,
                bool(terminated),
                bool(truncated),
                info,
            )
        furthest = deeper_level(
            furthest,
            (
                int(info.get("zone") or observation["player"][PlayerFeature.ZONE]),
                int(info.get("floor") or observation["player"][PlayerFeature.FLOOR]),
            ),
        )
    expected_events = {str(key): int(value) for key, value in trace["event_counts"].items()}
    checks = {
        "all_actions_executed": executed == int(trace["turns"]),
        "terminal": terminal,
        "status": status == str(trace["status"]),
        "furthest_zone": furthest[0] == int(trace["furthest_zone"]),
        "furthest_floor": furthest[1] == int(trace["furthest_floor"]),
        "event_counts": dict(actual_events) == expected_events,
    }
    result = {
        "trace_id": str(trace["trace_id"]),
        "seed": seed,
        "valid": not error and all(checks.values()),
        "error": error,
        "checks": checks,
        "expected": {
            "status": trace["status"],
            "turns": trace["turns"],
            "furthest_zone": trace["furthest_zone"],
            "furthest_floor": trace["furthest_floor"],
            "event_counts": expected_events,
        },
        "actual": {
            "status": status,
            "turns": executed,
            "furthest_zone": furthest[0],
            "furthest_floor": furthest[1],
            "event_counts": dict(actual_events),
            "final_observation": _observation_summary(observation),
        },
        "elapsed_seconds": time.monotonic() - started,
        "turn_digests": turn_digests,
    }
    if recurrent_capture is not None:
        recurrent_capture.finalize(trace, valid=bool(result["valid"]))
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def qualify_bank(arguments: argparse.Namespace) -> dict[str, Any]:
    bank = json.loads(arguments.bank.read_text(encoding="utf-8"))
    validate_demonstration_bank(bank)
    validate_demonstration_sources(bank)
    traces = list(bank["traces"])
    if not traces:
        raise ValueError("demonstration bank contains no successful traces")
    capacity = int(arguments.num_instances)
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=capacity,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(int(trace["turns"]) for trace in traces) + 1,
        curriculum_commands_enabled=True,
        diagnostic_root=arguments.output.parent / "controller-diagnostics",
        affinity_policy=arguments.affinity,
    )
    results: list[dict[str, Any]] = []
    recurrent_demonstrations: list[RecurrentDemonstration] = []
    capture_recurrent = arguments.recurrent_output is not None
    feedback_config = (
        load_reward_config(arguments.policy_feedback_reward_config)
        if capture_recurrent
        else None
    )
    infrastructure_error = ""
    restarts = 0
    with AutoDancerSupervisor(config) as supervisor:
        assignments = [traces[index::capacity] for index in range(capacity)]

        def worker(
            slot: int,
        ) -> tuple[list[dict[str, Any]], list[RecurrentDemonstration]]:
            environment = supervisor.environment(supervisor.worker_ids[slot])
            worker_results: list[dict[str, Any]] = []
            worker_demonstrations: list[RecurrentDemonstration] = []
            for trace in assignments[slot]:
                capture = (
                    RecurrentReplayCapture(arguments.action_contract, feedback_config)
                    if feedback_config is not None
                    else None
                )
                result = replay_trace(environment, trace, recurrent_capture=capture)
                worker_results.append(result)
                if capture is not None and capture.demonstration is not None:
                    worker_demonstrations.append(capture.demonstration)
            return worker_results, worker_demonstrations

        try:
            with ThreadPoolExecutor(max_workers=capacity) as executor:
                for completed_results, completed_demonstrations in executor.map(
                    worker, range(capacity)
                ):
                    results.extend(completed_results)
                    recurrent_demonstrations.extend(completed_demonstrations)
        except BaseException as error:
            infrastructure_error = f"{type(error).__name__}: {error}"
        restarts = sum(handle.restart_count for handle in supervisor.workers.values())
    results.sort(key=lambda result: (result["seed"], result["trace_id"]))
    valid = bool(
        not infrastructure_error
        and restarts == 0
        and len(results) == len(traces)
        and all(result["valid"] for result in results)
    )
    report = {
        "schema_version": 1,
        "kind": "qualified-live-action-traces-report-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "bank": str(arguments.bank.resolve()),
        "bank_sha256": bank["bank_sha256"],
        "num_instances": capacity,
        "trace_count": len(traces),
        "qualified_trace_count": sum(bool(result["valid"]) for result in results),
        "worker_restarts": restarts,
        "infrastructure_error": infrastructure_error,
        "valid": valid,
        "results": results,
    }
    if capture_recurrent:
        report["recurrent_artifact"] = str(arguments.recurrent_output.resolve())
    _write_json_atomic(arguments.output, report)
    if capture_recurrent and valid:
        if len(recurrent_demonstrations) != len(traces):
            raise RuntimeError("qualified replay did not capture every recurrent sequence")
        recurrent_demonstrations.sort(key=lambda item: (item.seed, item.trace_id))
        feedback_path = arguments.policy_feedback_reward_config.resolve()
        write_recurrent_demonstrations(
            arguments.recurrent_output,
            recurrent_demonstrations,
            provenance={
                "bank": str(arguments.bank.resolve()),
                "bank_sha256": str(bank["bank_sha256"]),
                "qualification": str(arguments.output.resolve()),
                "qualification_sha256": _sha256_file(arguments.output),
                "action_contract": arguments.action_contract,
                "policy_feedback_reward": feedback_config.specification(),
                "policy_feedback_reward_config": str(feedback_path),
                "policy_feedback_reward_config_sha256": _sha256_file(feedback_path),
            },
        )
    return report


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or qualify successful live action traces")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a hash-bound bank from episode journals")
    build.add_argument("--episodes", type=Path, action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser("qualify", help="replay every bank trace in fresh live runs")
    qualify.add_argument("--game-dir", type=Path, required=True)
    qualify.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    qualify.add_argument("--bank", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--num-instances", type=_positive_integer, default=4)
    qualify.add_argument("--startup-timeout", type=float, default=45.0)
    qualify.add_argument("--turn-timeout", type=float, default=10.0)
    qualify.add_argument("--reset-timeout", type=float, default=30.0)
    qualify.add_argument("--affinity", choices=("auto", "none", "spread"), default="auto")
    qualify.add_argument("--recurrent-output", type=Path)
    qualify.add_argument("--action-contract", choices=ACTION_CONTRACTS)
    qualify.add_argument("--policy-feedback-reward-config", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "build":
        payload = build_demonstration_bank(arguments.episodes)
        write_demonstration_bank(arguments.output, payload)
        summary = {
            "output": str(arguments.output),
            "bank_sha256": payload["bank_sha256"],
            "trace_count": len(payload["traces"]),
        }
    else:
        if arguments.mod_dir is None:
            parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
        recurrent_arguments = (
            arguments.recurrent_output,
            arguments.action_contract,
            arguments.policy_feedback_reward_config,
        )
        if any(value is not None for value in recurrent_arguments) and not all(
            value is not None for value in recurrent_arguments
        ):
            parser.error(
                "--recurrent-output, --action-contract, and "
                "--policy-feedback-reward-config must be provided together"
            )
        report = qualify_bank(arguments)
        summary = {
            "output": str(arguments.output),
            "valid": report["valid"],
            "trace_count": report["trace_count"],
            "qualified_trace_count": report["qualified_trace_count"],
            "recurrent_artifact": report.get("recurrent_artifact"),
        }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("valid", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
