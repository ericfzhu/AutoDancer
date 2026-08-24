"""Reproducible real-game probes for the action contract and core mechanics."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from autodancer.constants import DIRECTION_DELTAS, Action, GridChannel, Terrain
from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.live.explore import LiveExplorer
from autodancer.live.protocol import SUPPORTED_GAME_VERSION, SUPPORTED_STEAM_BUILD
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig

OPPOSITE = {
    Action.UP: Action.DOWN,
    Action.RIGHT: Action.LEFT,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
}
SPECIAL_ACTIONS = (
    Action.ITEM_1,
    Action.ITEM_2,
    Action.SPELL_1,
    Action.SPELL_2,
)
MECHANIC_CATEGORIES = (
    "move",
    "wall_attempt",
    "dig",
    "combat",
    "combat_attempt",
    "interaction",
    "wait",
    "floor_transition",
)


def default_mod_dir() -> Path | None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    return Path(local_app_data) / "NecroDancer" / "mods" / "AutoDancer" if local_app_data else None


@dataclass(slots=True)
class ActionEvidence:
    action: str
    logical_id: int
    legal_seen: bool = False
    acknowledged: bool = False
    engine_action: int | None = None
    observed_action: int | None = None
    outcome: str | None = None
    seed: int | None = None


@dataclass(slots=True)
class ProbeLedger:
    actions: dict[Action, ActionEvidence] = field(
        default_factory=lambda: {
            action: ActionEvidence(action.name, int(action)) for action in Action
        }
    )
    mechanics: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: int = 0
    episodes: int = 0
    staircase_visible: bool = False
    staircase_route_attempted: bool = False
    backtrack_succeeded: bool = False

    def observe_mask(self, observation: dict[str, np.ndarray]) -> None:
        for action in Action:
            self.actions[action].legal_seen |= bool(observation["action_mask"][int(action)])
        grid = observation["grid"]
        self.staircase_visible |= bool(
            np.any(
                (grid[..., GridChannel.TERRAIN_CLASS] == int(Terrain.STAIRS))
                & (grid[..., GridChannel.VISIBILITY] > 0)
            )
        )

    def observe_step(
        self,
        action: Action,
        info: dict[str, Any],
        *,
        seed: int,
    ) -> None:
        self.turns += 1
        evidence = self.actions[action]
        acknowledgement = dict(info.get("bridge") or {})
        acknowledged = (
            acknowledgement.get("requested_action") == int(action)
            and acknowledgement.get("engine_action") == acknowledgement.get("observed_action")
        )
        outcome = dict(info.get("action_outcome") or {})
        category = str(outcome.get("category", "unknown"))
        # Preserve the first successful proof for each action. The mechanic
        # phase may execute the same action thousands of times; overwriting the
        # isolated contract evidence would make its seed dependent on the last
        # exploratory turn.
        if acknowledged and not evidence.acknowledged:
            evidence.acknowledged = True
            evidence.engine_action = acknowledgement.get("engine_action")
            evidence.observed_action = acknowledgement.get("observed_action")
            evidence.seed = seed
            evidence.outcome = category
        self.mechanics.setdefault(
            category,
            {
                "observed": True,
                "seed": seed,
                "turn": self.turns,
                "action": action.name,
                "outcome": outcome,
            },
        )

    def report(self) -> dict[str, Any]:
        action_report = {}
        for action, evidence in self.actions.items():
            value = asdict(evidence)
            value["status"] = (
                "acknowledged"
                if evidence.acknowledged
                else "legal_not_executed"
                if evidence.legal_seen
                else "masked_unavailable"
            )
            action_report[action.name] = value
        mechanic_report = {
            category: self.mechanics.get(category, {"observed": False})
            for category in MECHANIC_CATEGORIES
        }
        mechanic_report["staircase_visible"] = {"observed": self.staircase_visible}
        mechanic_report["staircase_route_attempted"] = {
            "observed": self.staircase_route_attempted
        }
        mechanic_report["return_to_visited_position"] = {
            "observed": self.backtrack_succeeded
        }
        core_actions = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.WAIT)
        return {
            "actions": action_report,
            "mechanics": mechanic_report,
            "turns": self.turns,
            "episodes": self.episodes,
            "core_action_contract_passed": all(
                self.actions[action].acknowledged for action in core_actions
            ),
            "all_legal_actions_acknowledged": all(
                not evidence.legal_seen or evidence.acknowledged
                for evidence in self.actions.values()
            ),
        }


class LiveMechanicProbes:
    def __init__(self, environment: AutoDancerLiveEnv, *, step_delay: float = 0.12) -> None:
        self.environment = environment
        self.step_delay = step_delay
        self.ledger = ProbeLedger()

    def _step(
        self,
        observation: dict[str, np.ndarray],
        action: Action,
        seed: int,
    ) -> tuple[dict[str, np.ndarray], bool]:
        if self.step_delay:
            time.sleep(self.step_delay)
        next_observation, _, terminated, truncated, info = self.environment.step(action)
        self.ledger.observe_step(action, info, seed=seed)
        self.ledger.observe_mask(next_observation)
        return next_observation, bool(terminated or truncated)

    def verify_action_contract(self, seed: int) -> None:
        for action in Action:
            observation, _ = self.environment.reset(seed=seed + int(action))
            self.ledger.episodes += 1
            self.ledger.observe_mask(observation)
            if observation["action_mask"][int(action)]:
                self._step(observation, action, seed + int(action))
            else:
                # Do not issue RESET immediately after RESET. A resolved idle
                # turn gives the engine a stable boundary before the next run.
                self._step(observation, Action.WAIT, seed + int(action))
            print(
                json.dumps(
                    {
                        "phase": "action_contract",
                        "action": action.name,
                        "legal": self.ledger.actions[action].legal_seen,
                        "acknowledged": self.ledger.actions[action].acknowledged,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def exercise_mechanics(self, seeds: list[int], max_steps: int) -> None:
        for seed in seeds:
            observation, info = self.environment.reset(seed=seed)
            self.ledger.episodes += 1
            self.ledger.observe_mask(observation)
            explorer = LiveExplorer()
            floor = int(observation["player"][7])
            pending_backtrack: tuple[Action, tuple[int, int]] | None = None
            for _ in range(max_steps):
                if pending_backtrack is not None:
                    candidate, expected = pending_backtrack
                    pending_backtrack = None
                    action = candidate
                else:
                    expected = None
                    action = next(
                        (
                            special
                            for special in SPECIAL_ACTIONS
                            if observation["action_mask"][int(special)]
                            and not self.ledger.actions[special].acknowledged
                        ),
                        None,
                    )
                    if action is None:
                        try:
                            action = explorer.choose(observation)
                        except RuntimeError:
                            action = Action.WAIT
                    if explorer.last_reason == "stairs":
                        self.ledger.staircase_route_attempted = True
                before_position = tuple(int(value) for value in observation["player"][4:6])
                observation, done = self._step(observation, action, seed)
                after_position = tuple(int(value) for value in observation["player"][4:6])
                if expected is not None and after_position == expected:
                    self.ledger.backtrack_succeeded = True
                outcome = self.ledger.actions[action].outcome
                if (
                    not self.ledger.backtrack_succeeded
                    and outcome == "move"
                    and action in DIRECTION_DELTAS
                ):
                    pending_backtrack = (OPPOSITE[action], before_position)
                next_floor = int(observation["player"][7])
                if next_floor != floor:
                    explorer.reset_level()
                    floor = next_floor
                    pending_backtrack = None
                if done:
                    break
            print(
                json.dumps(
                    {
                        "phase": "mechanics",
                        "seed": seed,
                        "turns_total": self.ledger.turns,
                        "observed": sorted(self.ledger.mechanics),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


def run_diagnostics(arguments: argparse.Namespace) -> dict[str, Any]:
    config = SupervisorConfig(
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        num_instances=1,
        startup_timeout=arguments.startup_timeout,
        turn_timeout=arguments.turn_timeout,
        reset_timeout=arguments.reset_timeout,
        max_turns=max(arguments.max_steps + 20, 100),
        affinity_policy="none",
    )
    with AutoDancerSupervisor(config) as supervisor:
        environment = supervisor.environment("worker-0000")
        probes = LiveMechanicProbes(environment, step_delay=arguments.step_delay)
        probes.verify_action_contract(arguments.contract_seed)
        probes.exercise_mechanics(arguments.seeds, arguments.max_steps)
        report = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "game_version": SUPPORTED_GAME_VERSION,
            "steam_build": SUPPORTED_STEAM_BUILD,
            "contract_seed": arguments.contract_seed,
            "mechanic_seeds": arguments.seeds,
            "max_steps_per_seed": arguments.max_steps,
            "worker_restarts": supervisor.workers["worker-0000"].restart_count,
            **probes.ledger.report(),
        }
        environment.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe live Bard actions and mechanics")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=default_mod_dir())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-seed", type=int, default=46_000)
    parser.add_argument("--seeds", default="46001,46002,46003")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--turn-timeout", type=float, default=30.0)
    parser.add_argument("--reset-timeout", type=float, default=60.0)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.12,
        help="minimum wall-clock delay between live actions (default: 0.12 seconds)",
    )
    arguments = parser.parse_args()
    if arguments.mod_dir is None:
        parser.error("--mod-dir is required when LOCALAPPDATA is unavailable")
    try:
        arguments.seeds = [int(value.strip()) for value in arguments.seeds.split(",")]
    except ValueError:
        parser.error("--seeds must be a comma-separated list of integers")
    if not arguments.seeds or arguments.max_steps <= 0 or arguments.step_delay < 0:
        parser.error("--seeds/max-steps must be valid and --step-delay cannot be negative")
    report = run_diagnostics(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(arguments.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["core_action_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
