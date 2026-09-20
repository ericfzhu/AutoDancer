"""Bounded random-action smoke of native tasks, observations and reward contracts.

No teacher, checkpoint, demonstration, navigation heuristic or PPO update.
This checks wiring, not learnability or comprehensive mechanics coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from autodancer.curriculum import load_curriculum_mixture
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.observation import observation_space
from autodancer.rewards import RewardTracker, load_reward_config
from autodancer.training.action_contract import apply_action_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, default=Path("mods/AutoDancer"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--seed", type=int, default=230001)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    args.output.mkdir(parents=True)
    sparse = load_reward_config("configs/reward-trial-sparse-v5.json")
    shaped = load_reward_config("configs/reward-trial-exploration-v5.json")
    space = observation_space()
    report = {
        "passed": False,
        "training": False,
        "action_contract": "bounded-bard-v1",
        "seed": args.seed,
        "steps_per_task": args.steps,
        "tasks": [],
        "hashes": {},
    }
    paths = [
        Path(__file__),
        *Path("src/autodancer").rglob("*.py"),
        *Path("configs").glob("*trial*.json"),
        *Path("configs").glob("task-*-v1.json"),
        *args.mod_dir.rglob("*.lua"),
        args.mod_dir / "mod.json",
    ]
    report["hashes"] = {str(path): sha256_file(path) for path in paths}
    try:
        for task_index, name in enumerate(("first-floor", "boss-floor", "zone-one")):
            spec = load_curriculum_mixture(f"configs/task-{name}-v1.json")[0].spec
            config = SupervisorConfig(
                game_dir=args.game_dir,
                mod_dir=args.mod_dir,
                num_instances=1,
                max_turns=64,
                reward_config=shaped,
                affinity_policy="none",
                steam_presence_worker=0,
                curriculum_commands_enabled=spec.start_level > 1,
                curriculum_start_level=spec.start_level,
                curriculum_target_level=spec.target_level,
                curriculum_profile=spec.profile,
                profile_root=args.output / name / "profiles",
                diagnostic_root=args.output / name / "diagnostics",
            )
            result = {
                "task": spec.as_dict(),
                "steps": 0,
                "resets": 0,
                "statuses": {},
                "reward_components": {},
                "nonzero_rewards": 0,
            }
            report["tasks"].append(result)
            statuses, components = Counter(), Counter()
            rng = np.random.default_rng(args.seed + task_index)
            reference = RewardTracker(sparse)
            with AutoDancerSupervisor(config) as supervisor:
                env = supervisor.environment(supervisor.worker_ids[0])
                try:
                    terminated = truncated = False
                    for step in range(args.steps):
                        if step == 0 or terminated or truncated:
                            obs, info = env.reset(seed=args.seed + task_index * 1000 + step)
                            reference.reset(obs, info)
                            result["resets"] += 1
                        if not space.contains(obs):
                            raise AssertionError("Raw observation outside declared space")
                        policy_obs = apply_action_contract(obs, "bounded-bard-v1")
                        if not space.contains(policy_obs):
                            raise AssertionError("Policy observation outside declared space")
                        legal = np.flatnonzero(policy_obs["action_mask"])
                        action = int(rng.choice(legal))
                        obs, reward, terminated, truncated, info = env.step(action)
                        sparse_reward, _ = reference.score(
                            obs,
                            info,
                            info["raw_events"],
                            terminated=terminated,
                            truncated=truncated,
                        )
                        if not np.isfinite(reward) or not np.isclose(
                            reward, sum(info["reward_components"].values())
                        ):
                            raise AssertionError("Invalid reward decomposition")
                        if not np.isclose(sparse_reward, info["extrinsic_reward"]):
                            raise AssertionError("Sparse and shaped task objectives differ")
                        # Validate the final observation too, including terminal loadouts.
                        apply_action_contract(obs, "bounded-bard-v1")
                        if not space.contains(obs):
                            raise AssertionError("Next observation outside declared space")
                        components.update(info["reward_components"])
                        result["steps"] += 1
                        result["nonzero_rewards"] += int(reward != 0)
                        if terminated or truncated:
                            statuses[info["episode_status"]] += 1
                        result["statuses"] = dict(statuses)
                        result["reward_components"] = dict(components)
                finally:
                    env.close()
            atomic_json(args.output / "result.json", report)
            print(f"{name}: {result['steps']} transitions, {dict(statuses)}", flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        atomic_json(args.output / "result.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
