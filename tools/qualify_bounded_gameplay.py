"""Bounded Bard mechanics qualification; no learning or held-out gameplay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import probe_throw_semantics as probe

from autodancer.constants import Action, GridChannel, PlayerFeature
from autodancer.envs.vector import AutoDancerVectorEnv
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.qualify import _normalized_signature
from autodancer.live.supervisor import SupervisorConfig
from autodancer.training.action_contract import ActionContractMemory
from autodancer.training.async_collector import VersionedAsyncRolloutCollector
from autodancer.training.model import ModelConfig, RecurrentActorCritic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--host", type=Path, default=probe.ROOT / "runs/bounded-bard-dagger-verification/game-host"
    )
    args = parser.parse_args()
    root, host = args.output.resolve(), args.host.resolve()
    root.mkdir(parents=True, exist_ok=False)
    mod = host / "probe-mods/AutoDancer"
    for path in (probe.ROOT / "mods/AutoDancer/scripts").glob("*.lua"):
        assert sha256_file(path) == sha256_file(mod / "scripts" / path.name)
    bank_path = probe.ROOT / "runs/qualified-death-metal-trace-search/demonstration-bank.json"
    bank = json.loads(bank_path.read_text())
    training_seeds = [
        214001,
        214003,
        214004,
        214009,
        214012,
        214014,
        214017,
        214020,
        214031,
        214035,
        214037,
        214040,
        214041,
        214050,
        214058,
        214060,
    ]
    sources = [bank_path, Path(__file__).resolve()]
    sources += list((probe.ROOT / "mods/AutoDancer/scripts").glob("*.lua"))
    sources += [
        probe.ROOT / p
        for p in (
            "src/autodancer/training/action_contract.py",
            "src/autodancer/outcomes.py",
            "src/autodancer/live/protocol.py",
        )
    ]
    hashes = {str(p): sha256_file(p) for p in sources}
    atomic_json(
        root / "protocol.json",
        {
            "character": "Bard",
            "profile": "player20",
            "start_level": 4,
            "target_level": 5,
            "contract": "bounded-bard-v1",
            "training_seeds": training_seeds,
            "trace_seeds": [t["seed"] for t in bank["traces"]],
            "trace_repeats": 2,
            "random_actions_per_training_seed": 128,
            "hashes": hashes,
            "scope_violation": "Abort, never score or silently drop the episode",
            "training": False,
            "held_out_gameplay": False,
        },
    )
    report = {"passed": False, "cases": [], "events": {}, "outcomes": {}, "actions": 0}
    started = time.monotonic()
    supervisor = probe.PrivateSupervisor(
        SupervisorConfig(
            game_dir=host,
            mod_dir=mod,
            num_instances=1,
            startup_timeout=30,
            max_turns=500,
            affinity_policy="none",
            steam_presence_worker=0,
            curriculum_start_level=4,
            curriculum_target_level=5,
            curriculum_profile="player20",
            profile_root=root / "profiles",
            diagnostic_root=root / "diagnostics",
        )
    )
    try:
        with supervisor:
            environment = AutoDancerVectorEnv(supervisor)
            worker = environment.environments[supervisor.worker_ids[0]]
            memory = ActionContractMemory("bounded-bard-v1", 1)

            def reset(seed):
                obs, info = worker.reset(seed=seed)
                memory.reset_slot(0, obs)
                assert info["character"] == "Bard"
                assert int(obs["player"][PlayerFeature.HEALTH]) == 20
                assert int(obs["player"][PlayerFeature.TASK]) == 2
                return obs, info

            def step(obs, action, *, raw_replay=False):
                effective = memory.apply_slot(0, obs)
                if raw_replay:
                    assert obs["action_mask"][action]
                else:
                    assert effective["action_mask"][action], (
                        f"Script selected masked action {action}"
                    )
                after, reward, terminated, truncated, info = worker.step(int(action))
                memory.observe(0, obs, int(action), after, info)
                assert worker.observation_space.contains(after)
                for event in info.get("raw_events", []):
                    name = event["kind"]
                    report["events"][name] = report["events"].get(name, 0) + 1
                name = info["action_outcome"]["category"]
                report["outcomes"][name] = report["outcomes"].get(name, 0) + 1
                report["actions"] += 1
                with (root / "transitions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "seed": info["seed"],
                                "sequence": info["sequence"],
                                "action": int(action),
                                "policy_allowed": bool(effective["action_mask"][action]),
                                "inventory": after["inventory"].tolist(),
                                "status": info["episode_status"],
                                "outcome": info["action_outcome"],
                                "signature": _normalized_signature(after, info),
                                "reward": reward,
                            }
                        )
                        + "\n"
                    )
                return after, info, bool(terminated or truncated)

            def record(case):
                report["cases"].append(case)
                atomic_json(root / "result.json", report)
                print(json.dumps(case), flush=True)

            try:
                # Real engine traces, not old stored observation digests.
                for trace in bank["traces"]:
                    signatures = []
                    conflicts = []
                    for _repeat in range(2):
                        obs, info = reset(trace["seed"])
                        current = [_normalized_signature(obs, info)]
                        for index, action in enumerate(trace["action_sequence"]):
                            if not memory.apply_slot(0, obs)["action_mask"][action]:
                                if _repeat == 0:
                                    conflicts.append(index)
                            obs, info, done = step(obs, action, raw_replay=True)
                            current.append(_normalized_signature(obs, info))
                            if done:
                                break
                        assert info["episode_status"] == "curriculum_complete"
                        signatures.append(current)
                    assert signatures[0] == signatures[1], "Same-seed trace replay diverged"
                    record(
                        {
                            "case": "trace_replay",
                            "seed": trace["seed"],
                            "repeats": 2,
                            "turns": len(signatures[0]) - 1,
                            "policy_mask_conflicts": conflicts,
                            "all_actions_policy_allowed": not conflicts,
                            "passed": True,
                        }
                    )

                obs, _ = reset(92008)
                initial_bombs = int(obs["inventory"][6, 2])
                assert 0 < initial_bombs <= 3
                for _ in range(initial_bombs):
                    before_count = int(obs["inventory"][6, 2])
                    obs, info, done = step(obs, Action.BOMB)
                    assert not done and int(obs["inventory"][6, 2]) == before_count - 1
                    assert info["action_outcome"]["equipment_action"] == "bomb_placed"
                    assert np.any(obs["grid"][..., GridChannel.STATUS] & 1)
                    for _ in range(5):
                        obs, info, done = step(obs, Action.WAIT)
                        assert not done
                    assert not np.any(obs["grid"][..., GridChannel.STATUS] & 1)
                assert not obs["action_mask"][Action.BOMB]
                record(
                    {
                        "case": "bomb_consumption_detonation_exhaustion",
                        "bombs": initial_bombs,
                        "passed": True,
                    }
                )

                obs, _ = reset(92008)
                obs, _, _ = step(obs, Action.THROW)
                obs, _, _ = step(obs, Action.RIGHT)
                assert obs["inventory"][0, 0] == 0
                for _ in range(20):
                    obs, info, done = step(obs, Action.RIGHT)
                    if obs["inventory"][0, 0] or done:
                        break
                assert obs["equipment_controls"].tolist() == [1, 1, 0]
                assert obs["action_mask"][Action.THROW]
                record({"case": "dagger_retrieval", "passed": True})

                for seed in training_seeds:
                    obs, _ = reset(seed)
                    rng = np.random.default_rng(seed + 217000)
                    turns = 0
                    for _ in range(128):
                        effective = memory.apply_slot(0, obs)
                        action = int(rng.choice(np.flatnonzero(effective["action_mask"])))
                        obs, info, done = step(obs, action)
                        turns += 1
                        if done:
                            break
                    record(
                        {
                            "case": "training_seed_scope_audit",
                            "seed": seed,
                            "turns": turns,
                            "status": info["episode_status"],
                            "passed": True,
                        }
                    )
                report["passed"] = True
                # Exercise the actual collector with the new observation projection
                # and scope guard. No optimizer step or policy training is performed.
                import torch

                torch.set_num_threads(1)
                model = RecurrentActorCritic(
                    ModelConfig(
                        cell_size=16,
                        spatial_size=32,
                        hidden_size=16,
                        entity_limit=8,
                        attention_layers=1,
                        attention_heads=4,
                        map_size=0,
                    )
                )
                collector = VersionedAsyncRolloutCollector(
                    environment,
                    model,
                    device=torch.device("cpu"),
                    seed=217001,
                    action_contract="bounded-bard-v1",
                    training_seed_pool=(92008,),
                )
                try:
                    rollout = collector.collect(16)
                    assert list(rollout.actions.shape) == [16, 1]
                    assert not environment.infrastructure_events
                    record({"case": "scoped_collector", "shape": [16, 1], "passed": True})
                finally:
                    collector.close()
            finally:
                worker.close()
                handle = supervisor.workers[supervisor.worker_ids[0]]
                report["worker_restarts"] = handle.restart_count
                (root / "game.log").write_text(handle.log_path.read_text(errors="replace"))
                environment.close()
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["sources_unchanged"] = all(sha256_file(p) == hashes[str(p)] for p in sources)
        report["passed"] = bool(
            report["passed"] and report["sources_unchanged"] and report.get("worker_restarts") == 0
        )
        atomic_json(root / "result.json", report)
    assert report["sources_unchanged"] and report["worker_restarts"] == 0


if __name__ == "__main__":
    main()
