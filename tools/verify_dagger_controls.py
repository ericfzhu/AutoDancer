"""Verify the opt-in dagger interface in an isolated game host, without training."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import probe_throw_semantics as probe

from autodancer.constants import Action
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.supervisor import SupervisorConfig
from autodancer.training.action_contract import ActionContractMemory

OUT = probe.ROOT / "runs/dagger-controls-v1-verification"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    host = OUT / "game-host"
    mod = host / "probe-mods/AutoDancer"
    host.mkdir()
    originals = [probe.GAME / n for n in ("NecroDancer.exe", "NecroDancer.wsp", "config.json")]
    originals += list((probe.ROOT / "mods/AutoDancer/scripts").glob("*.lua"))
    originals += [
        probe.ROOT / name
        for name in (
            "src/autodancer/training/action_contract.py",
            "src/autodancer/outcomes.py",
            "src/autodancer/live/protocol.py",
            "src/autodancer/observation.py",
            "src/autodancer/training/baseline.py",
            "tools/verify_dagger_controls.py",
        )
    ]
    hashes = {str(p): sha256_file(p) for p in originals}
    for path in probe.GAME.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(path, host / path.name)
    for name in ("versions", "dlc"):
        shutil.copytree(probe.GAME / name, host / name)
    shutil.copytree(probe.ROOT / "mods/AutoDancer", mod)
    config = json.loads((probe.GAME / "config.json").read_text())
    config["wos"]["game"]["assets"]["external"]["path"] = probe.GAME.parent.as_posix()
    config["wos"]["game"]["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(config))
    cases = {
        "repeat_throw": [8, 8, 8],
        "right_only": [1],
        "throw_up": [8, 0],
        "throw_right": [8, 1],
        "throw_down": [8, 2],
        "throw_left": [8, 3],
        "double_throw_right": [8, 8, 1],
        "throw_wait_right": [8, 4, 1],
    }
    atomic_json(
        OUT / "protocol.json",
        {
            "seeds": [92008, 214001],
            "cases": cases,
            "hashes": hashes,
            "contract": "dagger-controls-v1",
            "training": False,
            "note": "Script deliberately sends redundant actions through raw engine availability",
        },
    )
    result = {"cases": []}
    started = time.monotonic()
    supervisor = probe.PrivateSupervisor(
        SupervisorConfig(
            game_dir=host,
            mod_dir=mod,
            num_instances=1,
            startup_timeout=30,
            max_turns=100,
            affinity_policy="none",
            steam_presence_worker=0,
            curriculum_start_level=4,
            curriculum_target_level=5,
            curriculum_profile="player20",
            profile_root=OUT / "profiles",
            diagnostic_root=OUT / "diagnostics",
        )
    )
    with supervisor:
        worker = supervisor.environment(supervisor.worker_ids[0])
        try:
            for seed in (92008, 214001):
                for name, actions in cases.items():
                    obs, info = worker.reset(seed=seed)
                    assert obs["equipment_controls"].tolist() == [1, 1, 0]
                    assert worker.observation_space.contains(obs)
                    memory = ActionContractMemory("dagger-controls-v1", 1)
                    memory.reset_slot(0, obs)
                    row = {"seed": seed, "case": name, "steps": []}
                    for action in actions:
                        before = obs
                        effective = memory.apply_slot(0, before)
                        armed = bool(before["equipment_controls"][2])
                        assert effective["inventory"][0, 7] == int(armed)
                        if armed:
                            assert effective["action_mask"][Action.THROW] == 0
                            assert np.array_equal(
                                effective["action_mask"][:5], before["action_mask"][:5]
                            )
                        obs, reward, terminated, truncated, info = worker.step(action)
                        assert not terminated and not truncated
                        assert worker.observation_space.contains(obs)
                        outcome = info["action_outcome"]
                        if action == Action.THROW:
                            expected = "throw_already_armed" if armed else "throw_prepared"
                            assert outcome["equipment_action"] == expected
                            assert obs["equipment_controls"].tolist() == [1, 1, 1]
                        elif armed and action < 4:
                            assert outcome["equipment_action"] == "weapon_thrown"
                            assert obs["equipment_controls"].tolist() == [1, 0, 0]
                        elif armed and action == Action.WAIT:
                            assert obs["equipment_controls"].tolist() == [1, 1, 1]
                        memory.observe(0, before, action, obs, info)
                        row["steps"].append(
                            {
                                "action": action,
                                "policy_allowed": bool(effective["action_mask"][action]),
                                "controls": obs["equipment_controls"].tolist(),
                                "outcome": outcome,
                                "reward": reward,
                            }
                        )
                    result["cases"].append(row)
                    print(json.dumps({"seed": seed, "case": name, "passed": True}), flush=True)
        finally:
            worker.close()
        handle = supervisor.workers[supervisor.worker_ids[0]]
        result["worker_restarts"] = handle.restart_count
        shutil.copy2(handle.log_path, OUT / "game.log")
    result["elapsed_seconds"] = time.monotonic() - started
    result["source_files_unchanged"] = all(sha256_file(p) == hashes[str(p)] for p in originals)
    assert result["source_files_unchanged"] and result["worker_restarts"] == 0
    result["passed"] = True
    atomic_json(OUT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
