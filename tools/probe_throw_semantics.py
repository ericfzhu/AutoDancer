"""Bounded THROW input diagnostic in a private game host; no production mod edits."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig

ROOT = Path(__file__).resolve().parents[1]
GAME = Path(r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64")
OUT = ROOT / "runs/throw-semantics/live"
PROBE = r"""
        local probeWeaponID = slots.weapon and slots.weapon[1]
        local probeWeapon = probeWeaponID and Entities.getEntityByID(probeWeaponID)
        print("AUTODANCER_THROW:" .. string.format(
            '{"sequence":%d,"x":%d,"y":%d,"weapon_id":%d,"weapon":%s,"activable":%s,"active":%s,"throwable":%s,"reloadable":%s,"throw_mask":%d}',
            sequence, player.position.x, player.position.y, probeWeaponID or 0,
            jsonEscape(probeWeapon and probeWeapon.name or ""),
            tostring(probeWeapon ~= nil and probeWeapon.itemActivable ~= nil),
            tostring(probeWeapon ~= nil and probeWeapon.itemActivable ~= nil
                and probeWeapon.itemActivable.active == true),
            tostring(probeWeapon ~= nil and probeWeapon.weaponThrowable ~= nil),
            tostring(probeWeapon ~= nil and probeWeapon.weaponReloadable ~= nil), mask[9]))
"""


class PrivateSupervisor(AutoDancerSupervisor):
    def _validate_installation(self):
        # The normal supervisor deploys to shared LOCALAPPDATA. This private host
        # loads only its own probe-mods directory, as in the engine-pacing diagnostic.
        for path in (
            self.config.executable,
            self.config.game_dir / "autodancer_native.dll",
            self.config.mod_dir / "scripts/Bridge.lua",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)


def snapshot(obs, info):
    return {
        "sequence": int(obs["player"][8]),
        "position": obs["player"][4:6].tolist(),
        "weapon": obs["inventory"][0].tolist(),
        "action_mask": obs["action_mask"].tolist(),
        "status": info.get("episode_status"),
        "action_outcome": info.get("action_outcome"),
        "raw_events": info.get("raw_events"),
    }


def validate_and_summarize(result):
    groups = []
    for sample in result["probe_samples"]:
        if sample["sequence"] == 0:
            groups.append([])
        groups[-1].append(sample)
    assert len(groups) == len(result["cases"]), "Probe/reset coverage mismatch"
    rows = []
    for case, samples in zip(result["cases"], groups, strict=True):
        by_sequence = {sample["sequence"]: sample for sample in samples}
        initial = by_sequence[case["reset"]["sequence"]]
        assert initial["weapon"] == "WeaponDagger" and not initial["active"]
        assert initial["throwable"] and initial["activable"]
        steps = case["steps"]
        states = [by_sequence[step["sequence"]] for step in steps]
        if case["name"] == "repeat_throw":
            assert all(state["active"] and state["throw_mask"] for state in states)
            assert all(step["changed_elements"]["inventory"] == 0 for step in steps)
            assert all(step["changed_elements"]["grid"] == 0 for step in steps)
        elif case["name"] == "right_only":
            assert states[-1]["weapon"] == "WeaponDagger" and not states[-1]["active"]
            assert steps[-1]["position"] != case["reset"]["position"]
        else:
            assert states[0]["active"]
            assert states[-1]["weapon_id"] == 0 and states[-1]["throw_mask"] == 0
            assert steps[-1]["position"] == case["reset"]["position"]
        rows.append(
            {
                "seed": case["seed"],
                "case": case["name"],
                "actions": [step["action"] for step in steps],
                "active": [state["active"] for state in states],
                "weapons": [state["weapon"] for state in states],
                "positions": [step["position"] for step in steps],
                "outcomes": [step["action_outcome"]["category"] for step in steps],
                "inventory_changed": [step["changed_elements"]["inventory"] for step in steps],
            }
        )
    assert result["worker_restarts"] == 0 and result["original_files_unchanged"]
    return {
        "validated": True,
        "cases": rows,
        "actions": sum(len(c["steps"]) for c in result["cases"]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    host = OUT / "game-host"
    mod = host / "probe-mods/AutoDancer"
    host.mkdir()
    original_files = [GAME / n for n in ("NecroDancer.exe", "NecroDancer.wsp", "config.json")]
    original_files += list((ROOT / "mods/AutoDancer/scripts").glob("*.lua"))
    hashes = {str(p): sha256_file(p) for p in original_files}
    for path in GAME.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(path, host / path.name)
    for name in ("versions", "dlc"):
        shutil.copytree(GAME / name, host / name)
    shutil.copytree(ROOT / "mods/AutoDancer", mod)
    config = json.loads((GAME / "config.json").read_text())
    config["wos"]["game"]["assets"]["external"]["path"] = GAME.parent.as_posix()
    config["wos"]["game"]["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(config))
    entry = mod / "scripts/AutoDancer.lua"
    source = entry.read_text()
    marker = "        result.map_bounds = currentMapBounds()"
    assert source.count(marker) == 1
    entry.write_text(source.replace(marker, PROBE + marker))
    cases = {
        "repeat_throw": [8, 8, 8, 8, 8, 8],
        "right_only": [1],
        "throw_up": [8, 0],
        "throw_right": [8, 1],
        "throw_down": [8, 2],
        "throw_left": [8, 3],
        "double_throw_right": [8, 8, 1],
        "throw_wait_right": [8, 4, 1],
    }
    result = {"cases": [], "original_hashes": hashes, "seeds": [92008, 214001]}
    atomic_json(
        OUT / "protocol.json",
        {
            "purpose": "Distinguish THROW activation, directional execution, and mask availability",
            "seeds": result["seeds"],
            "cases": cases,
            "original_hashes": hashes,
            "modification": "Private observation logging only; engine input mapping unchanged",
        },
    )
    started = time.monotonic()
    supervisor = PrivateSupervisor(
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
    try:
        with supervisor:
            worker = supervisor.environment(supervisor.worker_ids[0])
            try:
                for seed in result["seeds"]:
                    for name, actions in cases.items():
                        obs, info = worker.reset(seed=seed)
                        case = {
                            "seed": seed,
                            "name": name,
                            "reset": snapshot(obs, info),
                            "steps": [],
                        }
                        result["cases"].append(case)
                        for action in actions:
                            before = {k: v.copy() for k, v in obs.items()}
                            allowed = bool(obs["action_mask"][action])
                            if not allowed:
                                case["steps"].append({"action": action, "masked": True})
                                break
                            obs, reward, terminated, truncated, info = worker.step(action)
                            changed = {
                                k: int(np.count_nonzero(v != before[k])) for k, v in obs.items()
                            }
                            case["steps"].append(
                                {
                                    "action": action,
                                    "reward": reward,
                                    "changed_elements": changed,
                                    **snapshot(obs, info),
                                }
                            )
                            if terminated or truncated:
                                break
                        print(
                            json.dumps({"seed": seed, "case": name, "steps": len(case["steps"])}),
                            flush=True,
                        )
            finally:
                worker.close()
            handle = supervisor.workers[supervisor.worker_ids[0]]
            result["worker_restarts"] = handle.restart_count
            time.sleep(0.1)
            log = handle.log_path.read_text(encoding="utf-8", errors="replace")
            (OUT / "game.log").write_text(log, encoding="utf-8")
            result["probe_samples"] = [
                json.JSONDecoder().raw_decode(line.split("AUTODANCER_THROW:", 1)[1])[0]
                for line in log.splitlines()
                if "AUTODANCER_THROW:{" in line
            ]
            assert result["probe_samples"], "Private probe was not loaded"
            assert handle.restart_count == 0
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["original_files_unchanged"] = all(
            sha256_file(Path(p)) == h for p, h in hashes.items()
        )
        atomic_json(OUT / "result.json", result)
    assert result["original_files_unchanged"]
    atomic_json(OUT / "summary.json", validate_and_summarize(result))
    print(
        json.dumps({"seconds": result["elapsed_seconds"], "samples": len(result["probe_samples"])})
    )


if __name__ == "__main__":
    main()
