"""Run the predeclared EXP-0037 frozen navigation comparison, without training."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import run_bounded_learning_pilot as runtime

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.boss_identity import _load_qualification

ROOT = runtime.ROOT
OUT = ROOT / "runs/bounded-navigation-exp0037"


def validate_report(report, checkpoint_hash, stream, seeds, contract):
    assert report["checkpoint_sha256"] == checkpoint_hash
    assert report["controller_valid"] and report["worker_restarts"] == 0
    assert not report["infrastructure_events"]
    assert report["action_contract"] == contract
    assert report["policy_seed"] == stream and report["policy_mode"] == "stochastic"
    assert report["num_instances"] == 8 and report["max_steps_per_episode"] == 500
    assert report["curriculum_profile"] == "player20"
    assert report["curriculum_start_level"] == 4 and report["curriculum_target_level"] == 5
    assert not report["natural_prefix"] and not report["trace_prefix"]
    rows = report["trained"]["results"]
    assert len(rows) == len(seeds) and sorted(row["seed"] for row in rows) == sorted(seeds)
    assert all(row["status"] in {"dead", "step_limit", "curriculum_complete"} for row in rows)
    # Health is unavailable until the boss has entered observed tiles.
    assert all(row["boss_type"] == 2 and row["initial_boss_health"] in (None, 9) for row in rows)


def aggregate(rows):
    turns = sum(row["turns"] for row in rows)
    walls = sum(row["action_outcome_counts"].get("wall_attempt", 0) for row in rows)
    return {
        "episodes": len(rows),
        "completed": sum(row["status"] == "curriculum_complete" for row in rows),
        "deaths": sum(row["status"] == "dead" for row in rows),
        "boss_damage": sum(row["boss_damage"] for row in rows),
        "episodes_with_boss_damage": sum(row["boss_damage"] > 0 for row in rows),
        "turns": turns,
        "wall_attempts": walls,
        "wall_attempt_rate": walls / turns,
        "navigation_prior_turns": sum(row["navigation_prior_turns"] for row in rows),
        "known_invalid_wall_discoveries": sum(
            row["known_invalid_wall_discoveries"] for row in rows
        ),
        "mean_unique_positions": sum(row["unique_positions"] for row in rows) / len(rows),
    }


def main():
    store = ExperimentStore(ROOT / "experiments")
    spec = store.load("EXP-0037")
    pools = {name: spec.data["evaluation"][f"{name}_seeds"] for name in ("training", "development")}
    seeds = pools["training"] + pools["development"]
    streams = spec.data["evaluation"]["policy_streams"]
    source = ROOT / spec.data["source"]["checkpoint"]
    source_hash = sha256_file(source)
    qualification = ROOT / spec.data["source"]["controller_qualification"]
    gameplay = ROOT / spec.data["source"]["gameplay_qualification"]
    control_root = ROOT / spec.data["source"]["control_root"]
    reward = ROOT / "configs/reward-death-metal-potential-v5.json"
    _load_qualification(qualification, runtime.GAME, ROOT / "mods/AutoDancer")
    assert json.loads(gameplay.read_text())["passed"]
    protocol = gameplay.parent / "protocol.json"
    for path, digest in json.loads(protocol.read_text())["hashes"].items():
        assert sha256_file(Path(path)) == digest, f"Stale mechanics check: {path}"
    prior = json.loads((control_root / "manifest.json").read_text())
    assert json.loads((control_root / "status.json").read_text())["valid"]
    changed = []
    for path, digest in prior["hashes"].items():
        if sha256_file(Path(path)) != digest:
            assert Path(path).resolve() == ROOT / "src/autodancer/training/action_contract.py", path
            changed.append(
                {"path": path, "previous_sha256": digest, "current_sha256": sha256_file(Path(path))}
            )
    reports = {}
    for stream in streams:
        report = json.loads((control_root / f"frozen-{stream}.json").read_text())
        validate_report(report, source_hash, stream, seeds, "bounded-bard-v1")
        reports[("control", stream)] = report

    OUT.mkdir(parents=True, exist_ok=False)
    runtime.OUT = OUT
    files = [
        source,
        reward,
        qualification,
        gameplay,
        protocol,
        Path(__file__).resolve(),
        Path(runtime.__file__).resolve(),
        control_root / "manifest.json",
        spec.path,
    ]
    files += [control_root / f"frozen-{stream}.json" for stream in streams]
    files += list((ROOT / "src/autodancer").rglob("*.py"))
    files += list((ROOT / "mods/AutoDancer").rglob("*.lua"))
    hashes = {str(path): sha256_file(path) for path in files}
    started = time.monotonic()
    manifest = {
        "experiment_id": "EXP-0037",
        "spec_sha256": spec.digest,
        "started_at": datetime.now(UTC).isoformat(),
        "hashes": hashes,
        "historical_source_changes": changed,
        "budget_seconds": 1200,
        "invocations": [],
    }
    atomic_json(OUT / "manifest.json", manifest)
    common = [
        "--game-dir",
        str(runtime.GAME),
        "--mod-dir",
        str(ROOT / "mods/AutoDancer"),
        "--num-instances",
        "8",
        "--curriculum-start-level",
        "4",
        "--curriculum-target-level",
        "5",
        "--curriculum-profile",
        "player20",
        "--device",
        "cuda",
        "--affinity",
        "none",
        "--steam-presence-worker",
        "0",
        "--action-contract",
        "bounded-bard-navigation-v1",
        "--reward-config",
        str(reward),
        "--policy-feedback-reward-config",
        str(reward),
        "--reward-lineage-version",
        "DeathMetalPotentialV5",
        "--experiment-id",
        "EXP-0037",
        "--experiment-arm",
        "navigation",
        "--controller-qualification",
        str(qualification),
        "--checkpoint",
        str(source),
        "--max-steps",
        "500",
        "--trained-only",
        "--policy-mode",
        "stochastic",
        "--seeds",
        ",".join(map(str, seeds)),
    ]
    try:
        store.set_status("EXP-0037", "running")
        for stream in streams:
            name = f"navigation-{stream}"
            path = OUT / "evaluations" / name / "report.json"
            invocation = runtime.execute(
                name,
                "autodancer.training.baseline",
                common + ["--trial-id", name, "--output", str(path), "--policy-seed", str(stream)],
                min(started + 1200, time.monotonic() + 600),
            )
            manifest["invocations"].append(invocation)
            report = json.loads(path.read_text())
            validate_report(report, source_hash, stream, seeds, "bounded-bard-navigation-v1")
            control = reports[("control", stream)]
            for key in (
                "evaluation_reward",
                "policy_feedback_reward",
                "game_version",
                "steam_build",
                "recurrent_state_mode",
            ):
                assert report[key] == control[key], f"Unpaired setting: {key}"
            reports[("navigation", stream)] = report
            assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items()), (
                "Source drift"
            )
            atomic_json(OUT / "manifest.json", manifest)
        summary = {}
        for arm in ("control", "navigation"):
            rows = [
                row for stream in streams for row in reports[(arm, stream)]["trained"]["results"]
            ]
            summary[arm] = {
                "all": aggregate(rows),
                **{
                    name: aggregate([row for row in rows if row["seed"] in pool])
                    for name, pool in pools.items()
                },
            }
        summary["paired"] = [
            {
                "stream": stream,
                "seed": seed,
                **{
                    arm: next(
                        row
                        for row in reports[(arm, stream)]["trained"]["results"]
                        if row["seed"] == seed
                    )
                    for arm in ("control", "navigation")
                },
            }
            for stream in streams
            for seed in seeds
        ]
        summary["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "analysis.json", summary)
        atomic_json(OUT / "status.json", {"stage": "complete", "valid": True})
        print(
            json.dumps({key: value for key, value in summary.items() if key != "paired"}),
            flush=True,
        )
    except BaseException as error:
        atomic_json(
            OUT / "status.json",
            {"stage": "failed", "valid": False, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
