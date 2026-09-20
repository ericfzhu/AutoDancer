"""Validate and summarize EXP-0041 without selecting or promoting a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    spec = ExperimentStore(Path("experiments")).load("EXP-0041")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["spec_sha256"] == spec.digest
    status = json.loads((root / "status.json").read_text())
    assert status == {"stage": "complete", "valid": True}
    assert manifest["elapsed_seconds"] <= 2400
    for name, digest in manifest["hashes"].items():
        assert sha256_file(Path(name)) == digest, f"Source drift: {name}"
    metrics = [
        json.loads(line) for line in (root / "training/metrics.jsonl").read_text().splitlines()
    ]
    assert [row["global_step"] for row in metrics] == list(range(1024, 16385, 1024))
    assert metrics[-1]["updates"] == 16
    for row in metrics:
        assert row["worker_restarts"] == row["collector_recoveries_total"] == 0
        assert all(
            math.isfinite(row[key])
            for key in ("policy_loss", "value_loss", "entropy", "gradient_norm_preclip")
        )
    config = json.loads((root / "training/config.json").read_text())
    assert config["action_contract"] == "bounded-bard-navigation-v1"
    assert config["max_turns"] == 500
    assert config["freeze_base_updates"] == config["freeze_actor_updates"] == 0
    assert config["training_seed_pool"] == spec.data["training"]["seed_pool"]
    previous_config_path = Path("runs/bounded-navigation-learning-exp0038/training/config.json")
    previous_config = json.loads(previous_config_path.read_text())
    matched_settings = (
        "architecture",
        "reward",
        "policy_feedback_reward",
        "training_seed_pool",
        "initialized_from",
        "max_turns",
        "curriculum_start_level",
        "curriculum_target_level",
        "curriculum_profile",
        "freeze_base_updates",
        "freeze_actor_updates",
        "sequence_encoding_batch_size",
        "inference_transfer_mode",
    )
    for key in matched_settings:
        assert config[key] == previous_config[key], f"Unmatched training setting: {key}"
    expected_ppo = {**previous_config["ppo"], "learning_rate": 0.000003, "target_kl": 0.02}
    assert config["ppo"] == expected_ppo
    for row in metrics:
        assert row["pre_update_logprob_max_abs_difference"] < 0.001
        assert math.isfinite(row["post_update_full_batch_approx_kl"])
    evaluation = spec.data["evaluation"]
    pools = {
        "training": evaluation["training_seeds"],
        "development": evaluation["development_seeds"],
    }
    expected_seeds = sorted(pools["training"] + pools["development"])
    analysis = {"valid": True, "experiment_id": "EXP-0041", "pools": {}, "timing": {}, "paired": []}
    reports = {}
    source = Path(spec.data["source"]["checkpoint"])
    for stage in ("frozen", "pilot"):
        expected_checkpoint = source if stage == "frozen" else root / "training/final.pt"
        rows = []
        for stream in evaluation["policy_streams"]:
            report = json.loads((root / f"{stage}-{stream}.json").read_text())
            if stage == "frozen":
                original = (
                    Path(spec.data["source"]["frozen_reference"])
                    / "evaluations"
                    / f"navigation-{stream}"
                    / "report.json"
                )
                assert report == json.loads(original.read_text())
            else:
                before_report = json.loads((root / f"frozen-{stream}.json").read_text())
                for key in (
                    "evaluation_reward",
                    "policy_feedback_reward",
                    "game_version",
                    "steam_build",
                    "recurrent_state_mode",
                ):
                    assert report[key] == before_report[key], f"Unmatched evaluation setting: {key}"
            assert report["controller_valid"] and not report["worker_restarts"]
            assert not report["infrastructure_events"]
            assert report["num_instances"] == 8 and report["max_steps_per_episode"] == 500
            assert report["curriculum_profile"] == "player20"
            assert report["curriculum_start_level"] == 4
            assert report["curriculum_target_level"] == 5
            assert not report["natural_prefix"] and not report["trace_prefix"]
            assert report["checkpoint_sha256"] == sha256_file(expected_checkpoint)
            assert report["action_contract"] == "bounded-bard-navigation-v1"
            assert report["policy_seed"] == stream and report["policy_mode"] == "stochastic"
            entries = report["trained"]["results"]
            assert sorted(row["seed"] for row in entries) == expected_seeds
            assert all(
                row["status"] in {"dead", "step_limit", "curriculum_complete"} for row in entries
            )
            reports[stage, stream] = {row["seed"]: row for row in entries}
            rows.extend(entries)
        analysis[stage] = {}
        for pool, seeds in pools.items():
            selected = [row for row in rows if row["seed"] in seeds]
            completed = sum(row["status"] == "curriculum_complete" for row in selected)
            analysis[stage][pool] = {
                "episodes": len(selected),
                "completed": completed,
                "completion_rate": completed / len(selected),
                "deaths": sum(row["status"] == "dead" for row in selected),
                "mean_boss_damage": sum(row["boss_damage"] for row in selected) / len(selected),
                "mean_turns": sum(row["turns"] for row in selected) / len(selected),
                "episodes_with_boss_damage": sum(row["boss_damage"] > 0 for row in selected),
                "boss_damage": sum(row["boss_damage"] for row in selected),
                "wall_attempt_rate": sum(
                    row["action_outcome_counts"].get("wall_attempt", 0) for row in selected
                )
                / sum(row["turns"] for row in selected),
            }
    for pool, seeds in pools.items():
        before, after = analysis["frozen"][pool], analysis["pilot"][pool]
        analysis["pools"][pool] = {
            "completion_change": after["completion_rate"] - before["completion_rate"],
            "boss_damage_change": after["mean_boss_damage"] - before["mean_boss_damage"],
        }
        for seed in seeds:
            for stream in evaluation["policy_streams"]:
                before = reports["frozen", stream][seed]
                after = reports["pilot", stream][seed]
                analysis["paired"].append(
                    {
                        "pool": pool,
                        "seed": seed,
                        "stream": stream,
                        "before_status": before["status"],
                        "after_status": after["status"],
                        "before_boss_damage": before["boss_damage"],
                        "after_boss_damage": after["boss_damage"],
                    }
                )
    for invocation in manifest["invocations"]:
        assert invocation["exit_code"] == 0
        if invocation["stage"].startswith("frozen-"):
            assert (
                sha256_file(Path(invocation["reused_report"])) == invocation["reused_report_sha256"]
            )
        elif invocation["stage"].startswith("pilot-"):
            assert invocation["elapsed_seconds"] <= 600
        analysis["timing"][invocation["stage"]] = invocation["elapsed_seconds"]
    assert analysis["timing"]["training"] <= 1200
    analysis["elapsed_seconds"] = manifest["elapsed_seconds"]
    analysis["training"] = {
        "steps": 16384,
        "updates": 16,
        "optimizer_seed": 217001,
        "final_checkpoint_sha256": sha256_file(root / "training/final.pt"),
        "first_entropy": metrics[0]["entropy"],
        "last_entropy": metrics[-1]["entropy"],
        "max_gradient_norm": max(row["gradient_norm_preclip"] for row in metrics),
        "matched_previous_settings": list(matched_settings),
        "previous_config_sha256": sha256_file(previous_config_path),
    }
    episodes = [
        json.loads(line) for line in (root / "training/episodes.jsonl").read_text().splitlines()
    ]
    assert all(row["infrastructure_valid"] for row in episodes)
    assert all(row["seed"] in pools["training"] for row in episodes)
    analysis["training"].update(
        {
            "completed_episodes": len(episodes),
            "wins": sum(row["status"] == "curriculum_complete" for row in episodes),
            "episodes_with_boss_damage": sum(
                row["boss_progress"]["boss_damage"] > 0 for row in episodes
            ),
            "boss_damage": sum(row["boss_progress"]["boss_damage"] for row in episodes),
        }
    )
    analysis["replication_signal"] = (
        analysis["pools"]["development"]["completion_change"] >= 0.10
        and analysis["pools"]["training"]["completion_change"] >= 0
    )
    analysis["retention_screen_passed"] = (
        analysis["pilot"]["development"]["boss_damage"] >= 2
        and sum(analysis["pilot"][pool]["episodes_with_boss_damage"] for pool in pools) >= 8
    )
    analysis["update_stability"] = {
        "maximum_full_batch_approx_kl": max(
            row["post_update_full_batch_approx_kl"] for row in metrics
        ),
        "first_full_batch_approx_kl": metrics[0]["post_update_full_batch_approx_kl"],
        "updates_above_002": sum(row["post_update_full_batch_approx_kl"] > 0.02 for row in metrics),
        "kl_early_stops": sum(row["kl_early_stop"] for row in metrics),
    }
    analysis["decision"] = "inconclusive"
    analysis["limitation"] = (
        "Single optimizer seed and short transition budget; no promotion or final-test evaluation."
    )
    atomic_json(root / "validated-analysis.json", analysis)
    print(json.dumps({k: v for k, v in analysis.items() if k != "paired"}, indent=2))


if __name__ == "__main__":
    main()
