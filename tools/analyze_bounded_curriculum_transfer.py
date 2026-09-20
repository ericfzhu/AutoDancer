"""Validate EXP-0043 and compare assisted learning with direct-start transfer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from autodancer.constants import BossType
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/bounded-curriculum-transfer-exp0043"


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    turns = sum(row["turns"] for row in rows)
    return {
        "episodes": len(rows),
        "wins": sum(row["status"] == "curriculum_complete" for row in rows),
        "contact": sum(row["boss_damage"] > 0 for row in rows),
        "damage": sum(row["boss_damage"] for row in rows),
        "deaths": sum(row["status"] == "dead" for row in rows),
        "turns": turns,
        "initial_boss_health": [row["initial_boss_health"] for row in rows],
        "wall_attempt_rate": sum(
            row["action_outcome_counts"].get("wall_attempt", 0) for row in rows
        )
        / turns
        if turns
        else 0,
    }


def main():
    spec = ExperimentStore(ROOT / "experiments").load("EXP-0043").data
    manifest = read(OUT / "manifest.json")
    assert read(OUT / "status.json")["status"] == "completed"
    assert all(sha256_file(Path(p)) == h for p, h in manifest["hashes"].items())
    assert manifest["model_tensor_parity"]
    qualification = read(OUT / "qualification.json")
    assert qualification["valid"] and qualification["qualified_trace_count"] == 2
    assert not qualification["infrastructure_error"] and qualification["worker_restarts"] == 0
    preparation = read(OUT / "preparation.json")
    assert preparation["valid"] and preparation["episodes"] == 8
    assert manifest["elapsed_seconds"] <= 3600
    for invocation in manifest["invocations"]:
        assert invocation["exit_code"] == 0
        assert invocation["elapsed_seconds"] <= (1800 if invocation["stage"] == "training" else 600)
    result = {
        "valid": True,
        "preparation": preparation,
        "conditional": {},
        "elapsed_seconds": manifest["elapsed_seconds"],
        "timing": {i["stage"]: i["elapsed_seconds"] for i in manifest["invocations"]},
    }
    source = ROOT / spec["source"]["checkpoint"]
    reference = ROOT / spec["source"]["direct_control"]
    reference_report = read(reference / "frozen-215001.json")

    def report_rows(path, checkpoint, seeds, tail=None, stream=None):
        report = read(path)
        assert report["controller_valid"] and report["worker_restarts"] == 0
        for key in (
            "evaluation_reward",
            "policy_feedback_reward",
            "game_version",
            "steam_build",
            "recurrent_state_mode",
        ):
            assert report[key] == reference_report[key], key
        assert not report["infrastructure_events"]
        assert report["action_contract"] == "bounded-bard-navigation-v1"
        assert report["checkpoint_sha256"] == sha256_file(checkpoint)
        assert report["policy_mode"] == "stochastic" and report["policy_seed"] == stream
        assert report["max_steps_per_episode"] == 500 and report["num_instances"] == 8
        assert report["curriculum_profile"] == "player20"
        assert report["curriculum_start_level"] == 4 and report["curriculum_target_level"] == 5
        assert not report["natural_prefix"]
        if tail:
            prefix = report["trace_prefix"]
            assert prefix["tail_actions"] == tail and prefix["recurrent_state_mode"] == "warm"
            assert prefix["bank_sha256"] == read(OUT / "bank.json")["bank_sha256"]
            assert prefix["qualification_sha256"] == sha256_file(OUT / "qualification.json")
        else:
            assert not report["trace_prefix"]
        rows = report["trained"]["results"]
        assert sorted(r["seed"] for r in rows) == sorted(seeds)
        if tail:
            # The qualified seed92008/tail32 handoff is already past the boss.
            assert all(r["boss_type"] in (0, int(BossType.DEATH_METAL)) for r in rows)
            assert all(r["boss_type"] != 0 or (r["seed"] == 92008 and tail == 32) for r in rows)
        else:
            assert all(r["boss_type"] == int(BossType.DEATH_METAL) for r in rows)
            assert all(r["initial_boss_health"] in (None, 9) for r in rows)
        assert all(r["status"] in {"dead", "step_limit", "curriculum_complete"} for r in rows)
        return rows

    final = OUT / "training/final.pt"
    for stage in ["frozen", "curriculum"] if preparation["passed"] else ["frozen"]:
        result["conditional"][stage] = {}
        for tail in spec["evaluation"]["tails"]:
            rows = []
            for stream in spec["evaluation"]["streams"]:
                rows += report_rows(
                    OUT / "evaluations" / f"{stage}-tail{tail}-{stream}" / "report.json",
                    source if stage == "frozen" else final,
                    spec["evaluation"]["conditional_seeds"],
                    tail,
                    stream,
                )
            result["conditional"][stage][str(tail)] = {
                "all": summarize(rows),
                "by_seed": {
                    str(seed): summarize([r for r in rows if r["seed"] == seed])
                    for seed in spec["evaluation"]["conditional_seeds"]
                },
            }
    if preparation["passed"]:
        initial = torch.load(OUT / "initial-learner.pt", map_location="cpu", weights_only=False)
        declared = torch.load(source, map_location="cpu", weights_only=False)
        assert initial["model"].keys() == declared["model"].keys()
        assert all(
            torch.equal(value, declared["model"][key]) for key, value in initial["model"].items()
        )
        assert not initial["optimizer"]["state"]
        del initial, declared
        config = read(OUT / "training/config.json")
        control = read(reference / "training/config.json")
        matched = [
            "ppo",
            "architecture",
            "reward",
            "policy_feedback_reward",
            "action_contract",
            "max_turns",
            "curriculum_start_level",
            "curriculum_target_level",
            "curriculum_profile",
            "freeze_base_updates",
            "freeze_actor_updates",
            "sequence_encoding_batch_size",
            "inference_transfer_mode",
        ]
        assert all(config[k] == control[k] for k in matched)
        assert config["training_seed_pool"] == [92008, 92096]
        assert config["trace_prefix"]["tail_action_window"] == [32, 60]
        metrics = [
            json.loads(line) for line in (OUT / "training/metrics.jsonl").read_text().splitlines()
        ]
        assert len(metrics) == 16 and metrics[-1]["global_step"] == 16384
        for row in metrics:
            assert row["pre_update_logprob_max_abs_difference"] < 0.001
            assert not row["worker_restarts"] and not row["collector_recoveries_total"]
            assert all(
                math.isfinite(row[k])
                for k in ("policy_loss", "value_loss", "post_update_full_batch_approx_kl")
            )
        episodes = [
            json.loads(line) for line in (OUT / "training/episodes.jsonl").read_text().splitlines()
        ]
        assert all(r["infrastructure_valid"] and r["seed"] in [92008, 92096] for r in episodes)
        qualified = {r["seed"]: r for r in qualification["results"]}
        for episode in episodes:
            prefix = episode["natural_prefix"]
            tail = prefix["learner_tail_actions"]
            trace = qualified[episode["seed"]]
            guide_turns = len(trace["actual"]["qualified_action_sequence"]) - tail
            assert tail in (32, 60) and prefix["guide_turns"] == guide_turns
            assert not prefix["guide_transitions_in_ppo"]
            assert prefix["handoff_observation_digest"] == trace["turn_digests"][guide_turns]
        worker_times = [w for row in metrics for w in row["worker_stage_timings"]]
        fragment_seconds = sum(w["fragment_seconds"] for w in worker_times)
        result["training"] = {
            "prefix_share_worker_fragment_time": sum(
                w["prefix_total_seconds"] for w in worker_times
            )
            / fragment_seconds,
            "prefix_inference_share_worker_fragment_time": sum(
                w["prefix_inference_seconds"] for w in worker_times
            )
            / fragment_seconds,
            "prefix_environment_calls": sum(w["prefix_environment_calls"] for w in worker_times),
            "completed_episode_guide_turns": sum(
                r["natural_prefix"]["guide_turns"] for r in episodes
            ),
            "completed_episode_learner_turns": sum(r["turns"] for r in episodes),
            "episodes_by_tail": {
                str(t): sum(r["natural_prefix"]["learner_tail_actions"] == t for r in episodes)
                for t in (32, 60)
            },
            "steps": 16384,
            "updates": 16,
            "episodes": len(episodes),
            "wins": sum(r["status"] == "curriculum_complete" for r in episodes),
            "journal_boss_contact": sum(r["boss_progress"]["boss_damage"] > 0 for r in episodes),
            "journal_boss_damage": sum(r["boss_progress"]["boss_damage"] for r in episodes),
            "journal_damage_limitation": (
                "BossProgressTracker returns before reading events when TASK changes; "
                "journal damage can omit finishing hits. Use completion counts and "
                "separate frozen evaluation metrics."
            ),
            "max_full_batch_approx_kl": max(r["post_update_full_batch_approx_kl"] for r in metrics),
            "updates_above_002": sum(r["post_update_full_batch_approx_kl"] > 0.02 for r in metrics),
            "optimizer_steps": sum(r["optimizer_steps"] for r in metrics),
            "kl_early_stops": sum(r["kl_early_stop"] for r in metrics),
            "final_checkpoint_sha256": sha256_file(final),
            "matched_control_settings": matched,
        }
        pools = {
            key: spec["evaluation"][f"direct_{key}_seeds"] for key in ("training", "development")
        }
        seeds = pools["training"] + pools["development"]
        result["direct"] = {}
        for stage in ("frozen", "direct_control", "curriculum"):
            rows = []
            for stream in spec["evaluation"]["streams"]:
                if stage == "curriculum":
                    path = OUT / "evaluations" / f"curriculum-direct-{stream}" / "report.json"
                    checkpoint = final
                else:
                    path = reference / f"{'frozen' if stage == 'frozen' else 'pilot'}-{stream}.json"
                    checkpoint = (
                        ROOT / spec["source"]["original_checkpoint"]
                        if stage == "frozen"
                        else reference / "training/final.pt"
                    )
                rows += report_rows(path, checkpoint, seeds, stream=stream)
            result["direct"][stage] = {
                "all": summarize(rows),
                **{
                    pool: summarize([r for r in rows if r["seed"] in selected])
                    for pool, selected in pools.items()
                },
            }
    result["decision"] = "inconclusive"
    result["limitations"] = (
        "One optimizer trial, two success-selected historical maps, reused development "
        "maps. Curriculum changes both start distribution and training seed pool. Conditional "
        "success is not direct-start or full-game success. No promotion or final-test evaluation."
    )
    atomic_json(OUT / "validated-analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
