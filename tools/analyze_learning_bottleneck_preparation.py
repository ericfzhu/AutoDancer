"""Validate and summarize the frozen-policy EXP-0035 preparation artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/learning-bottleneck-exp0035"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main() -> None:
    manifest = read("manifest.json")
    pools = read("pools.json")
    spec = ExperimentStore(ROOT / "experiments").load("EXP-0035")
    assert spec.digest == manifest["spec_sha256"]
    assert sha256_file(Path(manifest["source_checkpoint"])) == manifest["source_sha256"]
    assert sha256_file(OUT / "identities.json") == pools["identity_sha256"]
    identities = read("identities.json")
    assert identities["controller_valid"] and not identities["infrastructure_events"]
    selected = sorted(r["seed"] for r in identities["results"] if r["boss_name"] == "DEATH_METAL")
    assert pools["training"] == selected[:16]
    assert pools["development"] == selected[16:24]
    assert pools["final_test"] == selected[24:40]
    all_episodes = []
    evidence = {}
    executions = [(215001, "stochastic", "215001"), (215002, "stochastic", "215002")]
    if (OUT / "trace-control-deterministic.json").exists():
        executions.append((0, "deterministic", "deterministic"))
    for kind in ("direct", "trace-control"):
        seeds = pools["historically_familiar"]
        if kind == "direct":
            seeds = pools["training"] + pools["development"] + seeds
        assert not set(seeds) & set(pools["final_test"])
        for stream, mode, label in executions:
            name = f"{kind}-{label}.json"
            report = read(name)
            assert report["seeds"] == seeds
            assert report["checkpoint_sha256"] == manifest["source_sha256"]
            assert report["controller_valid"] and report["worker_restarts"] == 0
            assert not report["infrastructure_events"]
            assert report["policy_mode"] == mode and report["policy_seed"] == stream
            assert report["policy_feedback_matches_checkpoint"]
            assert report["curriculum_profile"] == "player20"
            assert report["curriculum_start_level"] == 4 and report["curriculum_target_level"] == 5
            assert report["max_steps_per_episode"] == 500
            assert report["action_contract"] == "map-navigation-prior-v1"
            if kind == "trace-control":
                assert report["trace_prefix"]["tail_actions"] == 60
                assert report["trace_prefix"]["recurrent_state_mode"] == "warm"
            else:
                assert report["trace_prefix"] is None and report["natural_prefix"] is None
            episodes = report["trained"]["results"]
            assert Counter(e["seed"] for e in episodes) == Counter(seeds)
            condition = kind if mode == "stochastic" else f"{kind}-deterministic"
            all_episodes.extend(
                {**e, "condition": condition, "policy_stream": stream} for e in episodes
            )
            evidence[name] = sha256_file(OUT / name)
    rows = []
    for kind in dict.fromkeys(e["condition"] for e in all_episodes):
        for pool in ("training", "development", "historically_familiar"):
            episodes = [
                e for e in all_episodes if e["condition"] == kind and e["seed"] in pools[pool]
            ]
            if not episodes:
                continue
            rows.append(
                {
                    "condition": kind,
                    "pool": pool,
                    "episodes": len(episodes),
                    "completions": sum(e["status"] == "curriculum_complete" for e in episodes),
                    "statuses": dict(Counter(e["status"] for e in episodes)),
                    "mean_turns": mean(e["turns"] for e in episodes),
                    "mean_player_damage": mean(e["player_damage"] for e in episodes),
                    "mean_boss_damage": mean(e["boss_damage"] for e in episodes),
                    "mean_return": mean(e["episode_return"] for e in episodes),
                    "reported_initial_boss_health": sorted(
                        {
                            e["initial_boss_health"]
                            for e in episodes
                            if e["initial_boss_health"] is not None
                        }
                    ),
                    "guide_turns": sum(
                        (e.get("natural_prefix") or {}).get("guide_turns", 0) for e in episodes
                    ),
                    "action_outcome_counts": dict(
                        sum((Counter(e["action_outcome_counts"]) for e in episodes), Counter())
                    ),
                    "episodes_with_boss_damage": sum(e["boss_damage"] > 0 for e in episodes),
                    "streams": {
                        str(stream): {
                            "episodes": sum(e["policy_stream"] == stream for e in episodes),
                            "completions": sum(
                                e["policy_stream"] == stream
                                and e["status"] == "curriculum_complete"
                                for e in episodes
                            ),
                        }
                        for stream in sorted({e["policy_stream"] for e in episodes})
                    },
                }
            )
    result = {
        "validated": True,
        "rows": rows,
        "episodes": all_episodes,
        "evidence_sha256": evidence,
        "gate": read("preparation-result.json"),
        "invocation_seconds": {r["name"]: r["seconds"] for r in manifest["invocations"]},
        "limitations": [
            "Frozen-policy screen, not a learning curve or comparison of trained arms.",
            "Trace handoffs differ in game state, remaining horizon, "
            "and recurrent context from direct resets.",
            "Two action-sampling streams are not independent optimizer replications.",
            "New pools are excluded from this experiment's training; "
            "no exhaustive ancestral transition audit is claimed.",
            "Per-invocation time includes startup and cleanup; "
            "no pure engine or PPO timing fraction can be inferred.",
        ],
    }
    atomic_json(OUT / "analysis.json", result)
    print(
        json.dumps(
            {"validated": True, "rows": rows, "gate_passed": result["gate"]["gate_passed"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
