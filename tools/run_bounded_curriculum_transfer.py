"""Execute the predeclared EXP-0043 curriculum and transfer comparison."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.boss_identity import _load_qualification
from autodancer.training.demonstrations import (
    _canonical_json,
    validate_demonstration_bank,
    validate_demonstration_sources,
    write_demonstration_bank,
)
from tools import run_stable_navigation_learning as bounded

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/bounded-curriculum-transfer-exp0043"


def derive_bank():
    original = ROOT / "runs/qualified-death-metal-trace-search/demonstration-bank.json"
    replay = ROOT / "runs/current-trace-qualification-exp0042/report.json"
    manifest = ROOT / "runs/current-trace-qualification-exp0042/manifest.json"
    bank = json.loads(original.read_text())
    report = json.loads(replay.read_text())
    validate_demonstration_bank(bank)
    validate_demonstration_sources(bank)
    assert report["bank_sha256"] == bank["bank_sha256"]
    assert not report["infrastructure_error"] and report["worker_restarts"] == 0
    for path, digest in json.loads(manifest.read_text())["hashes"].items():
        assert sha256_file(Path(path)) == digest
    accepted = {r["trace_id"] for r in report["results"] if r["valid"]}
    traces = [t for t in bank["traces"] if t["trace_id"] in accepted]
    assert sorted(t["seed"] for t in traces) == [92008, 92096]
    # This is an explicit new provenance record backed by live effective-mask
    # validation, not a relabeling of the historical source or action sequence.
    attestation = OUT / "accepted-trace-provenance.json"
    atomic_json(
        attestation,
        {
            "action_contract": "bounded-bard-navigation-v1",
            "original_bank": str(original),
            "original_bank_sha256": sha256_file(original),
            "current_mask_replay": str(replay),
            "current_mask_replay_sha256": sha256_file(replay),
            "execution_manifest": str(manifest),
            "execution_manifest_sha256": sha256_file(manifest),
            "accepted_trace_ids": sorted(accepted),
            "excluded_seeds": [92116],
            "note": "Exact historical actions; only current-mask successful replays included.",
        },
    )
    derived = {
        "schema_version": 1,
        "kind": "qualified-live-action-traces-v1",
        "sources": [
            {
                "path": str(attestation),
                "sha256": sha256_file(attestation),
                "kind": "current-contract-live-replay-attestation",
                "action_contract": "bounded-bard-navigation-v1",
                "successful_trace_count": len(traces),
            }
        ],
        "traces": traces,
    }
    derived["bank_sha256"] = hashlib.sha256(_canonical_json(derived)).hexdigest()
    write_demonstration_bank(OUT / "bank.json", derived)


def main():
    OUT.mkdir(exist_ok=False)
    bounded.OUT = OUT
    store = ExperimentStore(ROOT / "experiments")
    spec = store.load("EXP-0043").data
    source = ROOT / spec["source"]["checkpoint"]
    original = ROOT / spec["source"]["original_checkpoint"]
    initial = torch.load(source, map_location="cpu", weights_only=False)
    historical = torch.load(original, map_location="cpu", weights_only=False)
    assert initial["model"].keys() == historical["model"].keys()
    assert all(torch.equal(v, historical["model"][k]) for k, v in initial["model"].items())
    assert initial["checkpoint_metadata"].get("trace_prefix") is None
    del initial, historical
    qualification = ROOT / "runs/controller-qualification-bounded-bard-v1/qualification.json"
    reward = ROOT / "configs/reward-death-metal-potential-v5.json"
    _load_qualification(qualification, bounded.GAME, ROOT / "mods/AutoDancer")
    derive_bank()
    paths = [
        *ROOT.glob("src/autodancer/**/*.py"),
        *ROOT.glob("mods/AutoDancer/**/*.lua"),
        Path(__file__),
        ROOT / "tools/run_stable_navigation_learning.py",
        ROOT / "tools/audit_first_update.py",
        ROOT / "experiments/EXP-0043/experiment.yaml",
        source,
        original,
        qualification,
        reward,
        OUT / "bank.json",
        OUT / "accepted-trace-provenance.json",
    ]
    reference = ROOT / spec["source"]["direct_control"]
    paths += [
        reference / f"{stage}-{stream}.json"
        for stage in ("frozen", "pilot")
        for stream in spec["evaluation"]["streams"]
    ]
    hashes = {str(p): sha256_file(p) for p in paths}
    manifest = {"hashes": hashes, "model_tensor_parity": True, "invocations": []}
    atomic_json(OUT / "manifest.json", manifest)
    store.set_status("EXP-0043", "running")
    started = time.monotonic()
    deadline = started + 3600

    def execute(name, module, args, cap=600, training=False):
        result = bounded.execute(
            name, module, args, min(deadline, time.monotonic() + cap), training=training
        )
        manifest["invocations"].append(result)
        atomic_json(OUT / "manifest.json", manifest)
        assert all(sha256_file(Path(p)) == h for p, h in hashes.items()), "Source drift"

    common = [
        "--game-dir",
        str(bounded.GAME),
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
        "--reward-lineage-version",
        "DeathMetalPotentialV5",
        "--policy-feedback-reward-config",
        str(reward),
        "--experiment-id",
        "EXP-0043",
        "--controller-qualification",
        str(qualification),
    ]
    prefix = [
        "--trace-prefix-bank",
        str(OUT / "bank.json"),
        "--trace-prefix-qualification",
        str(OUT / "qualification.json"),
        "--trace-prefix-recurrent-state",
        "warm",
    ]

    def evaluate(stage, checkpoint, tail, stream):
        name = f"{stage}-tail{tail}-{stream}" if tail else f"{stage}-direct-{stream}"
        path = OUT / "evaluations" / name / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        seeds = (
            spec["evaluation"]["conditional_seeds"]
            if tail
            else (
                spec["evaluation"]["direct_training_seeds"]
                + spec["evaluation"]["direct_development_seeds"]
            )
        )
        args = common + [
            "--experiment-arm",
            stage,
            "--trial-id",
            name,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(path),
            "--max-steps",
            "500",
            "--trained-only",
            "--policy-mode",
            "stochastic",
            "--policy-seed",
            str(stream),
            "--seeds",
            ",".join(map(str, seeds)),
        ]
        if tail:
            args += prefix + ["--trace-prefix-tail-actions", str(tail)]
        if stage == "frozen":
            args += ["--source-reference"]
        execute(name, "autodancer.training.baseline", args)
        report = json.loads(path.read_text())
        assert report["controller_valid"] and report["worker_restarts"] == 0
        assert not report["infrastructure_events"]
        assert report["checkpoint_sha256"] == sha256_file(checkpoint)
        assert report["action_contract"] == "bounded-bard-navigation-v1"
        rows = report["trained"]["results"]
        assert sorted(r["seed"] for r in rows) == sorted(seeds)
        return sum(r["status"] == "curriculum_complete" for r in rows)

    try:
        execute(
            "qualification",
            "autodancer.training.demonstration_replay",
            [
                "qualify",
                "--game-dir",
                str(bounded.GAME),
                "--mod-dir",
                str(ROOT / "mods/AutoDancer"),
                "--bank",
                str(OUT / "bank.json"),
                "--output",
                str(OUT / "qualification.json"),
                "--num-instances",
                "2",
                "--recurrent-output",
                str(OUT / "recurrent.npz"),
                "--action-contract",
                "bounded-bard-navigation-v1",
                "--policy-feedback-reward-config",
                str(reward),
            ],
        )
        report = json.loads((OUT / "qualification.json").read_text())
        assert report["valid"] and report["qualified_trace_count"] == 2
        for path in (OUT / "qualification.json", OUT / "recurrent.npz"):
            hashes[str(path)] = sha256_file(path)
        successes = sum(
            evaluate("frozen", source, tail, stream)
            for tail in spec["evaluation"]["tails"]
            for stream in spec["evaluation"]["streams"]
        )
        atomic_json(
            OUT / "preparation.json",
            {"valid": True, "successes": successes, "episodes": 8, "passed": successes > 0},
        )
        if not successes:
            atomic_json(OUT / "status.json", {"status": "completed", "gate_passed": False})
            return
        execute(
            "training",
            "tools.run_stable_navigation_learning",
            ["--train-child"]
            + common
            + prefix
            + [
                "--trace-prefix-tail-actions",
                "32",
                "--trace-prefix-tail-window",
                "32,60",
                "--experiment-arm",
                "curriculum",
                "--trial-id",
                "seed-217001",
                "--run-dir",
                str(OUT / "training"),
                "--fine-tune-from",
                str(source),
                "--architecture",
                "8",
                "--seed",
                "217001",
                "--total-steps",
                "16384",
                "--max-turns",
                "500",
                "--rollout-length",
                "128",
                "--sequence-length",
                "32",
                "--training-seed-pool",
                "92008,92096",
                "--training-level-distribution-version",
                "bounded-two-trace-window-v1",
                "--sequence-encoding-batch-size",
                "16",
                "--inference-transfer-mode",
                "legacy",
                "--checkpoint-interval",
                "4096",
                "--evaluation-interval",
                "0",
                "--freeze-base-updates",
                "0",
                "--freeze-actor-updates",
                "0",
                "--learning-rate",
                "0.000003",
                "--target-kl",
                "0.02",
                "--gamma",
                "0.99",
                "--gae-lambda",
                "0.95",
            ],
            cap=1800,
            training=True,
        )
        final = OUT / "training/final.pt"
        for tail in spec["evaluation"]["tails"]:
            for stream in spec["evaluation"]["streams"]:
                evaluate("curriculum", final, tail, stream)
        for stream in spec["evaluation"]["streams"]:
            evaluate("curriculum", final, None, stream)
        atomic_json(OUT / "status.json", {"status": "completed", "gate_passed": True})
    except BaseException as error:
        atomic_json(OUT / "status.json", {"status": "failed", "error": repr(error)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
