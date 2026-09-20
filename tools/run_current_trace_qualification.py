"""Bounded, hash-checked current-contract replay diagnostic (EXP-0042)."""

import json
import time
from pathlib import Path

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.boss_identity import _load_qualification
from tools import run_stable_navigation_learning as bounded

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/current-trace-qualification-exp0042"


def main():
    OUT.mkdir(exist_ok=False)
    bounded.OUT = OUT
    qualification = ROOT / "runs/controller-qualification-bounded-bard-v1/qualification.json"
    _load_qualification(qualification, bounded.GAME, ROOT / "mods/AutoDancer")
    paths = [
        *ROOT.glob("src/autodancer/**/*.py"),
        *ROOT.glob("mods/AutoDancer/**/*.lua"),
        Path(__file__),
        ROOT / "tools/run_stable_navigation_learning.py",
        qualification,
        ROOT / "experiments/EXP-0042/experiment.yaml",
        ROOT / "configs/reward-death-metal-potential-v5.json",
        ROOT / "runs/qualified-death-metal-trace-search/demonstration-bank.json",
    ]
    hashes = {str(p): sha256_file(p) for p in paths}
    atomic_json(OUT / "manifest.json", {"hashes": hashes})
    store = ExperimentStore(ROOT / "experiments")
    store.set_status("EXP-0042", "running")
    started = time.monotonic()
    error = ""
    try:
        bounded.execute(
            "replay",
            "autodancer.training.demonstration_replay",
            [
                "qualify",
                "--game-dir",
                str(bounded.GAME),
                "--mod-dir",
                str(ROOT / "mods/AutoDancer"),
                "--bank",
                str(ROOT / "runs/qualified-death-metal-trace-search/demonstration-bank.json"),
                "--output",
                str(OUT / "report.json"),
                "--num-instances",
                "3",
                "--recurrent-output",
                str(OUT / "recurrent.npz"),
                "--action-contract",
                "bounded-bard-navigation-v1",
                "--policy-feedback-reward-config",
                str(ROOT / "configs/reward-death-metal-potential-v5.json"),
            ],
            started + 600,
        )
    except RuntimeError as exc:
        error = str(exc)
    assert all(sha256_file(Path(p)) == h for p, h in hashes.items())
    report = json.loads((OUT / "report.json").read_text())
    assert not report["infrastructure_error"] and report["worker_restarts"] == 0
    assert len(report["results"]) == 3
    summary = {
        "execution_valid": True,
        "elapsed_seconds": time.monotonic() - started,
        "qualification_passed": report["valid"],
        "expected_nonzero_exit": error,
        "traces": [
            {
                "seed": r["seed"],
                "valid": r["valid"],
                "error": r["error"],
                "turns": r["actual"]["turns"],
            }
            for r in report["results"]
        ],
    }
    atomic_json(OUT / "validated-analysis.json", summary)
    atomic_json(
        OUT / "status.json", {"status": "completed", "qualification_passed": report["valid"]}
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
