"""Validate EXP-0037 artifacts and independently recompute paired outcomes."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from run_bounded_navigation_comparison import aggregate, validate_report

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json

ROOT = Path(__file__).resolve().parents[1]


def analyze(root):
    spec = ExperimentStore(ROOT / "experiments").load("EXP-0037")
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["spec_sha256"] == spec.digest
    assert json.loads((root / "status.json").read_text()) == {"stage": "complete", "valid": True}
    assert manifest["elapsed_seconds"] <= 1200
    assert len(manifest["invocations"]) == 2
    assert all(
        invocation["exit_code"] == 0 and invocation["elapsed_seconds"] <= 600
        for invocation in manifest["invocations"]
    )
    for path, digest in manifest["hashes"].items():
        current = Path(path)
        if sha256_file(current) == digest:
            continue
        # Preserve the executed runner before a formatting-only cleanup. Never
        # allow this exception for model, game, contract or evaluation code.
        assert current.resolve() == ROOT / "tools/run_bounded_navigation_comparison.py"
        snapshot = root / "executed-runner.py"
        assert sha256_file(snapshot) == digest
        assert ast.dump(ast.parse(current.read_text())) == ast.dump(ast.parse(snapshot.read_text()))
    source_hash = sha256_file(ROOT / spec.data["source"]["checkpoint"])
    controls = ROOT / spec.data["source"]["control_root"]
    pools = {name: spec.data["evaluation"][f"{name}_seeds"] for name in ("training", "development")}
    seeds = pools["training"] + pools["development"]
    reports = {}
    for stream in spec.data["evaluation"]["policy_streams"]:
        for arm, path, contract in (
            ("control", controls / f"frozen-{stream}.json", "bounded-bard-v1"),
            (
                "navigation",
                root / f"evaluations/navigation-{stream}/report.json",
                "bounded-bard-navigation-v1",
            ),
        ):
            report = json.loads(path.read_text())
            validate_report(report, source_hash, stream, seeds, contract)
            reports[(arm, stream)] = report
    result = {}
    for arm in ("control", "navigation"):
        rows = [
            row
            for (name, _), report in reports.items()
            if name == arm
            for row in report["trained"]["results"]
        ]
        result[arm] = {"all": aggregate(rows)}
        result[arm].update(
            {
                name: aggregate([row for row in rows if row["seed"] in pool])
                for name, pool in pools.items()
            }
        )
    original = json.loads((root / "analysis.json").read_text())
    assert all(original[arm] == result[arm] for arm in result)
    pairs = original["paired"]
    assert len(pairs) == 48
    assert {(pair["stream"], pair["seed"]) for pair in pairs} == {
        (stream, seed) for stream in spec.data["evaluation"]["policy_streams"] for seed in seeds
    }
    for pair in pairs:
        for arm in ("control", "navigation"):
            assert pair[arm] == next(
                row
                for row in reports[(arm, pair["stream"])]["trained"]["results"]
                if row["seed"] == pair["seed"]
            )
    result["paired_boss_damage"] = {
        "increased": sum(
            pair["navigation"]["boss_damage"] > pair["control"]["boss_damage"] for pair in pairs
        ),
        "equal": sum(
            pair["navigation"]["boss_damage"] == pair["control"]["boss_damage"] for pair in pairs
        ),
        "decreased": sum(
            pair["navigation"]["boss_damage"] < pair["control"]["boss_damage"] for pair in pairs
        ),
    }
    result["elapsed_seconds"] = manifest["elapsed_seconds"]
    result["validated"] = True
    atomic_json(root / "validated-analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/bounded-navigation-exp0037")
    print(json.dumps(analyze(parser.parse_args().run_dir.resolve()), indent=2))
