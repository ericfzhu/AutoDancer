"""Execute the immutable EXP-0036 pilot with hard budgets and no policy selection."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.boss_identity import _load_qualification

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/bounded-learning-exp0036"
GAME = Path(r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64")


def validate_metrics(path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # A writer may still be completing the final line.
        if row.get("worker_restarts", 0) or row.get("collector_recoveries_total", 0):
            raise RuntimeError("Training controller recovery invalidated the attempt")
        for key in ("policy_loss", "value_loss", "entropy", "gradient_norm_preclip"):
            if key in row and not math.isfinite(float(row[key])):
                raise RuntimeError(f"Nonfinite optimization metric: {key}")


def execute(name, module, arguments, deadline, *, training=False, reuse_dir=None):
    if reuse_dir is not None and name.startswith("frozen-"):
        previous = reuse_dir / f"{name}.json"
        if previous.is_file():
            target = Path(arguments[arguments.index("--output") + 1])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous, target)
            invocation = json.loads((reuse_dir / f"{name}-invocation.json").read_text())
            assert invocation["exit_code"] == 0
            invocation["reused_report"] = str(previous)
            invocation["reused_report_sha256"] = sha256_file(previous)
            atomic_json(OUT / f"{name}-invocation.json", invocation)
            print(json.dumps({"stage": name, "status": "reused_valid_report"}), flush=True)
            return invocation
    command = [sys.executable, "-m", module, *arguments]
    started = time.monotonic()
    atomic_json(OUT / "status.json", {"stage": name, "status": "running"})
    invocation = {"stage": name, "command": command, "started_at": datetime.now(UTC).isoformat()}
    atomic_json(OUT / f"{name}-invocation.json", invocation)
    print(json.dumps({"stage": name, "status": "started"}), flush=True)
    children = {}
    with (OUT / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        children[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Budget exhausted during {name}")
                if training:
                    validate_metrics(OUT / "training/metrics.jsonl")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"{name} exited {process.returncode}; inspect {name}.log")
        finally:
            for (_, created), child in children.items():
                try:
                    if child.is_running() and child.create_time() == created:
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
            invocation["elapsed_seconds"] = time.monotonic() - started
            invocation["exit_code"] = process.returncode
            atomic_json(OUT / f"{name}-invocation.json", invocation)
    print(
        json.dumps(
            {"stage": name, "status": "completed", "seconds": invocation["elapsed_seconds"]}
        ),
        flush=True,
    )
    return invocation


def summarize(reports, pools):
    results = {}
    for stage in ("frozen", "pilot"):
        episodes = [
            episode
            for (arm, _), report in reports.items()
            if arm == stage
            for episode in report["trained"]["results"]
        ]
        results[stage] = {}
        for pool, seeds in pools.items():
            rows = [row for row in episodes if row["seed"] in seeds]
            results[stage][pool] = {
                "episodes": len(rows),
                "completed": sum(row["status"] == "curriculum_complete" for row in rows),
                "completion_rate": sum(row["status"] == "curriculum_complete" for row in rows)
                / len(rows),
                "deaths": sum(row["status"] == "dead" for row in rows),
                "mean_boss_damage": sum(row["boss_damage"] for row in rows) / len(rows),
                "mean_turns": sum(row["turns"] for row in rows) / len(rows),
            }
    return results


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--reuse-frozen", type=Path)
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    origin = datetime.now(UTC)
    prior = None
    if args.reuse_frozen is not None:
        args.reuse_frozen = args.reuse_frozen.resolve()
        prior = json.loads((args.reuse_frozen / "manifest.json").read_text())
        origin = datetime.fromisoformat(prior["started_at"])
        started -= (datetime.now(UTC) - origin).total_seconds()
        for path, digest in prior["hashes"].items():
            if Path(path).resolve() != Path(__file__).resolve():
                assert sha256_file(Path(path)) == digest, f"Cannot reuse after source drift: {path}"
    deadline = started + 2700
    store = ExperimentStore(ROOT / "experiments")
    spec = store.load("EXP-0036")
    if prior is not None:
        assert prior["spec_sha256"] == spec.digest
    source = ROOT / spec.data["source"]["checkpoint"]
    qualification = ROOT / spec.data["source"]["controller_qualification"]
    gameplay = ROOT / spec.data["source"]["gameplay_qualification"]
    reward = ROOT / "configs/reward-death-metal-potential-v5.json"
    _load_qualification(qualification, GAME, ROOT / "mods/AutoDancer")
    assert json.loads(gameplay.read_text())["passed"]
    gameplay_protocol = gameplay.parent / "protocol.json"
    for path, digest in json.loads(gameplay_protocol.read_text())["hashes"].items():
        assert sha256_file(Path(path)) == digest, f"Scoped qualification stale: {path}"
    files = [source, qualification, gameplay, gameplay_protocol, reward, Path(__file__).resolve()]
    files += list((ROOT / "src/autodancer").rglob("*.py"))
    files += list((ROOT / "mods/AutoDancer").rglob("*.lua"))
    hashes = {str(p): sha256_file(p) for p in files}
    manifest = {
        "experiment_id": spec.experiment_id,
        "spec_sha256": spec.digest,
        "started_at": origin.isoformat(),
        "hashes": hashes,
        "total_budget_seconds": 2700,
        "training_budget_seconds": 1200,
        "invocations": [],
    }
    atomic_json(OUT / "manifest.json", manifest)
    pools = {
        "training": spec.data["evaluation"]["training_seeds"],
        "development": spec.data["evaluation"]["development_seeds"],
    }
    seeds = pools["training"] + pools["development"]
    streams = spec.data["evaluation"]["policy_streams"]
    common = [
        "--game-dir",
        str(GAME),
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
        "bounded-bard-v1",
        "--reward-config",
        str(reward),
        "--reward-lineage-version",
        "DeathMetalPotentialV5",
        "--policy-feedback-reward-config",
        str(reward),
        "--experiment-id",
        "EXP-0036",
        "--controller-qualification",
        str(qualification),
    ]
    reports = {}
    try:
        store.set_status("EXP-0036", "running")
        for stage in ("frozen", "pilot"):
            checkpoint = source if stage == "frozen" else OUT / "training/final.pt"
            if stage == "pilot":
                invocation = execute(
                    "training",
                    "autodancer.training.train",
                    common
                    + [
                        "--experiment-arm",
                        "pilot",
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
                        ",".join(map(str, pools["training"])),
                        "--training-level-distribution-version",
                        "bounded-direct-pilot-v1",
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
                        "0.0003",
                        "--gamma",
                        "0.99",
                        "--gae-lambda",
                        "0.95",
                    ],
                    min(deadline, time.monotonic() + 1200),
                    training=True,
                )
                manifest["invocations"].append(invocation)
                validate_metrics(OUT / "training/metrics.jsonl")
                assert checkpoint.is_file()
            for stream in streams:
                name = f"{stage}-{stream}"
                path = OUT / "evaluations" / name / "report.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                invocation = execute(
                    name,
                    "autodancer.training.baseline",
                    common
                    + [
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
                    ],
                    min(deadline, time.monotonic() + 600),
                    reuse_dir=args.reuse_frozen,
                )
                manifest["invocations"].append(invocation)
                report = json.loads(path.read_text())
                assert report["checkpoint_sha256"] == sha256_file(checkpoint)
                assert report["policy_seed"] == stream and report["policy_mode"] == "stochastic"
                atomic_json(OUT / f"{name}.json", report)
                assert report["controller_valid"] and report["worker_restarts"] == 0
                assert report["action_contract"] == "bounded-bard-v1"
                rows = report["trained"]["results"]
                assert len(rows) == len(seeds) and sorted(r["seed"] for r in rows) == sorted(seeds)
                reports[(stage, stream)] = report
                atomic_json(OUT / "manifest.json", manifest)
            assert all(sha256_file(Path(p)) == h for p, h in hashes.items()), "Source drift"
        summary = summarize(reports, pools)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["paired"] = [
            {
                "stream": stream,
                "seed": seed,
                "before": next(
                    r
                    for r in reports[("frozen", stream)]["trained"]["results"]
                    if r["seed"] == seed
                ),
                "after": next(
                    r for r in reports[("pilot", stream)]["trained"]["results"] if r["seed"] == seed
                ),
            }
            for stream in streams
            for seed in seeds
        ]
        atomic_json(OUT / "analysis.json", summary)
        atomic_json(OUT / "status.json", {"stage": "complete", "valid": True})
    except BaseException as error:
        atomic_json(
            OUT / "status.json",
            {
                "stage": "failed",
                "valid": False,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
