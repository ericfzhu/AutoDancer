"""Run the predeclared EXP-0035 preparation gate, without training or final testing."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/learning-bottleneck-exp0035"
GAME = Path(r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64")
SOURCE = ROOT / "runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt"
QUALIFICATION = ROOT / "runs/controller-qualification-steam-parity/qualification.json"
STREAMS = (215001, 215002)


def run_command(name: str, module: str, arguments: list[str], deadline: float) -> dict:
    remaining = deadline - time.monotonic()
    if remaining < 30:
        raise TimeoutError("Preparation budget exhausted")
    command = [sys.executable, "-m", module, *arguments]
    started = time.monotonic()
    atomic_json(OUT / "status.json", {"stage": name, "started_at": datetime.now(UTC).isoformat()})
    print(json.dumps({"stage": name, "event": "started"}), flush=True)
    with (OUT / f"{name}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children = {}
        try:
            while process.poll() is None:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        children[(child.pid, child.create_time())] = child
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Preparation budget exhausted in {name}")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"{name} failed with exit code {process.returncode}; see log")
        finally:
            # Only this subprocess and its captured descendants, checked against PID reuse.
            for (_pid, created), child in children.items():
                try:
                    if child.is_running() and child.create_time() == created:
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
    result = {"name": name, "command": command, "seconds": time.monotonic() - started}
    atomic_json(OUT / f"{name}-invocation.json", result)
    print(
        json.dumps({"stage": name, "event": "completed", "seconds": result["seconds"]}), flush=True
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 3600
    spec = ExperimentStore(ROOT / "experiments").load("EXP-0035")
    manifest = {
        "experiment_id": spec.experiment_id,
        "spec_sha256": spec.digest,
        "started_at": datetime.now(UTC).isoformat(),
        "source_checkpoint": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "qualification_sha256": sha256_file(QUALIFICATION),
        "preparation_budget_seconds": 3600,
        "invocations": [],
    }
    atomic_json(OUT / "manifest.json", manifest)
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
        "--affinity",
        "none",
    ]
    try:
        identity_path = OUT / "identities.json"
        manifest["invocations"].append(
            run_command(
                "identity",
                "autodancer.training.boss_identity",
                common
                + [
                    "--output",
                    str(identity_path),
                    "--seeds",
                    ",".join(map(str, range(214000, 214256))),
                    "--controller-qualification",
                    str(QUALIFICATION),
                ],
                deadline,
            )
        )
        identities = json.loads(identity_path.read_text())
        seeds = sorted(r["seed"] for r in identities["results"] if r["boss_name"] == "DEATH_METAL")
        if len(seeds) < 40:
            raise ValueError(f"Insufficient Death Metal identities: {len(seeds)}")
        pools = {
            "training": seeds[:16],
            "development": seeds[16:24],
            "final_test": seeds[24:40],
            "historically_familiar": [92008, 92096, 92116],
            "selection": "boss identity only",
            "identity_sha256": sha256_file(identity_path),
        }
        atomic_json(OUT / "pools.json", pools)
        evaluation_seeds = pools["training"] + pools["development"] + pools["historically_familiar"]
        evaluation = common + [
            "--checkpoint",
            str(SOURCE),
            "--max-steps",
            "500",
            "--device",
            "cuda",
            "--trained-only",
            "--policy-mode",
            "stochastic",
            "--action-contract",
            "map-navigation-prior-v1",
            "--steam-presence-worker",
            "0",
            "--reward-config",
            str(ROOT / "configs/reward-death-metal-potential-v5.json"),
            "--policy-feedback-reward-config",
            str(ROOT / "configs/reward-death-metal-potential-v5.json"),
        ]
        reports = []
        for kind in ("direct", "trace-control"):
            for stream in STREAMS:
                name = f"{kind}-{stream}"
                report_path = OUT / f"{name}.json"
                selected = evaluation_seeds if kind == "direct" else pools["historically_familiar"]
                extra = (
                    []
                    if kind == "direct"
                    else [
                        "--trace-prefix-bank",
                        str(
                            ROOT / "runs/qualified-death-metal-trace-search/demonstration-bank.json"
                        ),
                        "--trace-prefix-qualification",
                        str(ROOT / "runs/qualified-death-metal-trace-search/qualification.json"),
                        "--trace-prefix-tail-actions",
                        "60",
                        "--trace-prefix-recurrent-state",
                        "warm",
                    ]
                )
                manifest["invocations"].append(
                    run_command(
                        name,
                        "autodancer.training.baseline",
                        evaluation
                        + [
                            "--seeds",
                            ",".join(map(str, selected)),
                            "--policy-seed",
                            str(stream),
                            "--output",
                            str(report_path),
                        ]
                        + extra,
                        deadline,
                    )
                )
                report = json.loads(report_path.read_text())
                if not report["controller_valid"] or report["worker_restarts"]:
                    raise ValueError(f"Controller-invalid report: {name}")
                reports.append((kind, stream, report))
        summaries = {}
        for kind in ("direct", "trace-control"):
            for pool in ("training", "development", "historically_familiar"):
                episodes = [
                    e
                    for k, _, report in reports
                    if k == kind
                    for e in report["trained"]["results"]
                    if e["seed"] in pools[pool]
                ]
                if episodes:
                    completed = sum(e["status"] == "curriculum_complete" for e in episodes)
                    summaries[f"{kind}/{pool}"] = {
                        "episodes": len(episodes),
                        "completions": completed,
                        "completion_rate": completed / len(episodes),
                    }
        rate = summaries["direct/training"]["completion_rate"]
        result = {
            "gate_passed": 0.1 <= rate <= 0.9,
            "summaries": summaries,
            "elapsed_seconds": time.monotonic() - started,
            "final_test_evaluated": False,
        }
        atomic_json(OUT / "preparation-result.json", result)
        atomic_json(OUT / "status.json", {"stage": "preparation_complete", **result})
        print(json.dumps(result), flush=True)
    except BaseException as error:
        atomic_json(
            OUT / "status.json",
            {
                "stage": "failed",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


def deterministic_extension() -> None:
    """Run the protocol's optional secondary execution-mode check within its original cap."""
    manifest = json.loads((OUT / "manifest.json").read_text())
    age = (datetime.now(UTC) - datetime.fromisoformat(manifest["started_at"])).total_seconds()
    if age >= 2700:
        raise TimeoutError("Less than 15 minutes remain for the optional secondary check")
    deadline = time.monotonic() + 3600 - age
    for kind in ("direct", "trace-control"):
        name = f"{kind}-deterministic"
        output = OUT / f"{name}.json"
        if output.exists() or (OUT / f"{name}.log").exists():
            raise FileExistsError(f"Refusing to overwrite {name}")
        original = next(r for r in manifest["invocations"] if r["name"] == f"{kind}-215001")
        arguments = list(original["command"][3:])
        for flag, value in (
            ("--policy-mode", "deterministic"),
            ("--policy-seed", "0"),
            ("--output", str(output)),
        ):
            arguments[arguments.index(flag) + 1] = value
        manifest["invocations"].append(
            run_command(name, "autodancer.training.baseline", arguments, deadline)
        )
        manifest["elapsed_seconds_with_secondary_check"] = (
            datetime.now(UTC) - datetime.fromisoformat(manifest["started_at"])
        ).total_seconds()
        atomic_json(OUT / "manifest.json", manifest)
    atomic_json(OUT / "status.json", {"stage": "preparation_complete_with_secondary_check"})


if __name__ == "__main__":
    if "--deterministic" in sys.argv:
        deterministic_extension()
    else:
        main()
