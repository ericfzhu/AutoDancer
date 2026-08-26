"""Command-line workflow for experiment intent, runtime lineage, and decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import (
    EXPERIMENT_ID,
    STAGES,
    ExperimentError,
    ExperimentStore,
    atomic_yaml,
    load_yaml,
)
from autodancer.experiments.tracking import (
    ExperimentTracker,
    LineageConfig,
    _mlflow_client,
    default_artifact_root,
    default_tracking_uri,
)


def _store(arguments: argparse.Namespace) -> ExperimentStore:
    return ExperimentStore(arguments.root)


def _new(arguments: argparse.Namespace) -> dict[str, Any]:
    store = _store(arguments)
    registry = store.registry()
    numbers = [
        int(key.removeprefix("EXP-"))
        for key in registry["experiments"]
        if EXPERIMENT_ID.fullmatch(key)
    ]
    experiment_id = f"EXP-{max(numbers, default=0) + 1:04d}"
    directory = store.root / experiment_id
    directory.mkdir(parents=True, exist_ok=False)
    spec_path = directory / "experiment.yaml"
    component_versions: dict[str, str] = {}
    for component in arguments.component:
        if "=" not in component:
            raise ExperimentError("--component must use BLOCK=VERSION")
        block, version = component.split("=", 1)
        component_versions[block] = version
    spec = {
        "schema_version": 1,
        "id": experiment_id,
        "title": arguments.title,
        "question": arguments.question,
        "hypothesis": arguments.hypothesis,
        "changed_blocks": arguments.changed_block,
        "component_versions": component_versions,
        "invariants": arguments.invariant,
        "arms": [
            {"id": arm, "description": "TODO: define this arm exactly"} for arm in arguments.arm
        ],
        "training": {"TODO": "declare seeds, steps, initialization, and capacity"},
        "evaluation": {"TODO": "declare held-out seeds, turn cap, and metrics"},
        "decision_rule": arguments.decision_rule,
    }
    atomic_yaml(spec_path, spec)
    readme = directory / "README.md"
    readme.write_text(
        f"# {experiment_id}: {arguments.title}\n\n"
        "The immutable scientific contract is `experiment.yaml`. Runtime evidence is in "
        "MLflow; the final conclusion belongs in `decision.json`.\n",
        encoding="utf-8",
    )
    validated = store.register(spec_path)
    return {"experiment_id": experiment_id, "spec": str(spec_path), "spec_sha256": validated.digest}


def _validate(arguments: argparse.Namespace) -> dict[str, Any]:
    specs = _store(arguments).validate()
    return {"valid": True, "experiments": len(specs), "ids": [spec.experiment_id for spec in specs]}


def _list(arguments: argparse.Namespace) -> dict[str, Any]:
    registry = _store(arguments).registry()
    return {"experiments": [{"id": key, **value} for key, value in registry["experiments"].items()]}


def _decide(arguments: argparse.Namespace) -> dict[str, Any]:
    decision = {
        "outcome": arguments.outcome,
        "summary": arguments.summary,
        "selected_arm": arguments.selected_arm,
        "evidence": [str(path) for path in arguments.evidence],
        "mlflow_run_ids": arguments.mlflow_run_id,
    }
    tracking_uri = arguments.tracking_uri or default_tracking_uri()
    client = _mlflow_client(tracking_uri)
    for run_id in arguments.mlflow_run_id:
        run = client.get_run(run_id)
        if run.data.tags.get("autodancer.experiment_id") != arguments.experiment_id:
            raise ExperimentError(f"MLflow run {run_id} belongs to another experiment")
    path = _store(arguments).write_decision(arguments.experiment_id, decision)
    experiment = client.get_experiment_by_name("AutoDancer")
    synced_parent = None
    if experiment is not None:
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                f"tags.`autodancer.experiment_id` = '{arguments.experiment_id}' "
                "and tags.`autodancer.role` = 'parent'"
            ),
            max_results=1,
        )
        if runs:
            synced_parent = runs[0].info.run_id
            client.set_tag(synced_parent, "autodancer.decision", arguments.outcome)
            client.set_tag(
                synced_parent,
                "autodancer.selected_arm",
                arguments.selected_arm or "none",
            )
            client.log_artifact(synced_parent, str(path), artifact_path="decision")
            client.set_terminated(synced_parent, status="FINISHED")
    for run_id in arguments.mlflow_run_id:
        client.set_tag(run_id, "autodancer.decision", arguments.outcome)
    return {
        "decision": str(path),
        "outcome": arguments.outcome,
        "mlflow_parent_run_id": synced_parent,
    }


def _promote(arguments: argparse.Namespace) -> dict[str, Any]:
    store = _store(arguments)
    spec = store.load(arguments.experiment_id)
    spec.validate_arm(arguments.arm)
    registry_entry = store.registry()["experiments"][arguments.experiment_id]
    if registry_entry["status"] != "accepted":
        raise ExperimentError("Only an accepted experiment can promote a baseline")
    decision_path = store.root / arguments.experiment_id / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("selected_arm") != arguments.arm:
        raise ExperimentError("Promoted arm does not match the accepted decision")
    if not arguments.checkpoint.is_file():
        raise ExperimentError(f"Checkpoint does not exist: {arguments.checkpoint}")
    store.promote_baseline(
        arguments.name,
        {
            "experiment_id": arguments.experiment_id,
            "arm": arguments.arm,
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_sha256": sha256_file(arguments.checkpoint),
            "mlflow_run_id": arguments.mlflow_run_id,
            "architecture": arguments.architecture,
            "reward": arguments.reward,
            "reason": arguments.reason,
        },
    )
    return {"baseline": arguments.name, "checkpoint_sha256": sha256_file(arguments.checkpoint)}


def _artifact_run(arguments: argparse.Namespace, *, backfilled: bool) -> dict[str, Any]:
    tracker = ExperimentTracker(
        LineageConfig(
            experiment_id=arguments.experiment_id,
            arm=arguments.arm,
            trial=arguments.trial,
            stage=arguments.stage,
            run_dir=arguments.run_dir,
            store_root=arguments.root,
            tracking_uri=arguments.tracking_uri,
            qualification_report=arguments.qualification_report,
        ),
        game_dir=arguments.game_dir,
        mod_dir=arguments.mod_dir,
        device=arguments.device,
        parameters={"backfilled": backfilled, "notes": arguments.notes},
        source_checkpoint=arguments.source_checkpoint,
    )
    outputs = [path for path in arguments.artifact if path.is_file()]
    tracker.complete(outputs, summary={"backfilled": backfilled})
    return {
        "run_id": tracker.run_id,
        "parent_run_id": tracker.parent_run_id,
        "artifacts": len(outputs),
    }


def _backfill(arguments: argparse.Namespace) -> dict[str, Any]:
    return _artifact_run(arguments, backfilled=True)


def _attach(arguments: argparse.Namespace) -> dict[str, Any]:
    return _artifact_run(arguments, backfilled=False)


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--mod-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument(
        "--qualification-report",
        type=Path,
        default=Path("runs/controller-qualification/qualification.json"),
    )
    parser.add_argument("--tracking-uri")
    parser.add_argument("--device", default="historical")
    parser.add_argument("--notes", default="")


def _ui(arguments: argparse.Namespace) -> dict[str, Any]:
    tracking_uri = arguments.tracking_uri or default_tracking_uri()
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        tracking_uri,
        "--default-artifact-root",
        default_artifact_root(),
        "--host",
        arguments.host,
        "--port",
        str(arguments.port),
    ]
    print(
        json.dumps(
            {"url": f"http://{arguments.host}:{arguments.port}", "tracking_uri": tracking_uri}
        ),
        flush=True,
    )
    return_code = subprocess.run(command, check=False).returncode
    if return_code:
        raise ExperimentError(f"MLflow server exited with code {return_code}")
    return {"stopped": True}


def _sync(arguments: argparse.Namespace) -> dict[str, Any]:
    """Synchronize Git decisions and baseline promotions into the runtime store."""
    store = _store(arguments)
    store.validate()
    client = _mlflow_client(arguments.tracking_uri or default_tracking_uri())
    experiment = client.get_experiment_by_name("AutoDancer")
    if experiment is None:
        return {"decisions": 0, "baselines": 0, "message": "No AutoDancer MLflow store"}
    decision_count = 0
    child_runs: dict[str, Any] = {}
    for experiment_id in store.registry()["experiments"]:
        decision_path = store.root / experiment_id / "decision.json"
        if not decision_path.is_file():
            continue
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.`autodancer.experiment_id` = '{experiment_id}'",
            max_results=1000,
        )
        for run in runs:
            if run.data.tags.get("autodancer.role") == "parent":
                client.set_tag(run.info.run_id, "autodancer.decision", decision["outcome"])
                client.set_tag(
                    run.info.run_id,
                    "autodancer.selected_arm",
                    decision.get("selected_arm") or "none",
                )
                client.log_artifact(run.info.run_id, str(decision_path), artifact_path="decision")
                client.set_terminated(run.info.run_id, status="FINISHED")
                decision_count += 1
            else:
                client.set_tag(run.info.run_id, "autodancer.decision", decision["outcome"])
                child_runs[run.info.run_id] = run
    baseline_count = 0
    baseline_records = load_yaml(store.baselines_path)
    for name, baseline in baseline_records["baselines"].items():
        run_id = baseline.get("mlflow_run_id")
        if run_id in child_runs:
            client.set_tag(run_id, "autodancer.baseline", name)
            baseline_count += 1
    return {"decisions": decision_count, "baselines": baseline_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage AutoDancer experiment lineage")
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    new = subparsers.add_parser("new", help="create and register an immutable experiment contract")
    new.add_argument("--title", required=True)
    new.add_argument("--question", required=True)
    new.add_argument("--hypothesis", required=True)
    new.add_argument("--changed-block", action="append", required=True)
    new.add_argument("--component", action="append", required=True, metavar="BLOCK=VERSION")
    new.add_argument("--invariant", action="append", required=True)
    new.add_argument("--arm", action="append", required=True)
    new.add_argument("--decision-rule", required=True)
    new.set_defaults(handler=_new)

    validate = subparsers.add_parser(
        "validate", help="validate registry, specs, and immutable hashes"
    )
    validate.set_defaults(handler=_validate)
    listing = subparsers.add_parser("list", help="list registered experiments")
    listing.set_defaults(handler=_list)

    decide = subparsers.add_parser("decide", help="record one immutable experiment decision")
    decide.add_argument("--experiment-id", required=True)
    decide.add_argument(
        "--outcome", choices=("accepted", "rejected", "inconclusive"), required=True
    )
    decide.add_argument("--summary", required=True)
    decide.add_argument("--selected-arm")
    decide.add_argument("--evidence", action="append", type=Path, default=[])
    decide.add_argument("--mlflow-run-id", action="append", default=[])
    decide.add_argument("--tracking-uri")
    decide.set_defaults(handler=_decide)

    promote = subparsers.add_parser("promote", help="promote a content-addressed baseline")
    promote.add_argument("--name", required=True)
    promote.add_argument("--experiment-id", required=True)
    promote.add_argument("--arm", required=True)
    promote.add_argument("--checkpoint", type=Path, required=True)
    promote.add_argument("--mlflow-run-id", required=True)
    promote.add_argument("--architecture", required=True)
    promote.add_argument("--reward", required=True)
    promote.add_argument("--reason", required=True)
    promote.set_defaults(handler=_promote)

    backfill = subparsers.add_parser("backfill", help="attach an existing run/report to MLflow")
    _add_artifact_arguments(backfill)
    backfill.set_defaults(handler=_backfill)

    attach = subparsers.add_parser(
        "attach", help="attach a newly generated diagnostic/comparison artifact"
    )
    _add_artifact_arguments(attach)
    attach.set_defaults(handler=_attach)

    ui = subparsers.add_parser("ui", help="serve the local MLflow UI")
    ui.add_argument("--tracking-uri")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=5000)
    ui.set_defaults(handler=_ui)
    sync = subparsers.add_parser("sync", help="sync Git decisions/baselines into MLflow")
    sync.add_argument("--tracking-uri")
    sync.set_defaults(handler=_sync)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        result = arguments.handler(arguments)
    except ExperimentError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
