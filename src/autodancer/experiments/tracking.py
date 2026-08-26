"""MLflow runtime lineage paired with repository experiment specifications."""

from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autodancer.experiments.provenance import (
    controller_identity,
    environment_snapshot,
    git_identity,
    runtime_identity,
    sha256_file,
)
from autodancer.experiments.schema import STAGES, ExperimentError, ExperimentStore, atomic_json
from autodancer.live.protocol import SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LineageConfig:
    experiment_id: str
    arm: str
    trial: str
    stage: str
    run_dir: Path
    store_root: Path = Path("experiments")
    tracking_uri: str | None = None
    qualification_report: Path | None = Path("runs/controller-qualification/qualification.json")


def default_tracking_uri() -> str:
    configured = os.environ.get("AUTODANCER_MLFLOW_URI")
    if configured:
        return configured
    database = (Path(".runtime") / "mlflow" / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def default_artifact_root() -> str:
    path = (Path(".runtime") / "mlflow" / "artifacts").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path.as_uri()


def _mlflow_client(tracking_uri: str):
    try:
        from mlflow import MlflowClient
    except ImportError as error:
        raise ExperimentError(
            "MLflow lineage was requested but MLflow is not installed. "
            "Run `uv sync --extra train --extra lineage`."
        ) from error
    return MlflowClient(tracking_uri=tracking_uri)


def _flatten(prefix: str, value: Any, result: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, result)
    elif isinstance(value, (list, tuple)):
        result[prefix] = ",".join(str(item) for item in value)
    elif value is not None:
        result[prefix] = str(value)


def validate_qualification_freshness(
    report: dict[str, Any], *, game_dir: Path, mod_dir: Path
) -> None:
    """Reject a passed report that qualifies different controller artifacts."""
    configuration = report.get("configuration") or {}
    preflight = (report.get("phases") or {}).get("preflight") or {}
    if int(preflight.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ExperimentError("Controller qualification protocol schema is stale")
    recorded_game_dir = configuration.get("game_dir")
    recorded_mod_dir = configuration.get("mod_dir")
    if recorded_game_dir is None or Path(recorded_game_dir).resolve() != game_dir.resolve():
        raise ExperimentError("Controller qualification targets a different game directory")
    if recorded_mod_dir is None or Path(recorded_mod_dir).resolve() != mod_dir.resolve():
        raise ExperimentError("Controller qualification targets a different mod directory")
    recorded_files = preflight.get("mod_files")
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise ExperimentError("Controller qualification does not bind mod file hashes")
    mismatches = [
        relative
        for relative, expected in recorded_files.items()
        if sha256_file(mod_dir / Path(str(relative).replace("\\", "/"))) != expected
    ]
    if mismatches:
        raise ExperimentError(
            "Controller qualification is stale for mod files: " + ", ".join(mismatches)
        )


class ExperimentTracker:
    """Own one MLflow child run and an atomic local run manifest."""

    def __init__(
        self,
        config: LineageConfig,
        *,
        game_dir: Path,
        mod_dir: Path,
        device: str,
        parameters: dict[str, Any],
        source_checkpoint: Path | None = None,
    ) -> None:
        self.config = config
        if config.stage not in STAGES:
            raise ExperimentError(f"Unknown experiment stage {config.stage!r}")
        if not config.arm.strip() or not config.trial.strip():
            raise ExperimentError("Experiment arm and trial labels must be non-empty")
        self.store = ExperimentStore(config.store_root)
        self.spec = self.store.load(config.experiment_id)
        if config.arm == "aggregate":
            if config.stage not in {"comparison", "diagnostic"}:
                raise ExperimentError(
                    "The reserved aggregate arm is only valid for comparison/diagnostic stages"
                )
        else:
            self.spec.validate_arm(config.arm)
        if config.qualification_report is not None:
            if not config.qualification_report.is_file():
                raise ExperimentError(
                    f"Controller qualification report is missing: {config.qualification_report}"
                )
            try:
                qualification = json.loads(config.qualification_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ExperimentError("Controller qualification report is malformed") from error
            if qualification.get("passed") is not True:
                raise ExperimentError(
                    "Tracked experiments require a passed controller qualification"
                )
            validate_qualification_freshness(
                qualification, game_dir=game_dir, mod_dir=mod_dir
            )
        self.tracking_uri = config.tracking_uri or default_tracking_uri()
        self.client = _mlflow_client(self.tracking_uri)
        experiment = self.client.get_experiment_by_name("AutoDancer")
        if experiment is None:
            experiment_id = self.client.create_experiment(
                "AutoDancer", artifact_location=default_artifact_root()
            )
        else:
            experiment_id = experiment.experiment_id
        self.mlflow_experiment_id = experiment_id
        self.parent_run_id = self._parent_run()
        self.manifest_path = config.run_dir / "lineage.json"
        attempt, resume_of = self._preserve_previous_attempt()
        tags = {
            "mlflow.runName": f"{config.experiment_id}/{config.arm}/{config.trial}/{config.stage}",
            "mlflow.parentRunId": self.parent_run_id,
            "autodancer.experiment_id": config.experiment_id,
            "autodancer.arm": config.arm,
            "autodancer.trial": config.trial,
            "autodancer.stage": config.stage,
            "autodancer.spec_sha256": self.spec.digest,
            "autodancer.attempt": str(attempt),
        }
        if resume_of is not None:
            tags["autodancer.resume_of"] = resume_of
        self.run_id = self.client.create_run(experiment_id, tags=tags).info.run_id
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "experiment": {
                "id": config.experiment_id,
                "title": self.spec.title,
                "arm": config.arm,
                "trial": config.trial,
                "stage": config.stage,
                "spec": str(self.spec.path),
                "spec_sha256": self.spec.digest,
            },
            "mlflow": {
                "tracking_uri": self.tracking_uri,
                "experiment_id": experiment_id,
                "parent_run_id": self.parent_run_id,
                "run_id": self.run_id,
                "attempt": attempt,
                "resume_of": resume_of,
            },
            "git": git_identity(),
            "runtime": runtime_identity(device),
            "environment": environment_snapshot(),
            "controller": controller_identity(game_dir, mod_dir, config.qualification_report),
            "parameters": parameters,
            "source_checkpoint": (
                None
                if source_checkpoint is None
                else {
                    "path": str(source_checkpoint.resolve()),
                    "sha256": sha256_file(source_checkpoint),
                }
            ),
            "resolved": {},
            "outputs": [],
        }
        config.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.manifest_path, self.manifest)
        flattened: dict[str, str] = {}
        _flatten("", parameters, flattened)
        for name, value in flattened.items():
            self.client.log_param(self.run_id, name[:250], value[:6000])
        self.client.log_param(self.run_id, "spec_sha256", self.spec.digest)
        self.client.log_artifact(self.run_id, str(self.spec.path), artifact_path="spec")

    def _preserve_previous_attempt(self) -> tuple[int, str | None]:
        if not self.manifest_path.is_file():
            return 1, None
        try:
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExperimentError(
                f"Existing lineage manifest is malformed: {self.manifest_path}"
            ) from error
        identity = previous.get("experiment", {})
        expected = {
            "id": self.config.experiment_id,
            "arm": self.config.arm,
            "trial": self.config.trial,
            "stage": self.config.stage,
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ExperimentError(
                f"Run directory already belongs to a different lineage identity: {identity}"
            )
        mlflow = previous.get("mlflow", {})
        attempt = int(mlflow.get("attempt", 1)) + 1
        resume_of = mlflow.get("run_id")
        archive = self.manifest_path.with_name(
            f"lineage-attempt-{attempt - 1:02d}-{str(resume_of)[:8]}.json"
        )
        if archive.exists():
            raise ExperimentError(f"Previous-attempt archive already exists: {archive}")
        os.replace(self.manifest_path, archive)
        return attempt, None if resume_of is None else str(resume_of)

    def _parent_run(self) -> str:
        escaped = self.config.experiment_id.replace("'", "\\'")
        parent_filter = (
            f"tags.`autodancer.experiment_id` = '{escaped}' and tags.`autodancer.role` = 'parent'"
        )
        runs = self.client.search_runs(
            [self.mlflow_experiment_id],
            filter_string=parent_filter,
            max_results=1,
        )
        if runs:
            if runs[0].data.tags.get("autodancer.spec_sha256") != self.spec.digest:
                raise ExperimentError(
                    f"MLflow parent for {self.config.experiment_id} has a stale spec hash"
                )
            return runs[0].info.run_id
        tags = {
            "mlflow.runName": f"{self.config.experiment_id}: {self.spec.title}",
            "autodancer.experiment_id": self.config.experiment_id,
            "autodancer.role": "parent",
            "autodancer.spec_sha256": self.spec.digest,
        }
        parent = self.client.create_run(self.mlflow_experiment_id, tags=tags)
        self.client.log_artifact(parent.info.run_id, str(self.spec.path), artifact_path="spec")
        return parent.info.run_id

    def set_resolved(self, values: dict[str, Any]) -> None:
        self.manifest["resolved"] = values
        atomic_json(self.manifest_path, self.manifest)

    def validate_component_versions(
        self,
        observed: dict[str, str],
        *,
        config_hashes: dict[str, str | None] | None = None,
        require_declared: bool = False,
    ) -> None:
        expected = self.spec.components_for_arm(self.config.arm)
        for block, actual in observed.items():
            declared = expected.get(block)
            if require_declared and declared is None:
                raise ExperimentError(
                    f"Run component {block}={actual} is not declared by "
                    f"{self.config.experiment_id}/{self.config.arm}"
                )
            allowed = declared if isinstance(declared, list) else [declared]
            if declared is not None and actual not in allowed:
                raise ExperimentError(
                    f"Run component {block}={actual} does not match "
                    f"{self.config.experiment_id}/{self.config.arm}: {allowed}"
                )
            definition = self.store.component_definition(block, actual)
            if isinstance(definition, dict) and definition.get("config_sha256"):
                actual_hash = (config_hashes or {}).get(block)
                if actual_hash != definition["config_sha256"]:
                    raise ExperimentError(
                        f"Run {block} config hash {actual_hash!r} does not match {actual}"
                    )

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    self.client.log_metric(self.run_id, name[:250], numeric, step=step)

    def complete(self, outputs: list[Path], *, summary: dict[str, Any] | None = None) -> None:
        records = []
        for path in outputs:
            if path.is_file():
                records.append(
                    {
                        "path": str(path.resolve()),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        self.manifest.update(
            status="completed",
            completed_at=datetime.now(UTC).isoformat(),
            outputs=records,
            summary=summary or {},
        )
        atomic_json(self.manifest_path, self.manifest)
        self.client.log_artifact(self.run_id, str(self.manifest_path), artifact_path="lineage")
        for path in outputs:
            if path.is_file() and path.stat().st_size <= 32 * 1024 * 1024:
                self.client.log_artifact(self.run_id, str(path), artifact_path="outputs")
        self.client.set_terminated(self.run_id, status="FINISHED")

    def fail(self, error: BaseException) -> None:
        self.manifest.update(
            status="failed",
            failed_at=datetime.now(UTC).isoformat(),
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_json(self.manifest_path, self.manifest)
        self.client.log_artifact(self.run_id, str(self.manifest_path), artifact_path="lineage")
        self.client.set_tag(self.run_id, "autodancer.failure_type", type(error).__name__)
        self.client.set_terminated(self.run_id, status="FAILED")
