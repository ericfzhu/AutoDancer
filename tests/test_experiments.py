from __future__ import annotations

import json
from pathlib import Path

import pytest

from autodancer.experiments.schema import (
    ExperimentError,
    ExperimentStore,
    atomic_yaml,
    validate_spec,
)
from autodancer.experiments.tracking import (
    ExperimentTracker,
    LineageConfig,
    validate_qualification_freshness,
)
from autodancer.live.protocol import SCHEMA_VERSION


def _experiment_store(root: Path) -> tuple[ExperimentStore, Path]:
    experiments = root / "experiments"
    directory = experiments / "EXP-0001"
    directory.mkdir(parents=True)
    spec_path = directory / "experiment.yaml"
    atomic_yaml(
        spec_path,
        {
            "schema_version": 1,
            "id": "EXP-0001",
            "title": "Test intervention",
            "question": "Does the intervention work?",
            "hypothesis": "The treatment will beat control.",
            "changed_blocks": ["architecture"],
            "component_versions": {"architecture": "A-test"},
            "invariants": ["reward-v2"],
            "arms": [
                {"id": "control", "description": "unchanged"},
                {"id": "treatment", "description": "changed"},
            ],
            "training": {"seeds": [1, 2, 3]},
            "evaluation": {"seeds": [11, 12, 13]},
            "decision_rule": "Treatment must improve progress.",
        },
    )
    digest = validate_spec(spec_path).digest
    atomic_yaml(
        experiments / "registry.yaml",
        {
            "schema_version": 1,
            "experiments": {
                "EXP-0001": {
                    "spec": "experiments/EXP-0001/experiment.yaml",
                    "spec_sha256": digest,
                    "status": "planned",
                    "historical": False,
                }
            },
        },
    )
    atomic_yaml(experiments / "baselines.yaml", {"schema_version": 1, "baselines": {}})
    atomic_yaml(
        experiments / "components.yaml",
        {
            "schema_version": 1,
            "blocks": {"architecture": {"versions": {"A-test": "test architecture"}}},
        },
    )
    return ExperimentStore(experiments), spec_path


def test_registry_pins_immutable_spec_and_validates_arms(tmp_path: Path) -> None:
    store, spec_path = _experiment_store(tmp_path)
    spec = store.load("EXP-0001")
    spec.validate_arm("treatment")
    with pytest.raises(ExperimentError, match="Unknown arm"):
        spec.validate_arm("missing")

    spec_path.write_text(spec_path.read_text() + "notes: changed later\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="changed after registration"):
        store.load("EXP-0001")


def test_decisions_are_single_and_update_registry(tmp_path: Path) -> None:
    store, _ = _experiment_store(tmp_path)
    decision = store.write_decision(
        "EXP-0001", {"outcome": "rejected", "summary": "Did not pass.", "evidence": []}
    )
    assert json.loads(decision.read_text())["outcome"] == "rejected"
    assert store.registry()["experiments"]["EXP-0001"]["status"] == "rejected"
    with pytest.raises(ExperimentError, match="already exists"):
        store.write_decision(
            "EXP-0001", {"outcome": "accepted", "summary": "Changed mind.", "evidence": []}
        )


def test_component_catalog_and_baseline_names_are_strict(tmp_path: Path) -> None:
    store, _ = _experiment_store(tmp_path)
    components = store.components_path
    atomic_yaml(components, {"schema_version": 1, "blocks": {}})
    with pytest.raises(ExperimentError, match="unknown block"):
        store.validate()

    atomic_yaml(
        components,
        {
            "schema_version": 1,
            "blocks": {"architecture": {"versions": {"A-test": "test architecture"}}},
        },
    )
    store.promote_baseline("candidate-v1", {"experiment_id": "EXP-0001"})
    with pytest.raises(ExperimentError, match="already exists"):
        store.promote_baseline("candidate-v1", {"experiment_id": "EXP-0001"})


def test_mlflow_parent_child_manifest_and_output_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _experiment_store(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(
        "autodancer.experiments.tracking.default_artifact_root", artifact_root.as_uri
    )
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_dir = tmp_path / "run"
    output = run_dir / "report.json"
    output.parent.mkdir()
    output.write_text('{"score": 2}\n', encoding="utf-8")

    tracker = ExperimentTracker(
        LineageConfig(
            experiment_id="EXP-0001",
            arm="treatment",
            trial="seed-1",
            stage="evaluation",
            run_dir=run_dir,
            store_root=store.root,
            tracking_uri=tracking_uri,
            qualification_report=None,
        ),
        game_dir=tmp_path / "game",
        mod_dir=tmp_path / "mod",
        device="cpu",
        parameters={"seed": 1, "nested": {"value": 2}},
    )
    tracker.set_resolved({"architecture": {"version": 9}})
    tracker.validate_component_versions({"architecture": "A-test"})
    with pytest.raises(ExperimentError, match="does not match"):
        tracker.validate_component_versions({"architecture": "A-other"})
    tracker.log_metrics({"score": 2.0, "ignored": "text"}, step=10)
    tracker.complete([output], summary={"score": 2.0})

    manifest = json.loads((run_dir / "lineage.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["experiment"]["spec_sha256"] == store.load("EXP-0001").digest
    assert manifest["outputs"][0]["sha256"]
    child = tracker.client.get_run(tracker.run_id)
    assert child.data.tags["mlflow.parentRunId"] == tracker.parent_run_id
    assert child.data.metrics["score"] == 2.0


def test_tracker_rejects_arm_not_declared_before_creating_run(tmp_path: Path) -> None:
    store, _ = _experiment_store(tmp_path)
    with pytest.raises(ExperimentError, match="Unknown arm"):
        ExperimentTracker(
            LineageConfig(
                experiment_id="EXP-0001",
                arm="unknown",
                trial="seed-1",
                stage="training",
                run_dir=tmp_path / "run",
                store_root=store.root,
                tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                qualification_report=None,
            ),
            game_dir=tmp_path,
            mod_dir=tmp_path,
            device="cpu",
            parameters={},
        )

    with pytest.raises(ExperimentError, match="aggregate arm"):
        ExperimentTracker(
            LineageConfig(
                experiment_id="EXP-0001",
                arm="aggregate",
                trial="seed-1",
                stage="training",
                run_dir=tmp_path / "aggregate",
                store_root=store.root,
                tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                qualification_report=None,
            ),
            game_dir=tmp_path,
            mod_dir=tmp_path,
            device="cpu",
            parameters={},
        )


def test_tracker_requires_passed_controller_qualification(tmp_path: Path) -> None:
    store, _ = _experiment_store(tmp_path)
    qualification = tmp_path / "qualification.json"
    qualification.write_text('{"passed": false}\n', encoding="utf-8")
    with pytest.raises(ExperimentError, match="require a passed"):
        ExperimentTracker(
            LineageConfig(
                experiment_id="EXP-0001",
                arm="control",
                trial="seed-1",
                stage="training",
                run_dir=tmp_path / "run",
                store_root=store.root,
                tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
                qualification_report=qualification,
            ),
            game_dir=tmp_path,
            mod_dir=tmp_path,
            device="cpu",
            parameters={},
        )


def test_qualification_freshness_binds_current_mod_hashes(tmp_path: Path) -> None:
    game_dir = tmp_path / "game"
    mod_dir = tmp_path / "mod"
    game_dir.mkdir()
    (mod_dir / "scripts").mkdir(parents=True)
    script = mod_dir / "scripts" / "Bridge.lua"
    script.write_text("qualified", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    report = {
        "configuration": {"game_dir": str(game_dir), "mod_dir": str(mod_dir)},
        "phases": {
            "preflight": {
                "schema_version": SCHEMA_VERSION,
                "mod_files": {"scripts\\Bridge.lua": digest},
            }
        },
    }
    validate_qualification_freshness(report, game_dir=game_dir, mod_dir=mod_dir)
    script.write_text("changed", encoding="utf-8")
    with pytest.raises(ExperimentError, match="stale.*Bridge.lua"):
        validate_qualification_freshness(report, game_dir=game_dir, mod_dir=mod_dir)


def test_new_attempt_preserves_previous_manifest(tmp_path: Path) -> None:
    store, _ = _experiment_store(tmp_path)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_dir = tmp_path / "run"

    def create_tracker() -> ExperimentTracker:
        return ExperimentTracker(
            LineageConfig(
                experiment_id="EXP-0001",
                arm="control",
                trial="seed-1",
                stage="training",
                run_dir=run_dir,
                store_root=store.root,
                tracking_uri=tracking_uri,
                qualification_report=None,
            ),
            game_dir=tmp_path,
            mod_dir=tmp_path,
            device="cpu",
            parameters={},
        )

    first = create_tracker()
    first.fail(RuntimeError("interrupted"))
    second = create_tracker()
    manifest = json.loads((run_dir / "lineage.json").read_text())
    assert manifest["mlflow"]["attempt"] == 2
    assert manifest["mlflow"]["resume_of"] == first.run_id
    assert list(run_dir.glob("lineage-attempt-01-*.json"))
    second.complete([])
