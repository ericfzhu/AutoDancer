"""Git-tracked experiment registry and immutable specification validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_ID = re.compile(r"^EXP-[0-9]{4}$")
STATUSES = frozenset({"planned", "running", "completed", "accepted", "rejected", "inconclusive"})
DECISIONS = frozenset({"accepted", "rejected", "inconclusive"})
STAGES = frozenset({"training", "evaluation", "comparison", "diagnostic"})


class ExperimentError(ValueError):
    """Raised when experiment lineage is incomplete or inconsistent."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    rendered = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ExperimentError(f"Could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExperimentError(f"{path} must contain a YAML object")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    path: Path
    data: dict[str, Any]
    digest: str

    @property
    def experiment_id(self) -> str:
        return str(self.data["id"])

    @property
    def title(self) -> str:
        return str(self.data["title"])

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(str(arm["id"]) for arm in self.data["arms"])

    def validate_arm(self, arm: str) -> None:
        if arm not in self.arms:
            raise ExperimentError(
                f"Unknown arm {arm!r} for {self.experiment_id}; expected one of {self.arms}"
            )

    def components_for_arm(self, arm: str) -> dict[str, Any]:
        self.validate_arm(arm)
        components = dict(self.data["component_versions"])
        arm_spec = next(item for item in self.data["arms"] if item["id"] == arm)
        overrides = arm_spec.get("component_versions", {})
        if not isinstance(overrides, dict):
            raise ExperimentError(f"{self.experiment_id}/{arm} component_versions is invalid")
        components.update(overrides)
        return components


def validate_spec(path: Path, expected_id: str | None = None) -> ExperimentSpec:
    data = load_yaml(path)
    required = (
        "schema_version",
        "id",
        "title",
        "question",
        "hypothesis",
        "changed_blocks",
        "component_versions",
        "invariants",
        "arms",
        "evaluation",
        "decision_rule",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ExperimentError(f"{path} is missing required fields: {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ExperimentError(f"{path} has unsupported schema_version {data['schema_version']!r}")
    experiment_id = data["id"]
    if not isinstance(experiment_id, str) or EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ExperimentError(f"Invalid experiment id {experiment_id!r}; expected EXP-0001")
    if expected_id is not None and experiment_id != expected_id:
        raise ExperimentError(f"Registry id {expected_id} does not match spec id {experiment_id}")
    for field in ("title", "question", "hypothesis", "decision_rule"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise ExperimentError(f"{path}: {field} must be non-empty text")
    for field in ("changed_blocks", "invariants"):
        if not isinstance(data[field], list) or not data[field]:
            raise ExperimentError(f"{path}: {field} must be a non-empty list")
    if not isinstance(data["component_versions"], dict):
        raise ExperimentError(f"{path}: component_versions must be an object")
    missing_components = set(data["changed_blocks"]) - set(data["component_versions"])
    if missing_components:
        raise ExperimentError(
            f"{path}: changed blocks lack component versions: {sorted(missing_components)}"
        )
    arms = data["arms"]
    if not isinstance(arms, list) or not arms:
        raise ExperimentError(f"{path}: arms must be a non-empty list")
    arm_ids: list[str] = []
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("id"), str):
            raise ExperimentError(f"{path}: every arm needs a string id")
        arm_ids.append(arm["id"])
    if len(arm_ids) != len(set(arm_ids)):
        raise ExperimentError(f"{path}: arm ids must be unique")
    if not isinstance(data["evaluation"], dict):
        raise ExperimentError(f"{path}: evaluation must be an object")
    digest = sha256_bytes(canonical_json(data))
    return ExperimentSpec(path.resolve(), data, digest)


class ExperimentStore:
    """Repository source of truth for experiment intent and decisions."""

    def __init__(self, root: Path = Path("experiments")) -> None:
        self.root = root
        self.registry_path = root / "registry.yaml"
        self.baselines_path = root / "baselines.yaml"
        self.components_path = root / "components.yaml"

    def registry(self) -> dict[str, Any]:
        return load_yaml(self.registry_path)

    def _validate_components(self, spec: ExperimentSpec) -> None:
        components = load_yaml(self.components_path)
        if components.get("schema_version") != 1 or not isinstance(components.get("blocks"), dict):
            raise ExperimentError(f"{self.components_path} has an invalid component schema")
        component_sets = [spec.data["component_versions"]]
        component_sets.extend(arm.get("component_versions", {}) for arm in spec.data["arms"])
        for component_set in component_sets:
            if not isinstance(component_set, dict):
                raise ExperimentError(
                    f"{spec.experiment_id} arm component_versions must be an object"
                )
            self._validate_component_set(spec.experiment_id, component_set, components)

    @staticmethod
    def _validate_component_set(
        experiment_id: str,
        component_set: dict[str, Any],
        components: dict[str, Any],
    ) -> None:
        for block, versions in component_set.items():
            definition = components["blocks"].get(block)
            if not isinstance(definition, dict):
                raise ExperimentError(f"{experiment_id} references unknown block {block!r}")
            requested = versions if isinstance(versions, list) else [versions]
            unknown = set(requested) - set(definition.get("versions", {}))
            if unknown:
                raise ExperimentError(
                    f"{experiment_id} references unknown {block} versions: {sorted(unknown)}"
                )

    def component_definition(self, block: str, version: str) -> Any:
        components = load_yaml(self.components_path)
        try:
            return components["blocks"][block]["versions"][version]
        except KeyError as error:
            raise ExperimentError(f"Unknown component version {block}={version}") from error

    def load(self, experiment_id: str) -> ExperimentSpec:
        registry = self.registry()
        entry = registry.get("experiments", {}).get(experiment_id)
        if not isinstance(entry, dict):
            raise ExperimentError(f"Experiment {experiment_id} is not registered")
        path = self.root.parent / str(entry.get("spec"))
        spec = validate_spec(path, experiment_id)
        if spec.digest != entry.get("spec_sha256"):
            raise ExperimentError(
                f"{experiment_id} specification changed after registration: "
                f"expected {entry.get('spec_sha256')}, got {spec.digest}. Create a new experiment."
            )
        return spec

    def validate(self) -> list[ExperimentSpec]:
        registry = self.registry()
        if registry.get("schema_version") != 1 or not isinstance(registry.get("experiments"), dict):
            raise ExperimentError(f"{self.registry_path} has an invalid registry schema")
        specs: list[ExperimentSpec] = []
        for experiment_id, entry in registry["experiments"].items():
            if not isinstance(entry, dict) or entry.get("status") not in STATUSES:
                raise ExperimentError(f"{experiment_id} has an invalid registry entry/status")
            spec = self.load(experiment_id)
            specs.append(spec)
            self._validate_components(spec)
            decision_path = self.root / experiment_id / "decision.json"
            if entry["status"] in DECISIONS:
                if not decision_path.is_file():
                    raise ExperimentError(
                        f"{experiment_id} is {entry['status']} but has no decision.json"
                    )
                try:
                    decision = json.loads(decision_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise ExperimentError(f"Invalid decision file {decision_path}") from error
                if decision.get("experiment_id") != experiment_id:
                    raise ExperimentError(f"{decision_path} has the wrong experiment_id")
                if decision.get("outcome") != entry["status"]:
                    raise ExperimentError(f"{decision_path} outcome does not match registry status")
        baselines = load_yaml(self.baselines_path)
        if baselines.get("schema_version") != 1 or not isinstance(baselines.get("baselines"), dict):
            raise ExperimentError(f"{self.baselines_path} has an invalid baseline schema")
        for name, baseline in baselines["baselines"].items():
            if not isinstance(baseline, dict):
                raise ExperimentError(f"Baseline {name} must be an object")
            experiment_id = baseline.get("experiment_id")
            if experiment_id not in registry["experiments"]:
                raise ExperimentError(
                    f"Baseline {name} references unknown experiment {experiment_id!r}"
                )
            checkpoint_hash = baseline.get("checkpoint_sha256")
            if checkpoint_hash is not None and (
                not isinstance(checkpoint_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash) is None
            ):
                raise ExperimentError(f"Baseline {name} has an invalid checkpoint hash")
        return specs

    def register(
        self, spec_path: Path, *, status: str = "planned", historical: bool = False
    ) -> ExperimentSpec:
        if status not in STATUSES:
            raise ExperimentError(f"Invalid experiment status {status!r}")
        spec = validate_spec(spec_path)
        self._validate_components(spec)
        registry = self.registry()
        entries = registry.setdefault("experiments", {})
        if spec.experiment_id in entries:
            raise ExperimentError(f"Experiment {spec.experiment_id} is already registered")
        try:
            relative = spec.path.relative_to(self.root.parent.resolve()).as_posix()
        except ValueError as error:
            raise ExperimentError(
                "Experiment specifications must live inside the repository"
            ) from error
        entries[spec.experiment_id] = {
            "spec": relative,
            "spec_sha256": spec.digest,
            "status": status,
            "historical": bool(historical),
            "registered_at": datetime.now(UTC).isoformat(),
        }
        atomic_yaml(self.registry_path, registry)
        return spec

    def set_status(self, experiment_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ExperimentError(f"Invalid experiment status {status!r}")
        registry = self.registry()
        if experiment_id not in registry.get("experiments", {}):
            raise ExperimentError(f"Experiment {experiment_id} is not registered")
        registry["experiments"][experiment_id]["status"] = status
        registry["experiments"][experiment_id]["updated_at"] = datetime.now(UTC).isoformat()
        atomic_yaml(self.registry_path, registry)

    def write_decision(self, experiment_id: str, decision: dict[str, Any]) -> Path:
        self.load(experiment_id)
        outcome = decision.get("outcome")
        if outcome not in DECISIONS:
            raise ExperimentError(f"Decision outcome must be one of {sorted(DECISIONS)}")
        path = self.root / experiment_id / "decision.json"
        if path.exists():
            raise ExperimentError(
                f"{path} already exists; decisions are append-only via new experiments"
            )
        decision = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "decided_at": datetime.now(UTC).isoformat(),
            **decision,
        }
        atomic_json(path, decision)
        self.set_status(experiment_id, str(outcome))
        return path

    def promote_baseline(self, name: str, record: dict[str, Any]) -> None:
        baselines = load_yaml(self.baselines_path)
        if name in baselines.setdefault("baselines", {}):
            raise ExperimentError(
                f"Baseline {name!r} already exists; use a new versioned baseline name"
            )
        baselines["baselines"][name] = {
            **record,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        atomic_yaml(self.baselines_path, baselines)
