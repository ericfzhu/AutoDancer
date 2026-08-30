"""Checkpointable competence-driven scheduling for legal live-game starts."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from autodancer.curriculum import EpisodeResetSpec

SUCCESS_OUTCOMES = frozenset({"curriculum_complete", "victory"})


def load_adaptive_curriculum_config(path: str | Path) -> AdaptiveCurriculumConfig:
    """Load a strict schema-1 adaptive curriculum configuration."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load adaptive curriculum config {source}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("adaptive curriculum config must be a schema_version 1 object")
    unknown = set(payload) - {"schema_version", "config"}
    if unknown:
        raise ValueError(f"unknown adaptive curriculum config fields: {sorted(unknown)}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("adaptive curriculum config.config must be an object")
    allowed = set(AdaptiveCurriculumConfig.__dataclass_fields__)
    unknown_config = set(config) - allowed
    if unknown_config:
        raise ValueError(f"unknown adaptive curriculum parameters: {sorted(unknown_config)}")
    return AdaptiveCurriculumConfig(**config)


def wilson_interval(successes: int, samples: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli outcome."""

    if samples < 0 or successes < 0 or successes > samples:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= samples")
    if samples == 0:
        return 0.0, 1.0
    estimate = successes / samples
    z_squared = z * z
    denominator = 1.0 + z_squared / samples
    centre = estimate + z_squared / (2.0 * samples)
    variance = estimate * (1.0 - estimate) / samples
    correction = z_squared / (4.0 * samples * samples)
    radius = z * sqrt(variance + correction)
    lower = max(0.0, (centre - radius) / denominator)
    upper = min(1.0, (centre + radius) / denominator)
    return lower, upper


@dataclass(frozen=True, slots=True)
class AdaptiveCurriculumConfig:
    """Promotion and episode-allocation contract for an ordered curriculum."""

    window_size: int = 100
    minimum_samples: int = 40
    promotion_lower_bound: float = 0.60
    demotion_upper_bound: float = 0.10
    acquisition_probability: float = 0.65
    mastery_replay_probability: float = 0.25
    frontier_probe_probability: float = 0.10
    confidence_z: float = 1.96

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.minimum_samples <= 0:
            raise ValueError("adaptive curriculum sample counts must be positive")
        if self.minimum_samples > self.window_size:
            raise ValueError("minimum_samples cannot exceed window_size")
        probabilities = (
            self.acquisition_probability,
            self.mastery_replay_probability,
            self.frontier_probe_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("adaptive curriculum probabilities must be in [0, 1]")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("adaptive curriculum probabilities must sum to one")
        if not 0.0 <= self.demotion_upper_bound < self.promotion_lower_bound <= 1.0:
            raise ValueError("adaptive curriculum confidence thresholds are invalid")
        if self.confidence_z <= 0:
            raise ValueError("confidence_z must be positive")


@dataclass(slots=True)
class AdaptiveEpisodeCurriculumSchedule:
    """Sample acquisition, mastered replay, and frontier starts by competence.

    Boundaries are ordered from easiest (closest to the goal) to hardest. Only
    gameplay-valid outcomes enter competence windows. Promotion and demotion use
    confidence bounds, preventing a handful of lucky episodes from changing the
    live start distribution.
    """

    base_seed: int
    slots: int
    boundaries: tuple[EpisodeResetSpec, ...]
    config: AdaptiveCurriculumConfig = field(default_factory=AdaptiveCurriculumConfig)
    _rngs: list[np.random.Generator] = field(init=False, repr=False)
    _draws: list[int] = field(init=False, repr=False)
    _windows: dict[str, deque[bool]] = field(init=False, repr=False)
    _selected: dict[str, int] = field(init=False, repr=False)
    _ignored_infrastructure: dict[str, int] = field(init=False, repr=False)
    _active_index: int = field(init=False, default=0)
    _mastered: set[int] = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.slots <= 0:
            raise ValueError("adaptive curriculum slots must be positive")
        if not self.boundaries:
            raise ValueError("adaptive curriculum requires at least one boundary")
        ids = [boundary.id for boundary in self.boundaries]
        if len(set(ids)) != len(ids):
            raise ValueError("adaptive curriculum boundary ids must be unique")
        sequence = np.random.SeedSequence([int(self.base_seed), 0xADA9711E])
        self._rngs = [np.random.default_rng(child) for child in sequence.spawn(self.slots)]
        self._draws = [0] * self.slots
        self._windows = {item.id: deque(maxlen=self.config.window_size) for item in self.boundaries}
        self._selected = {item.id: 0 for item in self.boundaries}
        self._ignored_infrastructure = {item.id: 0 for item in self.boundaries}
        self._mastered = set()
        self._lock = Lock()

    def _allocation(self) -> tuple[list[int], np.ndarray]:
        weights: dict[int, float] = {}

        def add(index: int, value: float) -> None:
            weights[index] = weights.get(index, 0.0) + value

        add(self._active_index, self.config.acquisition_probability)
        replay = sorted(self._mastered)
        if replay:
            share = self.config.mastery_replay_probability / len(replay)
            for index in replay:
                add(index, share)
        else:
            add(self._active_index, self.config.mastery_replay_probability)
        frontier = min(self._active_index + 1, len(self.boundaries) - 1)
        add(frontier, self.config.frontier_probe_probability)
        indices = sorted(weights)
        probabilities = np.asarray([weights[index] for index in indices], dtype=np.float64)
        probabilities /= probabilities.sum()
        return indices, probabilities

    def next(self, slot: int) -> EpisodeResetSpec:
        if not 0 <= slot < self.slots:
            raise IndexError(f"Curriculum slot {slot} is outside capacity {self.slots}")
        with self._lock:
            indices, probabilities = self._allocation()
            index = indices[int(self._rngs[slot].choice(len(indices), p=probabilities))]
            boundary = self.boundaries[index]
            self._draws[slot] += 1
            self._selected[boundary.id] += 1
            return boundary

    def record_outcome(
        self, spec: EpisodeResetSpec, status: str, *, infrastructure_valid: bool = True
    ) -> None:
        indices = {boundary.id: index for index, boundary in enumerate(self.boundaries)}
        if spec.id not in indices:
            raise ValueError(f"reset specification {spec.id!r} is not in this schedule")
        with self._lock:
            if not infrastructure_valid:
                self._ignored_infrastructure[spec.id] += 1
                return
            self._windows[spec.id].append(str(status) in SUCCESS_OUTCOMES)
            index = indices[spec.id]
            _, samples, _, upper = self._bounds(index)
            if (
                index in self._mastered
                and samples >= self.config.minimum_samples
                and upper < self.config.demotion_upper_bound
            ):
                self._mastered = {value for value in self._mastered if value < index}
                self._active_index = min(self._active_index, index)
            self._update_active_boundary()

    def _bounds(self, index: int) -> tuple[int, int, float, float]:
        values = self._windows[self.boundaries[index].id]
        successes = sum(values)
        lower, upper = wilson_interval(successes, len(values), self.config.confidence_z)
        return successes, len(values), lower, upper

    def _update_active_boundary(self) -> None:
        _, samples, lower, upper = self._bounds(self._active_index)
        if samples < self.config.minimum_samples:
            return
        if lower >= self.config.promotion_lower_bound:
            self._mastered.add(self._active_index)
            if self._active_index < len(self.boundaries) - 1:
                self._active_index += 1
            return
        if upper < self.config.demotion_upper_bound and self._active_index > 0:
            self._mastered.discard(self._active_index - 1)
            self._active_index -= 1

    def specification(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "adaptive-competence-v1",
            "boundaries": [boundary.as_dict() for boundary in self.boundaries],
            "config": asdict(self.config),
        }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.specification(),
                "base_seed": int(self.base_seed),
                "slots": self.slots,
                "active_index": self._active_index,
                "mastered_indices": sorted(self._mastered),
                "draws": list(self._draws),
                "selected": dict(self._selected),
                "ignored_infrastructure": dict(self._ignored_infrastructure),
                "windows": {key: list(values) for key, values in self._windows.items()},
                "rng_states": [deepcopy(rng.bit_generator.state) for rng in self._rngs],
            }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            **self.specification(),
            "base_seed": int(self.base_seed),
            "slots": self.slots,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("adaptive curriculum state does not match this run")
        ids = {boundary.id for boundary in self.boundaries}
        draws = [int(value) for value in state.get("draws", [])]
        selected = {str(key): int(value) for key, value in dict(state.get("selected", {})).items()}
        ignored = {
            str(key): int(value)
            for key, value in dict(state.get("ignored_infrastructure", {})).items()
        }
        windows = {
            str(key): [bool(value) for value in values]
            for key, values in dict(state.get("windows", {})).items()
        }
        rng_states = list(state.get("rng_states", []))
        active = int(state.get("active_index", -1))
        mastered = {int(value) for value in state.get("mastered_indices", [])}
        valid_indices = set(range(len(self.boundaries)))
        if len(draws) != self.slots or len(rng_states) != self.slots:
            raise ValueError("adaptive curriculum state has the wrong capacity")
        if set(selected) != ids or set(ignored) != ids or set(windows) != ids:
            raise ValueError("adaptive curriculum state has the wrong boundary ids")
        if active not in valid_indices or not mastered <= valid_indices:
            raise ValueError("adaptive curriculum state has invalid boundary indices")
        if any(value < 0 for value in (*draws, *selected.values(), *ignored.values())):
            raise ValueError("adaptive curriculum state counters must be non-negative")
        if any(len(values) > self.config.window_size for values in windows.values()):
            raise ValueError("adaptive curriculum state window exceeds its configured size")
        for rng, rng_state in zip(self._rngs, rng_states, strict=True):
            rng.bit_generator.state = deepcopy(rng_state)
        with self._lock:
            self._draws = draws
            self._selected = selected
            self._ignored_infrastructure = ignored
            self._windows = {
                key: deque(values, maxlen=self.config.window_size)
                for key, values in windows.items()
            }
            self._active_index = active
            self._mastered = mastered

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            boundaries = []
            for index, boundary in enumerate(self.boundaries):
                successes, samples, lower, upper = self._bounds(index)
                boundaries.append(
                    {
                        **boundary.as_dict(),
                        "index": index,
                        "active": index == self._active_index,
                        "mastered": index in self._mastered,
                        "successes": successes,
                        "samples": samples,
                        "success_rate": successes / samples if samples else None,
                        "wilson_lower": lower,
                        "wilson_upper": upper,
                        "selected": self._selected[boundary.id],
                        "ignored_infrastructure": self._ignored_infrastructure[boundary.id],
                    }
                )
            indices, probabilities = self._allocation()
            return {
                "active_index": self._active_index,
                "allocation": {
                    self.boundaries[index].id: float(probability)
                    for index, probability in zip(indices, probabilities, strict=True)
                },
                "boundaries": boundaries,
            }
