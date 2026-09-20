"""Offline contracts for the next harness; not yet a live protocol revision.

Facts describe current state, effects describe an acknowledged transition, and
timing never infers game turns from bridge sequence numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from autodancer.constants import Action


class InformationSource(StrEnum):
    STATIC_RULE = "static_rule"
    VISIBLE = "visible"
    REMEMBERED = "remembered"
    INTERNAL = "internal"
    DIAGNOSTIC = "diagnostic"


class InformationPolicy(StrEnum):
    PLAYER = "player"
    PRIVILEGED = "privileged"

    def permits(self, source: InformationSource) -> bool:
        # Debug IDs, protocol state and future RNG are never policy features.
        return source != InformationSource.DIAGNOSTIC and (
            self == InformationPolicy.PRIVILEGED or source != InformationSource.INTERNAL
        )


@dataclass(frozen=True)
class StateFact:
    value: float | int | bool | None
    known: bool
    applicable: bool
    source: InformationSource

    def __post_init__(self) -> None:
        if not isinstance(self.source, InformationSource):
            raise ValueError("Fact source must be an InformationSource")
        if not self.applicable and self.known:
            raise ValueError("An inapplicable fact cannot be known")
        if self.known != (self.value is not None):
            raise ValueError("Unknown/inapplicable values must be None; known values need a value")
        if self.value is not None and (
            not isinstance(self.value, (float, int, bool)) or not math.isfinite(self.value)
        ):
            raise ValueError("State facts must contain finite numeric values")


def project_facts(
    facts: Mapping[str, StateFact], policy: InformationPolicy
) -> dict[str, StateFact]:
    """Refuse accidental leakage rather than silently change the input schema."""
    if not isinstance(policy, InformationPolicy):
        raise ValueError("Select an explicit information policy")
    blocked = [name for name, fact in facts.items() if not policy.permits(fact.source)]
    if blocked:
        raise ValueError(f"Fields outside {policy.value} information policy: {sorted(blocked)}")
    return dict(facts)


class Availability(StrEnum):
    EXECUTABLE = "executable"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ControlState:
    action: Action
    availability: Availability
    owner: str | None = None
    pending_mode: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action) or not isinstance(self.availability, Availability):
            raise ValueError("Control requires an Action and Availability")
        if (self.availability == Availability.UNAVAILABLE) != bool(self.unavailable_reason):
            raise ValueError("Only unavailable controls require an unavailability reason")


def executable_mask(controls: tuple[ControlState, ...]) -> tuple[int, ...]:
    """Conservative contract: unknown semantics block collection, not one input.

    This deliberately does not prune redundancy or encode strategic preferences.
    An input that advances time remains executable even if its item effect fails.
    Terminal observations should bypass action selection entirely.
    """
    by_action = {control.action: control for control in controls}
    if len(by_action) != len(controls) or set(by_action) != set(Action):
        raise ValueError("Exactly one descriptor for every action is required")
    if any(c.availability == Availability.UNKNOWN for c in controls):
        raise ValueError("Unknown action semantics require qualification before collection")
    mask = tuple(
        int(by_action[action].availability == Availability.EXECUTABLE) for action in Action
    )
    if not any(mask):
        raise ValueError("Nonterminal state has no executable input")
    return mask


@dataclass(frozen=True)
class TransitionTiming:
    command_id: int
    game_turns_advanced: int | None

    def __post_init__(self) -> None:
        if type(self.command_id) is not int or self.command_id < 0:
            raise ValueError("command_id must be a nonnegative integer")
        turns = self.game_turns_advanced
        if turns is not None and (type(turns) is not int or turns < 0):
            raise ValueError("Game turns must be a nonnegative integer or unknown")

    def discount(self, gamma: float, *, clock: str = "decision") -> float:
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must be finite and in (0, 1]")
        if clock == "decision":
            return gamma
        if clock != "game_turn":
            raise ValueError("clock must be decision or game_turn")
        if self.game_turns_advanced is None:
            raise ValueError("Game-turn discounting requires measured duration")
        return gamma**self.game_turns_advanced


@dataclass(frozen=True)
class ObservedEffect:
    command_id: int
    kind: str
    evidence: str
    authoritative: bool


def validate_effects(timing: TransitionTiming, effects: tuple[ObservedEffect, ...]) -> None:
    for effect in effects:
        if effect.command_id != timing.command_id:
            raise ValueError("Effect belongs to another command")
        if not effect.kind or not effect.evidence:
            raise ValueError("Effects require a kind and an evidence reference")
