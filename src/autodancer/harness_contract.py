"""Information, task and timing contracts for the symbolic harness.

Facts describe current state, effects describe an acknowledged transition, and
timing never infers game turns from bridge sequence numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from autodancer.constants import Action


def task_contract() -> dict:
    """The task is distinct from the boss identity and the collection budget."""
    return {
        "version": 1,
        "context_fields": ["objective", "target_level"],
        "objectives": {"win_run": 1, "reach_level": 2},
        "target_level_zero": "not_applicable",
        "success": "won_or_reached_declared_target",
        "death": "terminal_failure",
        "collection_limit": "truncate_and_bootstrap_final_state",
        "abort": "invalid_run_no_learning_transition",
        "discount_clock": "acknowledged_decision",
        "finite_horizon": False,
    }


def information_contract(profile: str, *, world: bool) -> dict:
    """Declare actual inputs, including privilege remaining after HUD filtering.

    `player-v1` is a historical equipment profile name, not a certification of
    pixel/audio equivalence. Both live profiles are explicitly symbolic.
    """
    if profile not in {"legacy", "internal-v1", "player-v1"}:
        raise ValueError("Unknown information profile")
    if world and profile == "legacy":
        raise ValueError("World information requires an explicit equipment profile")
    return {
        "version": 1,
        "profile": profile,
        "scope": "world-and-equipment" if world else "equipment-and-legacy-grid",
        "human_equivalent": False,
        "actor_critic": "same_inputs",
        "grid": "revealed_coarse_geometry" if world else "visible_engine_entity_aggregates",
        "map_memory": "revealed_terrain_plus_visit_history_including_map_item_reveal",
        "player": "engine_health_gold_groove_position_level_boss_and_music_state",
        "equipment": "HUD_filtered" if profile == "player-v1" else "engine_state",
        "world": (
            "display_filtered_exact_identities_terrain_and_selected_engine_facts"
            if profile == "player-v1"
            else "visible_cell_exact_identities_terrain_and_engine_facts"
        )
        if world
        else "legacy_grid",
        "remaining_privilege": [
            "exact_entity_identity_including_disguises",
            "exact_terrain_identity_including_embedded_loot" if world else "legacy_entity_counters",
            "engine_component_markers_and_relations",
            "engine_music_and_player_state",
            "current_engine_terrain_on_revealed_but_not_visible_cells",
        ],
        "previous_action": "own_input_history",
        "previous_reward": "excluded_by_model" if world else "configured_reward_feedback",
        "decision_count": "acknowledged_episode_inputs" if world else "bridge_sequence",
        "action_mask": "all_11_input_attempts" if profile != "legacy" else "legacy_rules",
        "task_context": "explicit_objective_and_target" if world else "unconditioned",
        "diagnostics": "seed_run_ids_raw_entity_handles_transport_and_raw_events_not_model_inputs",
        "future_rng": "excluded",
        "reward_sources": "declared_engine_events_and_tracker_history_may_be_privileged",
    }


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
