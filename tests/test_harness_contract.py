import pytest

from autodancer.constants import Action
from autodancer.harness_contract import (
    Availability,
    ControlState,
    InformationPolicy,
    InformationSource,
    ObservedEffect,
    StateFact,
    TransitionTiming,
    executable_mask,
    project_facts,
    validate_effects,
)


def test_zero_unknown_and_inapplicable_are_distinct():
    known_zero = StateFact(0, True, True, InformationSource.VISIBLE)
    unknown = StateFact(None, False, True, InformationSource.VISIBLE)
    absent = StateFact(None, False, False, InformationSource.STATIC_RULE)
    assert len({known_zero, unknown, absent}) == 3
    with pytest.raises(ValueError):
        StateFact(0, False, True, InformationSource.VISIBLE)


def test_information_policy_rejects_undeclared_internal_and_diagnostic_facts():
    facts = {"ammo": StateFact(2, True, True, InformationSource.INTERNAL)}
    with pytest.raises(ValueError, match="ammo"):
        project_facts(facts, InformationPolicy.PLAYER)
    assert project_facts(facts, InformationPolicy.PRIVILEGED) == facts
    facts["rng"] = StateFact(42, True, True, InformationSource.DIAGNOSTIC)
    with pytest.raises(ValueError, match="rng"):
        project_facts(facts, InformationPolicy.PRIVILEGED)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_state_cannot_enter_policy_inputs(value):
    with pytest.raises(ValueError, match="finite"):
        StateFact(value, True, True, InformationSource.VISIBLE)


def test_visible_and_remembered_facts_are_permitted_without_privileged_state():
    facts = {
        "armed": StateFact(True, True, True, InformationSource.VISIBLE),
        "seen_stairs": StateFact(True, True, True, InformationSource.REMEMBERED),
    }
    assert project_facts(facts, InformationPolicy.PLAYER) == facts


def test_masks_leave_walking_waiting_and_redundant_inputs_available():
    controls = tuple(ControlState(a, Availability.EXECUTABLE) for a in Action)
    assert executable_mask(controls) == (1,) * len(Action)
    unknown = (ControlState(Action.UP, Availability.UNKNOWN), *controls[1:])
    with pytest.raises(ValueError, match="Unknown"):
        executable_mask(unknown)
    with pytest.raises(ValueError, match="Exactly one"):
        executable_mask(controls[:-1])
    with pytest.raises(ValueError, match="Exactly one"):
        executable_mask((*controls, controls[0]))


def test_all_masked_is_a_contract_failure():
    controls = tuple(
        ControlState(a, Availability.UNAVAILABLE, unavailable_reason="engine restriction")
        for a in Action
    )
    with pytest.raises(ValueError, match="no executable"):
        executable_mask(controls)


@pytest.mark.parametrize("turns", [0, 1, 3])
def test_decision_and_game_turn_clocks_are_explicit(turns):
    timing = TransitionTiming(42, turns)
    assert timing.discount(0.99) == 0.99
    assert timing.discount(0.99, clock="game_turn") == pytest.approx(0.99**turns)


def test_unknown_duration_cannot_be_inferred_from_command_id():
    timing = TransitionTiming(1000, None)
    assert timing.discount(0.99) == 0.99
    with pytest.raises(ValueError, match="measured"):
        timing.discount(0.99, clock="game_turn")


def test_effects_are_post_action_evidence_and_allow_multiple_consequences():
    timing = TransitionTiming(7, 1)
    validate_effects(
        timing,
        (
            ObservedEffect(7, "attack", "event:7:0", True),
            ObservedEffect(7, "move", "position-delta:7", False),
        ),
    )
    with pytest.raises(ValueError, match="another command"):
        validate_effects(timing, (ObservedEffect(8, "move", "event:8:0", True),))
