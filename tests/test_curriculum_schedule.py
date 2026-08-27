from __future__ import annotations

import json

import pytest

from autodancer.curriculum import (
    EpisodeCurriculumSchedule,
    EpisodeResetSpec,
    WeightedResetSpec,
    load_curriculum_mixture,
)


def entries() -> tuple[WeightedResetSpec, ...]:
    return (
        WeightedResetSpec(
            EpisodeResetSpec("reduced", 4, 5, "player10"), 0.8
        ),
        WeightedResetSpec(
            EpisodeResetSpec("mastered-replay", 4, 5, "player20"), 0.2
        ),
    )


def test_curriculum_schedule_is_per_slot_deterministic_and_exactly_resumable() -> None:
    first = EpisodeCurriculumSchedule(901, 2, entries())
    second = EpisodeCurriculumSchedule(901, 2, entries())
    assert [first.next(0).id for _ in range(20)] == [
        second.next(0).id for _ in range(20)
    ]
    assert [first.next(1).id for _ in range(20)] == [
        second.next(1).id for _ in range(20)
    ]
    first.record_outcome(entries()[0].spec, "dead")
    state = first.state_dict()

    resumed = EpisodeCurriculumSchedule(901, 2, entries())
    resumed.load_state_dict(state)
    assert [first.next(0).id for _ in range(10)] == [
        resumed.next(0).id for _ in range(10)
    ]
    assert resumed.state_dict()["outcomes"]["reduced"] == {"dead": 1}


def test_curriculum_schedule_rejects_mismatched_resume_distribution() -> None:
    source = EpisodeCurriculumSchedule(4, 1, entries())
    source.next(0)
    reversed_entries = tuple(reversed(entries()))
    target = EpisodeCurriculumSchedule(4, 1, reversed_entries)
    with pytest.raises(ValueError, match="does not match"):
        target.load_state_dict(source.state_dict())


def test_curriculum_mixture_loader_validates_and_preserves_exact_weights(tmp_path) -> None:
    path = tmp_path / "mixture.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [entry.as_dict() for entry in entries()],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_curriculum_mixture(path)
    assert [entry.as_dict() for entry in loaded] == [entry.as_dict() for entry in entries()]


@pytest.mark.parametrize(
    "value, message",
    [
        ({"id": "bad id"}, "reset id"),
        ({"id": "bad-profile", "profile": "unknown"}, "profile"),
        (
            {"id": "bad-target", "start_level": 4, "target_level": 4},
            "after start_level",
        ),
    ],
)
def test_episode_reset_spec_rejects_ambiguous_or_invalid_values(value, message) -> None:
    with pytest.raises(ValueError, match=message):
        EpisodeResetSpec.from_mapping(value)
