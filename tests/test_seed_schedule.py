from __future__ import annotations

import json

import pytest

from autodancer.training.seed_schedule import (
    TrainingSeedSchedule,
    parse_training_seed_pool,
)


def test_uniform_pool_is_deterministic_and_slot_local() -> None:
    first = TrainingSeedSchedule(123, 2, (10, 20, 30))
    second = TrainingSeedSchedule(123, 2, (10, 20, 30))

    first_slot_zero = [first.next(0) for _ in range(8)]
    first_slot_one = [first.next(1) for _ in range(8)]
    second_slot_zero = [second.next(0) for _ in range(8)]
    second_slot_one = [second.next(1) for _ in range(8)]

    assert first_slot_zero == second_slot_zero
    assert first_slot_one == second_slot_one
    assert set(first_slot_zero + first_slot_one) <= {10, 20, 30}
    counts = first.state_dict()["pool_counts"]
    assert sum(counts.values()) == 16
    assert counts == second.state_dict()["pool_counts"]


def test_schedule_state_resumes_exact_next_draws() -> None:
    schedule = TrainingSeedSchedule(456, 2, (100, 200, 300, 400))
    schedule.next(0)
    schedule.next(1)
    state = schedule.state_dict()
    json.dumps(state)
    expected = [schedule.next(0), schedule.next(1), schedule.next(0)]

    resumed = TrainingSeedSchedule(456, 2, (100, 200, 300, 400))
    resumed.load_state_dict(state)

    assert [resumed.next(0), resumed.next(1), resumed.next(0)] == expected
    assert resumed.state_dict()["pool_counts"] == schedule.state_dict()["pool_counts"]


def test_old_schedule_state_preserves_unknown_historical_exposure() -> None:
    source = TrainingSeedSchedule(456, 2, (100, 200, 300))
    source.next(0)
    source.next(1)
    old_state = source.state_dict()
    del old_state["pool_counts"]
    del old_state["unattributed_draws"]

    resumed = TrainingSeedSchedule(456, 2, (100, 200, 300))
    resumed.load_state_dict(old_state)
    restored = resumed.state_dict()

    assert restored["pool_counts"] == {"100": 0, "200": 0, "300": 0}
    assert restored["unattributed_draws"] == 2


def test_schedule_rejects_counts_that_disagree_with_draws() -> None:
    schedule = TrainingSeedSchedule(456, 1, (100, 200))
    schedule.next(0)
    state = schedule.state_dict()
    state["pool_counts"] = {"100": 0, "200": 0}

    with pytest.raises(ValueError, match="total draws"):
        TrainingSeedSchedule(456, 1, (100, 200)).load_state_dict(state)


def test_unbounded_schedule_resume_does_not_invent_pool_exposure() -> None:
    source = TrainingSeedSchedule(456, 1)
    source.next(0)
    resumed = TrainingSeedSchedule(456, 1)

    resumed.load_state_dict(source.state_dict())

    assert resumed.state_dict()["pool_counts"] == {}
    assert resumed.state_dict()["unattributed_draws"] == 0


def test_schedule_rejects_mismatched_pool_state() -> None:
    source = TrainingSeedSchedule(1, 1, (10, 20))
    target = TrainingSeedSchedule(1, 1, (10, 30))

    with pytest.raises(ValueError, match="does not match"):
        target.load_state_dict(source.state_dict())


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100-103", (100, 101, 102, 103)),
        ("4,8,15,16", (4, 8, 15, 16)),
    ],
)
def test_parse_training_seed_pool(text: str, expected: tuple[int, ...]) -> None:
    assert parse_training_seed_pool(text) == expected


@pytest.mark.parametrize("text", ["", "3-1", "1,1", "-2,3", "hello"])
def test_parse_training_seed_pool_rejects_invalid_values(text: str) -> None:
    with pytest.raises(ValueError):
        parse_training_seed_pool(text)
