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
