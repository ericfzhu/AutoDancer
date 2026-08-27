from __future__ import annotations

from autodancer.training.progress import deeper_level, level_progress


def test_level_progress_orders_boss_before_next_zone() -> None:
    assert level_progress(1, 4) == 4
    assert level_progress(2, 1) == 5
    assert deeper_level((1, 4), (2, 1)) == (2, 1)


def test_deeper_level_does_not_mix_zone_and_floor_maxima() -> None:
    current = deeper_level((1, 4), (2, 1))
    assert current == (2, 1)
    assert current != (2, 4)
