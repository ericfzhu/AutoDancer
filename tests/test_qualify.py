from argparse import Namespace
from pathlib import Path

from autodancer.live.qualify import (
    _configuration,
    _memory_growth,
    _sustained_memory_growth,
)


def test_qualification_uses_the_production_episode_turn_cap(tmp_path: Path) -> None:
    arguments = Namespace(
        game_dir=tmp_path / "game",
        mod_dir=tmp_path / "mod",
        num_instances=8,
        transitions_per_worker=125_000,
        run_dir=tmp_path / "run",
        startup_timeout=60.0,
        turn_timeout=10.0,
        reset_timeout=60.0,
        affinity="none",
    )

    configuration = _configuration(arguments, 8, "natural-soak")

    assert configuration.max_turns == 10_000
    assert configuration.qualification_startup_fault_slot is None


def test_forced_recovery_keeps_the_same_turn_contract(tmp_path: Path) -> None:
    arguments = Namespace(
        game_dir=tmp_path / "game",
        mod_dir=tmp_path / "mod",
        num_instances=8,
        transitions_per_worker=1_000_000,
        run_dir=tmp_path / "run",
        startup_timeout=60.0,
        turn_timeout=10.0,
        reset_timeout=60.0,
        affinity="none",
    )

    configuration = _configuration(arguments, 8, "forced-recovery")

    assert configuration.max_turns == 10_000
    assert configuration.qualification_startup_fault_slot == 0


def test_sustained_memory_growth_rejects_a_terminal_linear_leak() -> None:
    samples = [1_000_000_000 + index * 1_500_000 for index in range(125)]

    assert _sustained_memory_growth(samples) > 0.05


def test_sustained_memory_growth_accepts_a_cache_step_then_plateau() -> None:
    samples = [1_000_000_000] * 85 + [1_100_000_000] * 40

    assert _memory_growth(samples) > 0.05
    assert _sustained_memory_growth(samples) == 0.0


def test_sustained_memory_growth_is_robust_to_gc_sawteeth() -> None:
    samples = [
        1_000_000_000 + (index % 4) * 8_000_000
        for index in range(125)
    ]

    assert _sustained_memory_growth(samples) <= 0.05


def test_sustained_memory_growth_accepts_a_terminal_gc_upswing() -> None:
    samples = [1_400_000_000] * 105 + [
        value * 1_000_000
        for value in (
            1_440,
            1_409,
            1_410,
            1_417,
            1_394,
            1_394,
            1_402,
            1_394,
            1_394,
            1_394,
            1_402,
            1_394,
            1_394,
            1_414,
            1_420,
            1_488,
            1_448,
            1_450,
            1_518,
            1_450,
        )
    ]

    assert _sustained_memory_growth(samples) <= 0.05


def test_sustained_memory_growth_rejects_a_noisy_full_half_leak() -> None:
    samples = [1_000_000_000] * 62 + [
        1_000_000_000 + index * 2_000_000 + (index % 4) * 3_000_000
        for index in range(63)
    ]

    assert _sustained_memory_growth(samples) > 0.05


def test_sustained_memory_growth_rejects_a_late_rise() -> None:
    samples = [1_000_000_000] * 105 + [
        1_000_000_000 + index * 3_000_000 for index in range(20)
    ]

    assert _sustained_memory_growth(samples) > 0.05
