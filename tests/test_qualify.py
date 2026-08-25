from argparse import Namespace
from pathlib import Path

from autodancer.live.qualify import _configuration


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
