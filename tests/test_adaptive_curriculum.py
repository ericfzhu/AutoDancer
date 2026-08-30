from __future__ import annotations

from autodancer.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    AdaptiveEpisodeCurriculumSchedule,
    load_adaptive_curriculum_config,
    wilson_interval,
)
from autodancer.curriculum import EpisodeResetSpec


def boundaries() -> tuple[EpisodeResetSpec, ...]:
    return (
        EpisodeResetSpec("boss-player20", 4, 5, "player20"),
        EpisodeResetSpec("boss-player10", 4, 5, "player10"),
        EpisodeResetSpec("boss-normal", 4, 5, "normal"),
        EpisodeResetSpec("floor3-zone2", 3, 5, "normal"),
    )


def config() -> AdaptiveCurriculumConfig:
    return AdaptiveCurriculumConfig(
        window_size=10,
        minimum_samples=5,
        promotion_lower_bound=0.20,
        demotion_upper_bound=0.10,
    )


def test_wilson_interval_handles_empty_and_certain_samples() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lower, upper = wilson_interval(10, 10)
    assert 0.72 < lower < 0.73
    assert upper == 1.0


def test_promotes_only_after_confident_valid_gameplay_success() -> None:
    schedule = AdaptiveEpisodeCurriculumSchedule(7, 1, boundaries(), config())
    first = boundaries()[0]
    for _ in range(4):
        schedule.record_outcome(first, "curriculum_complete")
    assert schedule.diagnostics()["active_index"] == 0
    schedule.record_outcome(first, "dead", infrastructure_valid=False)
    assert schedule.diagnostics()["active_index"] == 0
    schedule.record_outcome(first, "curriculum_complete")
    diagnostics = schedule.diagnostics()
    assert diagnostics["active_index"] == 1
    assert diagnostics["boundaries"][0]["mastered"] is True
    assert diagnostics["boundaries"][0]["ignored_infrastructure"] == 1


def test_allocation_preserves_mastery_replay_and_frontier_probe() -> None:
    schedule = AdaptiveEpisodeCurriculumSchedule(7, 1, boundaries(), config())
    first = boundaries()[0]
    for _ in range(5):
        schedule.record_outcome(first, "curriculum_complete")
    assert schedule.diagnostics()["allocation"] == {
        "boss-player20": 0.25,
        "boss-player10": 0.65,
        "boss-normal": 0.10,
    }


def test_checkpoint_resume_preserves_per_slot_draw_streams() -> None:
    source = AdaptiveEpisodeCurriculumSchedule(44, 3, boundaries(), config())
    first = boundaries()[0]
    for _ in range(5):
        source.record_outcome(first, "curriculum_complete")
    for slot in range(3):
        source.next(slot)
        source.next(slot)
    state = source.state_dict()
    resumed = AdaptiveEpisodeCurriculumSchedule(44, 3, boundaries(), config())
    resumed.load_state_dict(state)
    assert [source.next(slot).id for slot in range(3)] == [
        resumed.next(slot).id for slot in range(3)
    ]
    assert source.diagnostics() == resumed.diagnostics()


def test_demotes_when_active_boundary_is_confidently_too_hard() -> None:
    strict = AdaptiveCurriculumConfig(
        window_size=50,
        minimum_samples=40,
        promotion_lower_bound=0.60,
        demotion_upper_bound=0.10,
    )
    schedule = AdaptiveEpisodeCurriculumSchedule(7, 1, boundaries(), strict)
    first = boundaries()[0]
    for _ in range(40):
        schedule.record_outcome(first, "curriculum_complete")
    assert schedule.diagnostics()["active_index"] == 1
    second = boundaries()[1]
    for _ in range(40):
        schedule.record_outcome(second, "dead")
    diagnostics = schedule.diagnostics()
    assert diagnostics["active_index"] == 0
    assert diagnostics["boundaries"][0]["mastered"] is False


def test_mastery_replay_failure_can_restore_a_forgotten_boundary() -> None:
    strict = AdaptiveCurriculumConfig(
        window_size=50,
        minimum_samples=40,
        promotion_lower_bound=0.60,
        demotion_upper_bound=0.10,
    )
    schedule = AdaptiveEpisodeCurriculumSchedule(7, 1, boundaries(), strict)
    first, second = boundaries()[:2]
    for _ in range(40):
        schedule.record_outcome(first, "curriculum_complete")
    for _ in range(40):
        schedule.record_outcome(second, "curriculum_complete")
    assert schedule.diagnostics()["active_index"] == 2
    for _ in range(50):
        schedule.record_outcome(first, "dead")
    diagnostics = schedule.diagnostics()
    assert diagnostics["active_index"] == 0
    assert not any(boundary["mastered"] for boundary in diagnostics["boundaries"])


def test_load_adaptive_config_is_strict(tmp_path) -> None:
    path = tmp_path / "adaptive.json"
    path.write_text(
        '{"schema_version": 1, "config": {"minimum_samples": 50}}',
        encoding="utf-8",
    )
    assert load_adaptive_curriculum_config(path).minimum_samples == 50
    path.write_text(
        '{"schema_version": 1, "config": {"unknown": 1}}',
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="unknown adaptive curriculum parameters"):
        load_adaptive_curriculum_config(path)
