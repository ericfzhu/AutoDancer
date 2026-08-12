from autodancer.evaluation import EpisodeResult, summarize


def test_summary_uses_completion_metrics_not_shaped_reward() -> None:
    report = summarize(
        [
            EpisodeResult(1, True, 4, 4, 0, 100, 500),
            EpisodeResult(2, False, 2, 3, 1, 80, 100),
        ],
        source="simulator",
    )
    assert report["completion_rate"] == 0.5
    assert report["furthest_floor"] == 16
    assert report["deaths"] == 1
    assert "reward" not in report

