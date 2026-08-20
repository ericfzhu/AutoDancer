from autodancer.evaluation import EpisodeResult, run_episode, summarize


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


class _LiveLikeEnvironment:
    def reset(self, *, seed: int):
        del seed
        return {}, {"seed": 847291, "episode_status": "running"}

    def step(self, action: int):
        del action
        return (
            {},
            0.0,
            True,
            False,
            {
                "seed": 847291,
                "episode_status": "won",
                "zone": 4,
                "floor": 4,
                "turns": 1,
                "completed": 1,
            },
        )


def test_live_like_episode_uses_observed_game_seed() -> None:
    result = run_episode(_LiveLikeEnvironment(), lambda observation: 0, seed=0)
    assert result.seed == 847291
