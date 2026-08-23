from __future__ import annotations

from autodancer.training.reward_compare import choose_arm, evaluate_arm


def summary(*, progress: float, deaths: float, step_limits: float) -> dict[str, float]:
    return {
        "episodes": 10,
        "mean_progress": progress,
        "death_rate": deaths,
        "step_limit_rate": step_limits,
        "enemy_kills": 20,
        "item_pickups": 20,
        "idle_rate": 0.2,
    }


def report(progress: float) -> dict[str, object]:
    episodes = [
        {
            "seed": index,
            "worker_id": "worker-0000",
            "run_id": str(index),
            "episode_return": 0.0,
            "turns": 10,
            "furthest_zone": 1,
            "furthest_floor": 2 if index < round((progress - 1) * 10) else 1,
            "max_gold": 0,
            "enemy_kills": 2,
            "item_pickups": 2,
            "item_value": 0,
            "enemy_damage": 2,
            "player_damage": 1,
            "status": "dead" if index < 4 else "step_limit",
            "idle_turns": 1,
            "action_counts": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        for index in range(10)
    ]
    trained = summary(progress=progress, deaths=0.4, step_limits=0.6)
    return {"trained": {**trained, "results": episodes}}


def test_arm_decision_requires_consistent_progress_and_guardrails() -> None:
    reference = summary(progress=1.0, deaths=0.4, step_limits=0.7)
    arm = evaluate_arm(reference, [report(1.2), report(1.1), report(1.0)], "v4a")
    assert arm["passed"] is True
    assert choose_arm(reference, [arm]) == "v4a"

    regressed = evaluate_arm(reference, [report(1.0), report(1.0), report(1.1)], "v4b")
    assert regressed["passed"] is False
    assert choose_arm(reference, [regressed]) is None
