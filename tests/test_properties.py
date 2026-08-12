from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from autodancer.constants import Action
from autodancer.envs.sim import AutoDancerSimEnv


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=100_000),
    actions=st.lists(st.integers(0, 10), max_size=50),
)
def test_core_state_invariants(seed: int, actions: list[int]) -> None:
    environment = AutoDancerSimEnv(task="floor")
    environment.reset(seed=seed)
    for action in actions:
        state = environment.state
        assert state is not None
        positions = [state.player.position, *(enemy.position for enemy in state.enemies.values())]
        assert len(positions) == len(set(positions))
        assert all(state.in_bounds(x, y) for x, y in positions)
        assert 0 <= state.player.health <= state.player.max_health
        assert state.bombs >= 0
        _, _, terminated, truncated, _ = environment.step(Action(action))
        if terminated or truncated:
            break
