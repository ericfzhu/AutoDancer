from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from autodancer.world_observation import (
    FACTS,
    FIELDS,
    decode_world_snapshot,
    project_world,
    world_contract,
    world_registry,
    world_spaces,
)


@pytest.fixture
def snapshot():
    def entity(identity, name, components, **extra):
        return dict(
            id=identity,
            name=name,
            x=1,
            y=0,
            displayed=True,
            components=components,
            markers=[],
            relations={},
            **extra,
        )

    return dict(
        version=1,
        current=True,
        coverage=list(FIELDS),
        tiles=[dict(x=1, y=0, name="Floor")],
        entities=[
            entity(
                100,
                "Slime",
                {
                    "health": {"health": 3, "maxHealth": 4},
                    "beatDelay": {"counter": 1, "interval": 2},
                },
            ),
            entity(101, "SpikeTrap", {}),
            entity(102, "WeaponDagger", {}),
        ],
    )


def test_overlapping_entities_keep_independent_state(snapshot):
    state = decode_world_snapshot(snapshot)
    obs = project_world(state, 4, "internal-v1")
    for key, space in world_spaces(4).items():
        assert space.contains(obs[key])
    assert len(set(obs["world_entities"][:3, 0])) == 3
    hp = FACTS.index("health.health")
    assert obs["world_facts"][0, hp].tolist() == [3, 1, 1]
    assert obs["world_facts"][1, hp].tolist() == [0, 0, 0]
    assert (obs["world_entities"][:3, 1:3] == [1, 0]).all()
    assert obs["world_tiles"][10, 11] == world_registry()[0]["tile_ids"]["Floor"]


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "identity", "tile", "coverage", "field", "bounds", "overflow", "currency"],
)
def test_world_rejects_unrepresentable_or_incomplete_state(snapshot, fault):
    row = snapshot["entities"][0]
    if fault == "duplicate":
        snapshot["entities"].append(deepcopy(row))
    elif fault == "identity":
        row["name"] = "UnqualifiedFutureEnemy"
    elif fault == "tile":
        snapshot["tiles"][0]["name"] = "UnqualifiedFloor"
    elif fault == "coverage":
        snapshot["coverage"].pop()
    elif fault == "field":
        row["components"]["health"].pop("health")
    elif fault == "bounds":
        row["x"] = 11
    elif fault == "overflow":
        row["components"]["health"]["health"] = 2**32
    else:
        row["currency"] = "unqualified"
    with pytest.raises(ValueError):
        decode_world_snapshot(snapshot)


def test_visibility_unknowns_capacity_and_terminal_state(snapshot):
    snapshot["entities"][1]["displayed"] = False
    state = decode_world_snapshot(snapshot)
    player = project_world(state, 2, "player-v1")
    assert (player["world_entities"][:, 0] != 0).sum() == 2
    assert player["world_facts"][0, FACTS.index("beatDelay.counter")].tolist() == [0, 0, 1]
    changed = deepcopy(snapshot)
    changed["entities"][0]["components"]["beatDelay"]["counter"] = 20
    changed["entities"][1]["name"] = "Minotaur"
    for key, value in player.items():
        np.testing.assert_array_equal(
            value, project_world(decode_world_snapshot(changed), 2, "player-v1")[key]
        )
    with pytest.raises(ValueError, match="capacity"):
        project_world(state, 2, "internal-v1")
    state["current"] = False
    assert all(not v.any() for v in project_world(state, 2, "internal-v1").values())


def test_entity_handles_are_only_relational_joins(snapshot):
    snapshot["entities"][0]["relations"] = {"grab": 102}
    first = project_world(decode_world_snapshot(snapshot), 4, "internal-v1")
    for row in snapshot["entities"]:
        row["id"] += 100000
    snapshot["entities"][0]["relations"]["grab"] += 100000
    second = project_world(decode_world_snapshot(snapshot), 4, "internal-v1")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["world_entities"][0, 4] == 3


@pytest.mark.parametrize("profile", ["internal-v1", "player-v1"])
def test_diagnostic_metadata_and_bridge_sequence_do_not_enter_a10_observations(snapshot, profile):
    from autodancer.constants import PlayerFeature
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.equipment_state import EQUIPMENT_COMPONENTS, SPECIAL_ACTIONS
    from tests.test_live_protocol import record

    controls = [dict(action=a.name, owner_id=0) for a in SPECIAL_ACTIONS]
    payload = record(0, "reset")
    payload["telemetry_timing"] = "settled-input-v1"
    payload["observation"]["world_snapshot"] = snapshot
    payload["observation"]["equipment_snapshot"] = dict(
        version=3,
        current=True,
        coverage=sorted(EQUIPMENT_COMPONENTS),
        items=[],
        owned=[],
        slots=[],
        containers=[],
        controls=controls,
        player_effects={},
        display=dict(items=[], controls=controls),
        world=[],
    )
    other = deepcopy(payload)
    other.update(
        seed=123456,
        run_id="diagnostic-run",
        sequence=999,
        events=[dict(kind="enemy_damage", amount=999)],
    )
    other["observation"]["player"][PlayerFeature.TURN] = 999
    projections = []
    for raw in (payload, other):
        env = AutoDancerLiveEnv(
            input_mode="engine-input-v1",
            equipment_profile=profile,
            world_capacity=4,
        )
        projections.append(env._accept_record(raw)[0])
    for name in projections[0]:
        np.testing.assert_array_equal(projections[0][name], projections[1][name])


def test_world_policy_roundtrip_sensitivity_and_permutation(snapshot):
    from autodancer.equipment_observation import equipment_environment_options
    from autodancer.training.model import ModelConfig, WorldActorCritic, model_from_spec
    from autodancer.training.performance import benchmark_observations

    torch.set_num_threads(1)
    model = WorldActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
            map_size=8,
        ),
        world_capacity=4,
        equipment_capacity=4,
        initialize=False,
    ).eval()
    synthetic = benchmark_observations(
        1, 1, torch.device("cpu"), equipment_capacity=4, world_capacity=4
    )
    obs = {k: v[:, 0] for k, v in synthetic.items()}
    snapshot["entities"][0]["relations"] = {"grab": 102}

    def tensors(raw):
        return {
            k: torch.from_numpy(v).unsqueeze(0)
            for k, v in project_world(decode_world_snapshot(raw), 4, "internal-v1").items()
        }

    obs.update(tensors(snapshot))
    permuted = deepcopy(snapshot)
    permuted["entities"].reverse()
    torch.testing.assert_close(model.encode_world(obs), model.encode_world(tensors(permuted)))
    state = model.initial_state(1)
    logits, value, _ = model.step(obs, state)
    feedback_changed = {**obs, "previous_reward": torch.tensor([12345.0])}
    for actual, expected in zip(
        model.step(feedback_changed, state), model.step(obs, state), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    task_changed = {**obs, "task_context": torch.tensor([[2, 5]], dtype=torch.int32)}
    task_logits, task_value, _ = model.step(task_changed, state)
    assert not torch.equal(task_logits, logits) and not torch.equal(task_value, value)
    changed = deepcopy(obs)
    changed["world_facts"][0, 0, FACTS.index("health.health"), 0] += 2
    logits2, value2, _ = model.step(changed, state)
    assert not torch.equal(logits, logits2) and not torch.equal(value, value2)
    (logits.sum() + value.sum()).backward()
    assert model.world_identity.weight.grad.abs().sum() > 0
    assert model.task_encoder[0].weight.grad.abs().sum() > 0
    restored = model_from_spec(model.architecture_spec(), initialize=False)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.step(obs, state)[0], logits)
    assert equipment_environment_options(model.architecture_spec())["world_capacity"] == 4
    spec = deepcopy(model.architecture_spec())
    spec["world"]["registry_digest"] = "changed"
    with pytest.raises(ValueError, match="World checkpoint"):
        model_from_spec(spec)
    for contract in ("information", "task"):
        spec = deepcopy(model.architecture_spec())
        spec[contract]["version"] += 1
        for restore in (model_from_spec, equipment_environment_options):
            with pytest.raises(ValueError, match="checkpoint contract"):
                restore(spec)
    from autodancer.training.representation import analyze_model

    report = analyze_model(model, batch_size=1)
    assert report["groups"]["world"]["sensitivity"] > 0
    assert report["groups"]["world"]["gradient_norm"] > 0
    assert report["groups"]["task"]["sensitivity"] > 0
    assert report["groups"]["task"]["gradient_norm"] > 0


def test_lua_field_bindings_match_pinned_world_schema():
    import re

    lua = (Path(__file__).parents[1] / "mods/AutoDancer/scripts/World.lua").read_text()
    fields = lua.split("local fields = {", 1)[1].split("\n}", 1)[0]
    captured = {
        component + "." + field
        for component, keys in re.findall(r"(\w+)=\{([^}]+)\}", fields)
        for field in re.findall(r'"(\w+)"', keys)
    }
    assert captured == set(FIELDS)
    assert world_contract(4, "player-v1")["registry_digest"] == world_registry()[1]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("target", [None, 5])
def test_live_adapter_preserves_truncation_world_and_clears_terminal_world(
    snapshot, terminal, target
):
    from autodancer.constants import Action, PlayerFeature
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.equipment_state import EQUIPMENT_COMPONENTS, SPECIAL_ACTIONS
    from autodancer.live.protocol import QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    controls = [dict(action=a.name, owner_id=0) for a in SPECIAL_ACTIONS]
    equipment = dict(
        version=3,
        current=True,
        coverage=sorted(EQUIPMENT_COMPONENTS),
        items=[],
        owned=[],
        slots=[],
        containers=[],
        controls=controls,
        player_effects={},
        display=dict(items=[], controls=controls),
        world=[],
    )
    rows = [
        record(0, "reset"),
        record(
            1,
            "terminal" if terminal else "turn",
            status="dead" if terminal else "running",
            requested_action=Action.WAIT,
        ),
    ]
    for row in rows:
        row["telemetry_timing"] = "terminal" if row["kind"] == "terminal" else "settled-input-v1"
        row["observation"]["equipment_snapshot"] = deepcopy(equipment)
        row["observation"]["world_snapshot"] = deepcopy(snapshot)
        if row["kind"] == "terminal":
            row["observation"]["equipment_snapshot"]["current"] = False
            row["observation"]["world_snapshot"]["current"] = False
    env = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(rows),
        bridge=FakeBridge(),
        input_mode="engine-input-v1",
        equipment_profile="player-v1",
        world_capacity=4,
        max_turns=1,
        curriculum_target_level=target,
    )
    try:
        obs, _ = env.reset(seed=7)
        assert obs["task_context"].tolist() == ([1, 0] if target is None else [2, target])
        assert obs["player"][PlayerFeature.TURN] == 0
        assert obs["world_current"].item() == 1
        assert not obs["grid"][..., 1:9].any() and not obs["grid"][..., 10:].any()
        obs, _, terminated, truncated, info = env.step(Action.WAIT)
        assert obs["player"][PlayerFeature.TURN] == 1
        assert info["information_contract"]["human_equivalent"] is False
        assert info["task_contract"]["finite_horizon"] is False
        assert terminated == terminal and truncated != terminal
        assert bool(obs["world_current"].item()) != terminal
        assert info["world_observation_contract"]["capacity"] == 4
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_outcomes_and_boss_health_use_separate_world_entities(snapshot):
    from autodancer.constants import Action
    from autodancer.live.protocol import decode_observation
    from autodancer.outcomes import classify_action_outcome
    from autodancer.rewards import RewardTracker
    from tests.test_live_protocol import record

    snapshot["entities"][0]["markers"] = ["enemy"]
    snapshot["entities"][0]["components"]["boss"] = dict(type=1, defeated=False)
    base = decode_observation(record(0, "reset")["observation"])
    obs = {**base, **project_world(decode_world_snapshot(snapshot), 4, "internal-v1")}
    assert RewardTracker._visible_boss_health(obs) == 3
    outcome = classify_action_outcome(obs, obs, Action.RIGHT, {})
    assert outcome.category == "combat_attempt"


def test_evaluation_does_not_retry_or_replace_an_invalid_episode():
    from autodancer.envs.vector import VectorInfrastructureError
    from autodancer.live.protocol import ProtocolError
    from autodancer.training.baseline import _evaluate_model_async
    from autodancer.training.model import ModelConfig, RecurrentActorCritic
    from tests.test_async_collector import AsyncEnvironment

    environment = AsyncEnvironment()
    worker = environment.environments["worker-0001"]
    original = worker.step
    calls = 0

    def fail_second(action):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProtocolError("Unqualified world state")
        return original(action)

    worker.step = fail_second

    def no_recovery(*args, **kwargs):
        pytest.fail("Evaluation must not retry an invalid episode")

    environment.recover = no_recovery
    model = RecurrentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
        )
    )
    with pytest.raises(VectorInfrastructureError):
        _evaluate_model_async(
            environment,
            model,
            seeds=[1, 2],
            max_steps=3,
            policy_seed=1,
            device=torch.device("cpu"),
            dashboard_state=None,
            action_contract="current",
            deterministic=True,
        )
    assert calls == 2 and len(worker.reset_calls) == 1
    assert environment.infrastructure_events[0]["context"]["episode_valid"] is False
    assert environment.infrastructure_events[0]["context"]["partial_turns"] == 1


def test_throughput_benchmark_does_not_discard_failed_steps():
    from autodancer.envs.vector import AutoDancerVectorEnv, VectorInfrastructureError
    from autodancer.training.benchmark import _run_slot
    from tests.test_vector_env import FakeSupervisor

    supervisor = FakeSupervisor()
    environment = AutoDancerVectorEnv(supervisor)
    try:
        with pytest.raises(VectorInfrastructureError):
            _run_slot(environment, 1, 3)
        assert supervisor.replacements == []
        assert environment.infrastructure_events[0]["benchmark_valid"] is False
        with pytest.raises(RuntimeError, match="invalid"):
            environment.reset_at([1], [1])
    finally:
        environment.close()
