import pytest

from autodancer.equipment_state import map_equipment_state
from autodancer.harness_contract import InformationPolicy, project_facts


@pytest.fixture
def v3_snapshot(v2_snapshot):
    from autodancer.equipment_state import EQUIPMENT_COMPONENTS, SPECIAL_ACTIONS

    v2_snapshot.update(
        version=3,
        coverage=sorted(EQUIPMENT_COMPONENTS),
        player_effects={
            "barrier": {"remainingTurns": 7, "permanent": False},
            "regenerationKillCounter": {"killCount": 8, "killThreshold": 10, "totalIncrement": 1},
        },
        display={
            "items": [{"id": 3, "fields": ["activation_active"]}],
            "controls": [
                {"action": a.name, "owner_id": 3 if a.name == "THROW" else 0}
                for a in SPECIAL_ACTIONS
            ],
        },
    )
    return v2_snapshot


def test_player_projection_does_not_leak_hidden_equipment_state(v3_snapshot):
    from copy import deepcopy

    import numpy as np

    from autodancer.equipment_observation import (
        EQUIPMENT_FACTS,
        equipment_contract,
        project_equipment,
    )
    from autodancer.equipment_state import decode_equipment_snapshot

    state = decode_equipment_snapshot(v3_snapshot)
    player = project_equipment(state, 4, "player-v1")
    internal = project_equipment(state, 4)
    ammo = EQUIPMENT_FACTS.index("ammo")
    assert player["equipment_facts"][0, ammo].tolist() == [0, 0, 1]
    fields = equipment_contract(4)["player_effects"]
    assert player["equipment_effects"][fields.index("barrier_active")].tolist() == [1, 1, 1]
    assert player["equipment_effects"][fields.index("barrier_turns")].tolist() == [0, 0, 1]
    modified = deepcopy(v3_snapshot)
    modified["player_effects"]["barrier"]["remainingTurns"] = 99
    modified["player_effects"]["regenerationKillCounter"]["killCount"] = 9
    modified["items"][0]["bans"]["activate"] = True
    newer = decode_equipment_snapshot(modified)
    for key, value in player.items():
        np.testing.assert_array_equal(value, project_equipment(newer, 4, "player-v1")[key])
    assert not np.array_equal(
        internal["equipment_effects"], project_equipment(newer, 4)["equipment_effects"]
    )


def test_display_evidence_is_required_and_validated(v3_snapshot):
    from autodancer.equipment_observation import project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot

    v3_snapshot["display"]["items"][0]["fields"].append("holster_sticky")
    with pytest.raises(ValueError, match="displayed equipment field"):
        decode_equipment_snapshot(v3_snapshot)
    v3_snapshot.pop("display")
    with pytest.raises(ValueError, match="display evidence"):
        project_equipment(decode_equipment_snapshot(v3_snapshot), 4, "player-v1")


def test_storage_preserves_order_and_nested_contents(v3_snapshot):
    from copy import deepcopy

    from autodancer.equipment_observation import carried_item_counts, project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot

    first = v3_snapshot["items"][0]
    v3_snapshot["items"] += [
        dict(id=4, name="ItemStash", components={"storage": []}, bans=first["bans"]),
        dict(id=5, name="Food1", components={}, bans=first["bans"]),
    ]
    v3_snapshot["slots"] = [{"name": "stash", "index": 1, "item_id": 4}]
    v3_snapshot["owned"] = [4]
    v3_snapshot["containers"] = [
        dict(item_id=4, content_id=3, kind="storage", index=1),
        dict(item_id=4, content_id=5, kind="storage", index=2),
    ]
    state = decode_equipment_snapshot(v3_snapshot)
    tensors = project_equipment(state, 4)
    assert tensors["equipment_items"][:3, 2].tolist() == [2, 0, 2]
    assert tensors["equipment_items"][:3, 4].tolist() == [1, 0, 2]
    assert len(carried_item_counts(tensors)) == 3
    for change in ("duplicate", "gap", "empty"):
        broken = deepcopy(v3_snapshot)
        second = broken["containers"][1]
        if change == "duplicate":
            second["content_id"] = 3
        if change == "gap":
            second["index"] = 3
        if change == "empty":
            second["content_id"] = 0
        with pytest.raises(ValueError):
            decode_equipment_snapshot(broken)


def test_old_snapshots_do_not_invent_new_mutable_fields(v2_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    state = decode_equipment_snapshot(v2_snapshot)
    assert not state["items"][0]["facts"]["combo_reset"]["known"]
    assert not state["player_effects"]["barrier_active"]["known"]


def test_engine_damage_remains_damage_if_healed_during_the_same_action():
    import numpy as np

    from autodancer.envs.live import AutoDancerLiveEnv

    before, after = {"player": np.array([6])}, {"player": np.array([6])}
    events = [{"kind": "player_damage", "amount": 2, "data": {"source": "engine_health_loss"}}]
    assert AutoDancerLiveEnv._normalize_player_damage(events, before, after) == events


def test_visible_ground_items_have_exact_identities_and_bounded_capacity(v3_snapshot):
    from autodancer.equipment_observation import project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot, equipment_registry

    v3_snapshot["world"] = [
        dict(kind="item", name=n, x=-3, y=4) for n in ("Piston", "WeaponGoldenLongsword")
    ]
    state = decode_equipment_snapshot(v3_snapshot)
    obs = project_equipment(state, 4, "player-v1")
    ids = obs["equipment_world"][:2, 1].tolist()
    assert ids == [equipment_registry()[0][n] for n in ("Piston", "WeaponGoldenLongsword")]
    assert ids[0] != ids[1]  # These collide in the legacy type hash.
    assert obs["equipment_world"][:2, 2:].tolist() == [[-3, 4], [-3, 4]]
    assert obs["equipment_world_current"].tolist() == [1]
    with pytest.raises(ValueError, match="Visible equipment capacity"):
        project_equipment(state, 1)
    state["world"] = None
    assert project_equipment(state, 4)["equipment_world_current"].tolist() == [0]
    v3_snapshot["world"][0]["x"] = 11
    with pytest.raises(ValueError, match="outside observation grid"):
        decode_equipment_snapshot(v3_snapshot)


def test_player_environment_removes_legacy_equipment_leaks(v3_snapshot):
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    row = record(0, "reset")
    row["telemetry_timing"] = "settled-input-v1"
    row["observation"]["equipment_snapshot"] = v3_snapshot
    row["observation"]["inventory"][0] = [1, 1, 1, 1, 1, 1, 1, 1]
    row["observation"]["action_mask"][8] = 1
    env = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([row]),
        bridge=FakeBridge(),
        input_mode="engine-input-v1",
        equipment_profile="player-v1",
        equipment_capacity=4,
    )
    obs, info = env.reset(seed=7)
    assert not obs["inventory"].any()
    assert not obs["equipment_controls"].any()
    assert obs["action_mask"].tolist() == [1] * 11
    assert info["equipment_observation_contract"]["profile"] == "player-v1"


def test_player_checkpoint_preserves_information_profile():
    from autodancer.equipment_observation import equipment_environment_options
    from autodancer.training.model import EquipmentActorCritic, ModelConfig, model_from_spec

    model = EquipmentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
            map_size=8,
        ),
        equipment_capacity=4,
        equipment_profile="player-v1",
        initialize=False,
    )
    spec = model.architecture_spec()
    assert equipment_environment_options(spec)["equipment_profile"] == "player-v1"
    assert model_from_spec(spec, initialize=False).architecture_spec() == spec
    spec["equipment"]["profile"] = "unknown-profile"
    with pytest.raises(ValueError):
        model_from_spec(spec, initialize=False)


def test_engine_pickup_contract_prevents_inventory_grant_inference():
    from autodancer.constants import Action
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    rows = [record(0, "reset"), record(1, "turn", requested_action=Action.WAIT)]
    for row in rows:
        row["equipment_events"] = "engine-events-v1"
    rows[1]["observation"]["inventory"][0] = [1, 42, 1, 0, 0, 0, 1, 0]
    rows[1]["observation"]["action_mask"][8] = 1
    env = AutoDancerLiveEnv(turn_source=QueueTurnSource(rows), bridge=FakeBridge())
    env.reset(seed=7)
    _, _, _, _, info = env.step(Action.WAIT)
    assert not any(e["kind"] == "item_collected" for e in info["raw_events"])


def test_stateful_special_actions_are_not_reported_as_no_effect(v3_snapshot):
    from copy import deepcopy

    from autodancer.constants import Action
    from autodancer.equipment_observation import project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot
    from autodancer.live.protocol import decode_observation
    from autodancer.outcomes import classify_action_outcome
    from tests.test_live_protocol import record

    before = decode_observation(record(0, "reset")["observation"])
    before.update(project_equipment(decode_equipment_snapshot(v3_snapshot), 4))
    after = deepcopy(before)
    after["equipment_effects"][0, 0] += 1
    outcome = classify_action_outcome(before, after, Action.SPELL_1, {})
    assert outcome.category == "special_state_changed"
    assert not outcome.productive  # An observed change is not evidence of strategic benefit.
    assert (
        classify_action_outcome(before, before, Action.SPELL_1, {}).category
        == "special_no_observed_change"
    )


def test_arming_and_toggling_remain_distinct():
    facts = map_equipment_state(
        {
            "itemActivable": {"active": True},
            "itemToggleable": {"active": False},
            "weaponThrowable": [],
        },
        complete=True,
    )
    assert facts["activation_active"].value is True
    assert facts["toggle_active"].value is False
    assert facts["capability.throwable"].value is True
    assert not facts["ammo"].applicable


def test_ammunition_zero_and_missing_are_not_conflated():
    facts = map_equipment_state({"weaponReloadable": {"ammo": 0, "maximumAmmo": 3}})
    assert facts["ammo"].known and facts["ammo"].value == 0
    assert facts["ammo_capacity"].value == 3
    assert facts["ammo_per_reload"].applicable and not facts["ammo_per_reload"].known
    assert not facts["capability.bag"].known
    assert not map_equipment_state({}, complete=True)["ammo"].applicable
    assert map_equipment_state({})["ammo"].applicable


def test_partial_probe_empty_table_does_not_invent_values():
    facts = map_equipment_state({"itemActivable": []})
    assert facts["capability.activable"].value is True
    assert not facts["activation_active"].known


def test_mapping_does_not_grant_player_visibility_or_equipment_support():
    facts = map_equipment_state({"itemStack": {"quantity": 2}})
    with pytest.raises(ValueError, match="outside player"):
        project_facts(facts, InformationPolicy.PLAYER)
    assert project_facts(facts, InformationPolicy.PRIVILEGED) == facts


@pytest.mark.parametrize(
    "components",
    [
        {"weaponReloadable": {"ammo": True}},
        {"itemActivable": {"active": 1}},
        {"itemStack": {"quantity": None}},
        {"weaponReloadable": [1]},
        {"weaponThrowable": None},
    ],
)
def test_malformed_values_fail_instead_of_becoming_plausible_state(components):
    with pytest.raises(ValueError):
        map_equipment_state(components)


@pytest.fixture
def runtime_snapshot():
    from autodancer.equipment_state import SPECIAL_ACTIONS, V1_COMPONENTS

    return {
        "version": 1,
        "current": True,
        "coverage": sorted(V1_COMPONENTS),
        "items": [
            {
                "id": 3,
                "name": "WeaponDagger",
                "components": {
                    "weaponThrowable": [],
                    "itemActivable": {"active": False},
                    "weapon": {"damage": 1},
                },
                "bans": {"pickup": False, "equip": False, "activate": False},
            }
        ],
        "slots": [{"name": "weapon", "index": 1, "item_id": 3}],
        "controls": [
            {"action": a.name, "owner_id": 3 if a.name == "THROW" else 0} for a in SPECIAL_ACTIONS
        ],
    }


def test_live_selection_does_not_claim_availability(runtime_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    state = decode_equipment_snapshot(runtime_snapshot)
    throw = next(c for c in state["controls"] if c["action"] == "THROW")
    assert throw["owner"] == "WeaponDagger" and throw["selection_known"]
    assert all(c["availability"] == "unknown" for c in state["controls"])
    facts = state["items"][0]["facts"]
    assert facts["activation_active"]["known"] and facts["activation_active"]["value"] is False
    assert not facts["ammo"]["applicable"]
    assert facts["ban.equip"]["source"] == "internal"
    assert decode_equipment_snapshot(None) is None


def test_owner_can_be_outside_inventory_and_terminal_state_is_not_current(runtime_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    runtime_snapshot["slots"] = []  # virtual/action owners need not occupy an inventory row
    runtime_snapshot["current"] = False
    state = decode_equipment_snapshot(runtime_snapshot)
    assert state["items"] and not state["slots"]
    assert not any(c["selection_known"] for c in state["controls"])


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "coverage",
        "duplicate_item",
        "missing_slot",
        "duplicate_slot",
        "missing_owner",
        "missing_action",
        "duplicate_action",
        "bad_ban",
        "bad_field",
    ],
)
def test_live_snapshot_rejects_uncertified_or_inconsistent_state(runtime_snapshot, fault):
    from autodancer.equipment_state import decode_equipment_snapshot

    if fault == "version":
        runtime_snapshot["version"] = True
    elif fault == "coverage":
        runtime_snapshot["coverage"].pop()
    elif fault == "duplicate_item":
        runtime_snapshot["items"].append(runtime_snapshot["items"][0])
    elif fault == "missing_slot":
        runtime_snapshot["slots"][0]["item_id"] = 999
    elif fault == "duplicate_slot":
        runtime_snapshot["slots"].append(runtime_snapshot["slots"][0])
    elif fault == "missing_owner":
        runtime_snapshot["controls"][0]["owner_id"] = 999
    elif fault == "missing_action":
        runtime_snapshot["controls"].pop()
    elif fault == "duplicate_action":
        runtime_snapshot["controls"].append(runtime_snapshot["controls"][0])
    elif fault == "bad_ban":
        runtime_snapshot["items"][0]["bans"]["activate"] = 0
    else:
        runtime_snapshot["items"][0]["components"]["itemActivable"]["active"] = 0
    with pytest.raises(ValueError):
        decode_equipment_snapshot(runtime_snapshot)


def test_live_adapter_keeps_equipment_diagnostics_out_of_actor_input(runtime_snapshot):
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import ProtocolError, QueueTurnSource, validate_record
    from tests.test_live_protocol import FakeBridge, record

    payload = record(0, "reset")
    payload["observation"]["equipment_snapshot"] = runtime_snapshot
    env = AutoDancerLiveEnv(turn_source=QueueTurnSource([payload]), bridge=FakeBridge())
    obs, info = env.reset(seed=7)
    assert "equipment_snapshot" not in obs and "equipment_state" not in obs
    assert info["equipment_state"]["items"][0]["name"] == "WeaponDagger"
    assert not obs["action_mask"][8]  # owner alone must not silently change legacy masks
    runtime_snapshot["current"] = False
    with pytest.raises(ProtocolError, match="freshness"):
        validate_record(payload)


def test_terminal_equipment_state_is_explicitly_stale(runtime_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot
    from autodancer.live.protocol import ProtocolError, validate_record
    from tests.test_live_protocol import record

    payload = record(1, "terminal", status="dead")
    payload["telemetry_timing"] = "terminal"
    payload["observation"]["equipment_snapshot"] = runtime_snapshot
    with pytest.raises(ProtocolError, match="freshness"):
        validate_record(payload)
    runtime_snapshot["current"] = False
    validate_record(payload)
    assert not any(
        c["selection_known"] for c in decode_equipment_snapshot(runtime_snapshot)["controls"]
    )


def test_malformed_owner_action_fails_as_protocol_error(runtime_snapshot):
    from autodancer.live.protocol import ProtocolError, validate_record
    from tests.test_live_protocol import record

    payload = record(0, "reset")
    payload["observation"]["equipment_snapshot"] = runtime_snapshot
    runtime_snapshot["controls"][0]["action"] = []
    with pytest.raises(ProtocolError, match="Unknown equipment action"):
        validate_record(payload)


def test_settled_capture_timing_is_explicit_and_old_records_remain_distinguishable():
    from autodancer.live.protocol import ProtocolError, validate_record
    from tests.test_live_protocol import record

    payload = record(0, "reset")
    validate_record(payload)
    payload["telemetry_timing"] = "settled-input-v1"
    validate_record(payload)
    payload["telemetry_timing"] = "terminal"
    with pytest.raises(ProtocolError, match="capture timing"):
        validate_record(payload)


@pytest.fixture
def v2_snapshot(runtime_snapshot):
    from autodancer.equipment_state import V2_COMPONENTS

    runtime_snapshot.update(version=2, coverage=sorted(V2_COMPONENTS), owned=[3], containers=[])
    return runtime_snapshot


def test_catalog_ids_are_collision_free_and_bound_to_snapshot(v2_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot, equipment_registry

    registry, digest = equipment_registry()
    assert len(registry) == len(set(registry.values())) == 250
    assert 0 not in registry.values()
    state = decode_equipment_snapshot(v2_snapshot)
    assert state["registry_digest"] == digest
    assert state["items"][0]["catalog_id"] == registry["WeaponDagger"]
    v2_snapshot["items"][0]["name"] = "NewUnregisteredWeapon"
    with pytest.raises(ValueError, match="outside the pinned registry"):
        decode_equipment_snapshot(v2_snapshot)


def test_holstered_weapon_retains_its_own_state(v2_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    v2_snapshot["items"].append(
        {
            "id": 4,
            "name": "Holster",
            "components": {"itemHolster": []},
            "bans": {"pickup": False, "equip": False, "activate": False},
        }
    )
    v2_snapshot["slots"] = [{"name": "hud", "index": 1, "item_id": 4}]
    v2_snapshot["owned"] = [4]
    v2_snapshot["containers"] = [{"item_id": 4, "content_id": 3}]
    state = decode_equipment_snapshot(v2_snapshot)
    assert state["containers"] == [{"item_id": 4, "content_id": 3}]
    assert state["items"][0]["facts"]["activation_active"]["value"] is False
    v2_snapshot["slots"].append({"name": "weapon", "index": 1, "item_id": 3})
    with pytest.raises(ValueError, match="Stored item also"):
        decode_equipment_snapshot(v2_snapshot)


def test_containment_cycles_are_rejected(v2_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    bans = dict(pickup=False, equip=False, activate=False)
    v2_snapshot["items"] = [
        dict(id=i, name="Holster", components={"itemHolster": []}, bans=bans) for i in (3, 4)
    ]
    v2_snapshot["slots"] = []
    v2_snapshot["containers"] = [{"item_id": 3, "content_id": 4}, {"item_id": 4, "content_id": 3}]
    with pytest.raises(ValueError, match="Containment cycle"):
        decode_equipment_snapshot(v2_snapshot)


def test_cooldown_and_engine_spell_usability_are_independent(v2_snapshot):
    from autodancer.equipment_state import decode_equipment_snapshot

    item = v2_snapshot["items"][0]
    item.update(
        name="SpellFireball",
        components={
            "itemCastOnUse": [],
            "spellCooldownKills": {"remainingKills": 10},
            "spellBloodMagic": {"damage": 2},
        },
        spell_usable=True,
    )
    state = decode_equipment_snapshot(v2_snapshot)
    facts = state["items"][0]["facts"]
    assert facts["cooldown_kills"]["value"] == 10
    assert facts["spell_usable"]["value"] is True
    assert facts["blood_cast_damage"]["value"] == 2
    assert all(c["availability"] == "unknown" for c in state["controls"])
    item.pop("spell_usable")
    with pytest.raises(ValueError, match="Missing engine spell usability"):
        decode_equipment_snapshot(v2_snapshot)


def test_general_input_mode_does_not_block_empty_or_cooldown_attempts(v2_snapshot):
    from autodancer.constants import Action
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    rows = [record(0, "reset"), record(1, "turn", requested_action=Action.SPELL_1)]
    for row in rows:
        row["telemetry_timing"] = "settled-input-v1"
        row["observation"]["equipment_snapshot"] = v2_snapshot
    env = AutoDancerLiveEnv(
        turn_source=QueueTurnSource(rows), bridge=FakeBridge(), input_mode="engine-input-v1"
    )
    obs, info = env.reset(seed=7)
    assert obs["action_mask"].tolist() == [1] * 11
    assert info["input_mode"] == "engine-input-v1"
    env.step(Action.SPELL_1)


def test_general_input_mode_requires_new_state_and_settled_capture(runtime_snapshot):
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import ProtocolError, QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    row = record(0, "reset")
    row["observation"]["equipment_snapshot"] = runtime_snapshot
    env = AutoDancerLiveEnv(
        turn_source=QueueTurnSource([row]), bridge=FakeBridge(), input_mode="engine-input-v1"
    )
    with pytest.raises(ProtocolError, match="snapshot v2"):
        env.reset(seed=7)


def test_large_item_stack_quantity_is_not_truncated():
    import numpy as np

    from autodancer.live.protocol import decode_observation
    from tests.test_live_protocol import record

    raw = record(0, "reset")["observation"]
    raw["inventory"][6][2] = 1000000000
    obs = decode_observation(raw)
    assert obs["inventory"].dtype == np.int32
    assert obs["inventory"][6, 2] == 1000000000


def test_equipment_projection_preserves_unknown_and_inapplicable(v2_snapshot):
    from autodancer.equipment_observation import EQUIPMENT_FACTS, project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot

    v2_snapshot["items"][0]["components"]["itemStack"] = {"quantity": 1000000000}
    obs = project_equipment(decode_equipment_snapshot(v2_snapshot), 4)
    facts = obs["equipment_facts"][0]
    assert facts[EQUIPMENT_FACTS.index("quantity")].tolist() == [1000000000, 1, 1]
    assert facts[EQUIPMENT_FACTS.index("activation_active")].tolist() == [0, 1, 1]
    assert facts[EQUIPMENT_FACTS.index("ammo")].tolist() == [0, 0, 0]
    del v2_snapshot["items"][0]["components"]["itemStack"]["quantity"]
    obs = project_equipment(decode_equipment_snapshot(v2_snapshot), 4)
    assert obs["equipment_facts"][0, EQUIPMENT_FACTS.index("quantity")].tolist() == [0, 0, 1]


def test_equipment_projection_never_truncates_or_exposes_stale_state(v2_snapshot):
    import copy

    from autodancer.equipment_observation import project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot

    item = copy.deepcopy(v2_snapshot["items"][0])
    item["id"] = 99
    v2_snapshot["items"].append(item)
    state = decode_equipment_snapshot(v2_snapshot)
    with pytest.raises(ValueError, match="capacity 1 exceeded"):
        project_equipment(state, 1)
    state["current"] = False
    assert all(not array.any() for array in project_equipment(state, 1).values())
    state["current"] = True
    state["items"][0]["facts"]["quantity"].update(value=2**32, known=True, applicable=True)
    with pytest.raises(ValueError, match="losslessly"):
        project_equipment(state, 4)


def test_equipment_projection_is_opt_in_and_in_live_space(v2_snapshot):
    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.live.protocol import QueueTurnSource
    from tests.test_live_protocol import FakeBridge, record

    row = record(0, "reset")
    row["telemetry_timing"] = "settled-input-v1"
    row["observation"]["equipment_snapshot"] = v2_snapshot
    for profile in ("legacy", "internal-v1"):
        env = AutoDancerLiveEnv(
            turn_source=QueueTurnSource([row]),
            bridge=FakeBridge(),
            input_mode="engine-input-v1",
            equipment_profile=profile,
        )
        obs, info = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert ("equipment_items" in obs) == (profile == "internal-v1")
        assert bool(info["equipment_observation_contract"]) == (profile == "internal-v1")
    with pytest.raises(ValueError, match="require engine-input-v1"):
        AutoDancerLiveEnv(equipment_profile="internal-v1")


def test_equipment_policy_relations_permutation_gradients_and_checkpoint(v2_snapshot):
    import copy

    import numpy as np
    import torch

    from autodancer.equipment_observation import project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot
    from autodancer.training.model import EquipmentActorCritic, ModelConfig, model_from_spec
    from tests.test_training import observations

    torch.set_num_threads(1)
    torch.manual_seed(82)
    v2_snapshot["items"].append(
        dict(
            id=4,
            name="Holster",
            components={"itemHolster": []},
            bans=dict(pickup=False, equip=False, activate=False),
        )
    )
    v2_snapshot.update(
        slots=[dict(name="hud", index=1, item_id=4)],
        owned=[4],
        containers=[dict(item_id=4, content_id=3)],
    )
    state = decode_equipment_snapshot(v2_snapshot)
    projected = project_equipment(state, 4)
    assert projected["equipment_items"][0, 2] == 2
    # Relabel engine IDs without changing any observable equipment semantics.
    renamed = copy.deepcopy(v2_snapshot)
    for item in renamed["items"]:
        item["id"] += 10000
    renamed["owned"] = [10004]
    renamed["slots"][0]["item_id"] = 10004
    renamed["containers"] = [dict(item_id=10004, content_id=10003)]
    for control in renamed["controls"]:
        if control["owner_id"]:
            control["owner_id"] += 10000
    other = project_equipment(decode_equipment_snapshot(renamed), 4)
    assert all(np.array_equal(projected[k], other[k]) for k in projected)
    model = EquipmentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
            map_size=8,
        ),
        equipment_capacity=4,
        initialize=False,
    ).eval()
    obs = {key: value[0] for key, value in observations(1, 1).items()}
    obs.update({k: torch.from_numpy(v).unsqueeze(0) for k, v in projected.items()})
    permuted = copy.deepcopy(state)
    permuted["items"].reverse()
    permuted_obs = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in project_equipment(permuted, 4).items()
    }
    torch.testing.assert_close(model.encode_equipment(obs), model.encode_equipment(permuted_obs))
    # A stored weapon must affect both policy and value, and receive gradients.
    initial = model.initial_state(1)
    logits, values, _ = model.step(obs, initial)
    changed = {k: v.clone() for k, v in obs.items()}
    changed["equipment_facts"][0, 0, :, 0] += 1
    logits2, values2, _ = model.step(changed, initial)
    assert not torch.equal(logits, logits2)
    assert not torch.equal(values, values2)
    (logits.sum() + values.sum()).backward()
    assert model.equipment_identity.weight.grad.abs().sum() > 0
    restored = model_from_spec(model.architecture_spec(), initialize=False).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.step(obs, initial)[0], logits)
    spec = copy.deepcopy(model.architecture_spec())
    spec["equipment"]["registry_digest"] = "different"
    with pytest.raises(ValueError, match="registry does not match"):
        model_from_spec(spec, initialize=False)
    with pytest.raises(ValueError, match="capacity differs"):
        model.encode_equipment({**obs, "equipment_items": obs["equipment_items"][:, :3]})
    from autodancer.training.representation import analyze_model

    analysis = analyze_model(model, batch_size=1)
    assert analysis["groups"]["equipment"]["sensitivity"] > 0
    assert analysis["groups"]["equipment"]["gradient_norm"] > 0


def test_async_collection_preserves_full_equipment(v2_snapshot, monkeypatch):
    import torch

    import tests.test_async_collector as fixtures
    from autodancer.equipment_observation import equipment_environment_options, project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot
    from autodancer.training.async_collector import VersionedAsyncRolloutCollector
    from autodancer.training.model import EquipmentActorCritic, ModelConfig

    tensors = project_equipment(decode_equipment_snapshot(v2_snapshot), 4)
    old_observation = fixtures.observation

    def observation(slot):
        return {**old_observation(slot), **{key: value.copy() for key, value in tensors.items()}}

    monkeypatch.setattr(fixtures, "observation", observation)
    model = EquipmentActorCritic(
        ModelConfig(
            cell_size=16,
            spatial_size=16,
            hidden_size=16,
            entity_limit=4,
            attention_layers=1,
            attention_heads=2,
            map_size=8,
        ),
        equipment_capacity=4,
        initialize=False,
    )
    options = equipment_environment_options(model.architecture_spec())
    assert options == dict(
        input_mode="engine-input-v1", equipment_profile="internal-v1", equipment_capacity=4
    )
    assert equipment_environment_options({"version": 6}) == {}
    collector = VersionedAsyncRolloutCollector(
        fixtures.AsyncEnvironment(),
        model,
        device=torch.device("cpu"),
        seed=82,
    )
    try:
        rollout = collector.collect(1)
    finally:
        collector.close()
    for key, value in tensors.items():
        actual = rollout.observations[key]
        expected = torch.from_numpy(value)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected.expand_as(actual))


def test_holster_swaps_do_not_invent_pickups_or_new_reward_types(v2_snapshot):
    import copy

    import numpy as np

    from autodancer.envs.live import AutoDancerLiveEnv
    from autodancer.equipment_observation import carried_item_counts, project_equipment
    from autodancer.equipment_state import decode_equipment_snapshot, equipment_registry
    from autodancer.rewards import RewardTracker

    bans = dict(pickup=False, equip=False, activate=False)
    v2_snapshot["items"].extend(
        [
            dict(id=4, name="Holster", components={"itemHolster": []}, bans=bans),
            dict(id=5, name="WeaponCrossbow", components={}, bans=bans),
            # A selected virtual owner is not part of the carried inventory.
            dict(id=6, name="SpellHeal", components={}, bans=bans),
        ]
    )
    v2_snapshot.update(
        owned=[4, 5],
        containers=[dict(item_id=4, content_id=3)],
        slots=[dict(name="hud", index=1, item_id=4), dict(name="weapon", index=1, item_id=5)],
    )
    before = project_equipment(decode_equipment_snapshot(v2_snapshot), 8)
    swapped = copy.deepcopy(v2_snapshot)
    swapped["owned"] = [3, 4]
    swapped["containers"][0]["content_id"] = 5
    swapped["slots"][1]["item_id"] = 3
    after = project_equipment(decode_equipment_snapshot(swapped), 8)
    registry, _ = equipment_registry()
    expected = {registry[name]: 1 for name in ("Holster", "WeaponDagger", "WeaponCrossbow")}
    assert carried_item_counts(before) == carried_item_counts(after) == expected
    # The legacy slot projection really does change here.
    before["inventory"] = np.zeros((13, 8), dtype=np.int32)
    after["inventory"] = before["inventory"].copy()
    before["inventory"][0, 1] = 100
    after["inventory"][0, 1] = 200
    assert AutoDancerLiveEnv._inventory_pickup_events(before, after) == []
    assert RewardTracker._inventory_types(before) == RewardTracker._inventory_types(after)
