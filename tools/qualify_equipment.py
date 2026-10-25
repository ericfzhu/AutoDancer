"""Focused equipment exporter checks in a private host; no learning/training.

Fixtures are applied once per reset in the private mod only. They establish
specific handler/state transitions, not natural acquisition or full item support.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from probe_throw_semantics import PrivateSupervisor

from autodancer.constants import Action
from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.supervisor import SupervisorConfig
from autodancer.observation import observation_space

CASES = (
    (1015001, "dagger_throw", [Action.THROW, Action.RIGHT]),
    (1015002, "dagger_repeat", [Action.THROW, Action.THROW, Action.WAIT]),
    (1015003, "crossbow_reload", [Action.THROW]),
    (1015004, "full_crossbow_boots", [Action.THROW]),
    (1015005, "empty_crossbow_boots", [Action.THROW, Action.THROW]),
    (1015006, "dagger_boots", [Action.THROW]),
    (1015007, "bomb_terminal", [Action.BOMB] + [Action.WAIT] * 8),
)
EXTENDED_CASES = (
    (1015020, "holster_swap", [Action.ITEM_2, Action.ITEM_2]),
    (1015021, "backpack_items", [Action.ITEM_2, Action.ITEM_1]),
    (1015022, "bag_holding_items", [Action.ITEM_2, Action.ITEM_1]),
    (1015023, "fireball_blood_cast", [Action.SPELL_1, Action.SPELL_1]),
    (1015024, "war_drum", [Action.ITEM_1, Action.ITEM_1, Action.RIGHT]),
    (1015025, "blood_drum", [Action.ITEM_1, Action.RIGHT]),
    (1015026, "empty_inputs", [Action.ITEM_1, Action.ITEM_2, Action.SPELL_1, Action.SPELL_2]),
    (1015027, "empty_throw", [Action.THROW, Action.RIGHT, Action.THROW]),
    (1015028, "empty_bomb", [Action.BOMB, Action.BOMB, Action.RIGHT]),
)
MATERIALS = ("", "Titanium", "Obsidian", "Golden", "Blood", "Glass")
THROWABLES = tuple(
    "Weapon" + material + kind for kind in ("Dagger", "Spear") for material in MATERIALS
) + (
    "WeaponDaggerShard",
    "WeaponDaggerJeweled",
    "WeaponDaggerFrost",
    "WeaponDaggerPhasing",
    "WeaponDaggerDad",
)
RELOADABLES = tuple("Weapon" + material + "Crossbow" for material in MATERIALS) + (
    "WeaponBlunderbuss",
    "WeaponRifle",
    "BellHammer",
    "Club",
)
EFFECT_CASES = (
    ("dagger_hit", dict(items=["WeaponDagger"], enemies=[[1, 0]]), [Action.RIGHT]),
    ("longsword_reach", dict(items=["WeaponLongsword"], enemies=[[2, 0]]), [Action.RIGHT]),
    ("spear_reach", dict(items=["WeaponSpear"], enemies=[[2, 0]]), [Action.RIGHT]),
    (
        "broadsword_sweep",
        dict(items=["WeaponBroadsword"], enemies=[[1, -1], [1, 0], [1, 1]]),
        [Action.RIGHT],
    ),
    ("bow_range", dict(items=["WeaponBow"], enemies=[[3, 0]]), [Action.RIGHT]),
    (
        "crossbow_fire",
        dict(items=["WeaponCrossbow"], enemies=[[2, 0]]),
        [Action.RIGHT] * 3 + [Action.THROW],
    ),
    ("glass_break", dict(items=["WeaponGlassDagger"], incomingDamage=1), [Action.WAIT]),
    ("glass_armor", dict(items=["ArmorGlass"], incomingDamage=3), [Action.WAIT]),
    ("plate_armor", dict(items=["ArmorPlatemail"], incomingDamage=4), [Action.WAIT]),
    ("ring_war", dict(items=["RingWar"], enemies=[[1, 0]]), [Action.RIGHT]),
    ("peace_replacement", dict(items=["RingPeace"], maxHealth=6, replace="RingWar"), [Action.WAIT]),
    ("blood_drum_hit", dict(items=["BloodDrum"], enemies=[[1, 0]]), [Action.ITEM_1, Action.RIGHT]),
    ("courage_kill", dict(items=["RingCourage"], enemies=[[1, 0]], enemyHealth=1), [Action.RIGHT]),
    ("dig_dirt", dict(wall="DirtWall"), [Action.RIGHT]),
    ("dig_stone", dict(items=["ShovelTitanium"], wall="StoneWall"), [Action.RIGHT]),
    ("blood_dig", dict(items=["ShovelBlood"], wall="StoneWall"), [Action.RIGHT]),
    ("glass_dig", dict(items=["ShovelGlass"], wall="StoneWall"), [Action.RIGHT]),
    ("stash_contents", dict(stash=["WeaponCrossbow", "Food1"]), [Action.WAIT]),
    ("stash_prototypes", dict(stashPrototypes=["Food1", "Food1"], hud=True), [Action.WAIT]),
    ("ground_pickup", dict(pickup="WeaponSpear"), [Action.RIGHT]),
    ("throw_hit", dict(items=["WeaponDagger"], enemies=[[3, 0]]), [Action.THROW, Action.RIGHT]),
    ("spell_fireball", dict(items=["SpellFireball"], enemies=[[3, 0]]), [Action.SPELL_1]),
    ("spell_heal", dict(items=["SpellHeal"], health=10), [Action.SPELL_1]),
    ("spell_shield", dict(items=["SpellShield"], incomingDamage=3), [Action.SPELL_1]),
    ("spell_freeze", dict(items=["SpellFreezeEnemies"], enemies=[[1, 0]]), [Action.SPELL_1]),
    (
        "spell_bomb",
        dict(items=["SpellBomb"], enemies=[[1, 0]]),
        [Action.SPELL_1, Action.LEFT, Action.LEFT, Action.LEFT, Action.WAIT, Action.WAIT],
    ),
    ("spell_charm", dict(items=["SpellCharm"], enemies=[[1, 0]]), [Action.SPELL_1]),
    ("spell_transmute", dict(items=["SpellTransmute"], pickup="RingBecoming"), [Action.SPELL_1]),
    (
        "boots_leap",
        dict(items=["FeetBootsLeaping"], hud=True),
        [Action.RIGHT, Action.THROW, Action.RIGHT],
    ),
    ("boots_lunge", dict(items=["FeetBootsLunging"], enemies=[[2, 0]]), [Action.RIGHT]),
    ("glass_high_resistance", dict(items=["ShovelGlass"], wall="ShopWall"), [Action.RIGHT]),
    ("heal_after_hit", dict(incomingDamage=2, health=10, healAfterDamage=2), [Action.WAIT]),
    (
        "blood_regeneration",
        dict(
            items=["WeaponBloodDagger"],
            health=10,
            enemies=[[1, 0]],
            enemyHealth=1,
            playerFields={"regenerationKillCounter": {"killCount": 9}},
        ),
        [Action.RIGHT],
    ),
    (
        "spell_recharge",
        dict(
            items=["SpellFireball"],
            enemies=[[1, 0]],
            enemyHealth=1,
            itemFields={"spell": {"spellCooldownKills": {"remainingKills": 1}}},
        ),
        [Action.RIGHT],
    ),
    (
        "second_spell",
        dict(grant=["SpellHeal", "SpellShield"], health=10, incomingDamage=3),
        [Action.SPELL_2, Action.SPELL_1],
    ),
    ("potion_revive", dict(items=["MiscPotion"], health=2, incomingDamage=3), [Action.WAIT]),
    ("no_return", dict(items=["NoReturnShrineSquare"]), [Action.RIGHT, Action.LEFT]),
    ("earthquake_damage", dict(items=["ScrollEarthquake"], enemies=[[1, 0]]), [Action.ITEM_1]),
)

# These are registered objects, but are not a successful Bard ground pickup in
# this build: one banned scroll, internal markers/placeholders, a familiar and
# co-op link objects. They remain representable when engine-created/owned.
NON_WALK_PICKUPS = frozenset(
    {
        "Coldsteel_ScrollEnchantLeek",
        "EnemyPlaceholder",
        "FriendlyFireTeamChanger",
        "ItemPlaceholder",
        "ItemStash",
        "Marv",
        "NoReturnModeSquare",
        "NoReturnShrineSquare",
        "ShrineMarkerRhythm",
        "SoulLinkCoop",
        "SoulLinkDad",
        "quirks_DummyItem",
    }
)


def lua_literal(value):
    if isinstance(value, dict):
        return (
            "{" + ",".join(f"[{lua_literal(k)}]={lua_literal(v)}" for k, v in value.items()) + "}"
        )
    if isinstance(value, list):
        return "{" + ",".join(map(lua_literal, value)) + "}"
    return json.dumps(value)


FIXTURE = r"""
    if player and CurrentLevel.getSequentialNumber() == 1
        and fixtureLevel ~= CurrentLevel.getUniqueID() then
        fixtureLevel = CurrentLevel.getUniqueID()
        local seed = tonumber(CurrentLevel.getSeed())
        local Inventory = require "necro.game.item.Inventory"
        if seed == 1015003 or seed == 1015004 or seed == 1015005 then
            local crossbow = Inventory.replace("WeaponCrossbow", player)
            assert(crossbow and crossbow.weaponReloadable, "Crossbow fixture failed")
            crossbow.weaponReloadable.ammo = seed == 1015004 and 3 or 0
        end
        if seed == 1015004 or seed == 1015005 or seed == 1015006 then
            local boots = Inventory.replace("FeetBootsLeaping", player)
            assert(boots and boots.itemToggleable, "Boots fixture failed")
            boots.itemToggleable.active = true
        end
        if seed == 1015020 then
            local holster = Inventory.grant("Holster", player)
            Inventory.swapWithHolster(player, holster)
            local crossbow = Inventory.grant("WeaponCrossbow", player)
            crossbow.weaponReloadable.ammo = 1
        elseif seed == 1015021 or seed == 1015022 then
            Inventory.grant(seed == 1015021 and "HudBackpack" or "BagHolding", player)
            Inventory.grant("Food1", player)
            Inventory.grant("Food2", player)
        elseif seed == 1015023 then
            Inventory.grant("SpellFireball", player)
            player.health.maxHealth = 20
            player.health.health = 20
        elseif seed == 1015024 or seed == 1015025 then
            Inventory.grant(seed == 1015024 and "WarDrum" or "BloodDrum", player)
            player.health.maxHealth = 20
            player.health.health = 20
        end
        if fixtureItems and fixtureItems[seed] then
            local item = Inventory.replace(fixtureItems[seed], player)
            if fixtureReloads[seed] then
                assert(item and item.weaponReloadable, "Missing reloadable fixture")
                item.weaponReloadable.ammo = 0
            end
        end
    end
"""


def snapshot(obs, info):
    state = info["equipment_state"]
    assert state and state["current"], "Missing current live equipment extension"
    assert info.get("telemetry_timing") == "settled-input-v1"
    assert len(state["controls"]) == 6
    assert all(c["availability"] == "unknown" for c in state["controls"])
    assert all(
        f["known"] for item in state["items"] for f in item["facts"].values() if f["applicable"]
    ), "An applicable equipment field was not captured"
    contract = info.get("equipment_observation_contract")
    if contract is not None:
        assert observation_space(equipment_capacity=contract["capacity"]).contains(obs)
        count = (
            len(state["display"]["items"])
            if contract["profile"] == "player-v1"
            else len(state["items"])
        )
        assert int((obs["equipment_items"][:, 0] != 0).sum()) == count
    return {
        "equipment_observation_contract": contract,
        "equipment_tensors": {
            key: obs[key].tolist()
            for key in (
                "equipment_items",
                "equipment_facts",
                "equipment_current",
                "equipment_effects",
                "equipment_world",
                "equipment_world_current",
            )
            if key in obs
        },
        "equipment": state,
        "fixture": info.get("equipment_fixture"),
        "sequence": info["sequence"],
        "telemetry_timing": info.get("telemetry_timing"),
        "bridge": info.get("bridge"),
        "position": obs["player"][4:6].tolist(),
        "equipment_controls": obs["equipment_controls"].tolist(),
        "inventory": obs["inventory"].tolist(),
        "action_mask": obs["action_mask"].tolist(),
        "raw_events": info["raw_events"],
        "reward_components": info.get("reward_components", {}),
        "outcome": info.get("action_outcome"),
        "frame_bytes": info["frame_bytes"],
        "health": int(obs["player"][0]),
        "max_health": int(obs["player"][1]),
    }


def fact(state, name, field):
    items = [item for item in state["equipment"]["items"] if item["name"] == name]
    assert len(items) == 1, f"Expected exactly one {name}"
    value = items[0]["facts"][field]
    assert value["known"] and value["applicable"], f"Missing {name}.{field}"
    return value["value"]


def owner(state):
    return next(c["owner"] for c in state["equipment"]["controls"] if c["action"] == "THROW")


def validate_effect(case):
    states = [case["reset"], *case["steps"]]
    name = case["name"].removeprefix("effect:")
    first, last = states[0], states[-1]

    def damage(state, kind):
        return sum(e["amount"] for e in state["raw_events"] if e["kind"] == kind)

    def names(state):
        return {i["name"] for i in state["equipment"]["items"]}

    if name in {
        "dagger_hit",
        "longsword_reach",
        "spear_reach",
        "broadsword_sweep",
        "bow_range",
        "ring_war",
    }:
        hit = 2 if name == "ring_war" else 1
        assert all(e["health"] == 20 - hit for e in last["fixture"]["enemies"])
        assert damage(last, "enemy_damage") == hit * len(last["fixture"]["enemies"])
        assert first["position"] == last["position"]
    elif name == "crossbow_fire":
        assert [fact(s, "WeaponCrossbow", "ammo") for s in states] == [3, 2, 1, 0, 3]
        assert [s["fixture"]["enemies"][0]["health"] for s in states] == [20, 19, 18, 17, 17]
    elif name in {"glass_break", "glass_armor", "plate_armor"}:
        loss = 0 if name == "glass_armor" else 1
        assert first["health"] - last["health"] == loss
        assert damage(last, "player_damage") == loss
        target = {
            "glass_break": "WeaponGlassDagger",
            "glass_armor": "ArmorGlass",
            "plate_armor": "ArmorPlatemail",
        }[name]
        assert (target in names(last)) == (name == "plate_armor")
    elif name == "peace_replacement":
        assert [s["max_health"] for s in states] == [8, 6]
        assert fact(first, "RingPeace", "pickup_health_gained") == 2
        assert "RingWar" in names(last) and "RingPeace" not in names(last)
        assert damage(last, "item_collected") == 0
    elif name == "blood_drum_hit":
        assert [fact(s, "BloodDrum", "damage_boost_turns") for s in states] == [0, 1, 0]
        assert damage(states[1], "player_damage") == 1
        assert last["fixture"]["enemies"][0]["health"] == 0
        assert damage(last, "enemy_kill") == 1
        assert damage(last, "enemy_damage") == 20, "Overkill must not manufacture damage credit"
    elif name == "courage_kill":
        assert last["position"] == [first["position"][0] + 1, first["position"][1]]
        assert fact(last, "RingCourage", "dash_on_kill") is True
        assert fact(last, "RingCourage", "invincible") is True
        assert damage(last, "enemy_kill") == 1
    elif name in {"dig_dirt", "dig_stone", "blood_dig", "glass_dig"}:
        assert first["fixture"]["wall"] in {"DirtWall", "StoneWall"}
        assert last["fixture"]["wall"] == "Floor"
        assert first["position"] == last["position"]
        assert damage(last, "player_damage") == (1 if name == "blood_dig" else 0)
    elif name == "stash_contents":
        for state in states:
            links = state["equipment"]["containers"]
            assert [r["index"] for r in links] == [1, 2]
            assert all(r["kind"] == "storage" for r in links)
            assert fact(state, "WeaponCrossbow", "ammo") == 3
            assert {"ItemStash", "WeaponCrossbow", "Food1"} <= names(state)
    elif name == "stash_prototypes":
        for state in states:
            contents = [i for i in state["equipment"]["items"] if i["name"] == "Food1"]
            assert len(contents) == 2 and len({i["id"] for i in contents}) == 2
            shown = {i["id"] for i in state["equipment"]["display"]["items"]}
            assert all(i["id"] in shown for i in contents)
            assert all(i["id"] >= 2**52 for i in contents)
    elif name == "ground_pickup":
        assert "WeaponSpear" not in names(first) and "WeaponSpear" in names(last)
        events = [e for e in last["raw_events"] if e["kind"] == "item_collected"]
        assert len(events) == 1 and events[0]["data"]["source"] == "engine_pickup"
        assert events[0]["data"]["item_name"] == "WeaponSpear"
    elif name == "throw_hit":
        assert first["position"] == last["position"]
        assert last["fixture"]["enemies"][0]["health"] < 20
        assert "WeaponDagger" not in names(last)
    elif name in {
        "spell_fireball",
        "spell_heal",
        "spell_shield",
        "spell_freeze",
        "spell_bomb",
        "spell_charm",
        "spell_transmute",
    }:
        target = {
            "spell_fireball": "SpellFireball",
            "spell_heal": "SpellHeal",
            "spell_shield": "SpellShield",
            "spell_freeze": "SpellFreezeEnemies",
            "spell_bomb": "SpellBomb",
            "spell_charm": "SpellCharm",
            "spell_transmute": "SpellTransmute",
        }[name]
        assert fact(states[1], target, "cooldown_kills") > 0
        if name == "spell_heal":
            assert last["health"] == first["health"] + 1
        elif name == "spell_shield":
            assert last["health"] == first["health"] and damage(last, "player_damage") == 0
        elif name == "spell_freeze":
            assert last["fixture"]["enemies"][0]["frozen"] > 0
        elif name == "spell_transmute":
            assert last["fixture"]["pickup"]["name"] == "RingWonder"
        elif name in {"spell_fireball", "spell_bomb"}:
            assert last["fixture"]["enemies"][0]["health"] < 20
        elif name == "spell_charm":
            assert last["fixture"]["enemies"][0]["charmed"] is True
    elif name == "boots_leap":
        assert [s["position"][0] - first["position"][0] for s in states] == [0, 2, 2, 3]
    elif name == "boots_lunge":
        assert last["fixture"]["enemies"][0]["health"] == 16
        assert last["position"] != first["position"]
    elif name == "glass_high_resistance":
        assert last["fixture"]["wall"] == "Floor"
        assert "ShovelGlass" not in names(last)
    elif name == "heal_after_hit":
        assert last["health"] == first["health"]
        assert damage(last, "player_damage") == 2
    elif name == "blood_regeneration":
        assert first["equipment"]["player_effects"]["regeneration_kills"]["value"] == 9
        assert last["equipment"]["player_effects"]["regeneration_kills"]["value"] == 0
        assert last["health"] == first["health"] + 1
    elif name == "spell_recharge":
        assert [fact(s, "SpellFireball", "cooldown_kills") for s in states] == [1, 0]
        assert damage(last, "enemy_kill") == 1
    elif name == "second_spell":
        assert fact(states[1], "SpellShield", "cooldown_kills") > 0
        assert fact(states[1], "SpellHeal", "cooldown_kills") == 0
        assert damage(states[1], "player_damage") == 0
        assert last["health"] == first["health"] + 1
        assert fact(last, "SpellHeal", "cooldown_kills") > 0
    elif name == "potion_revive":
        assert "MiscPotion" not in names(last)
        assert last["health"] > first["health"]
        assert damage(last, "player_damage") > 0
    elif name == "no_return":
        assert fact(states[1], "NoReturnShrineSquare", "no_return_active") is True
        assert any(r["kind"] == "no_return" for r in states[1]["equipment"]["world"])
        assert "NoReturnShrineSquare" not in names(last)  # Shrine variant is single-use.
        assert damage(last, "player_damage") == 1
    elif name == "earthquake_damage":
        loss = first["fixture"]["enemies"][0]["health"] - last["fixture"]["enemies"][0]["health"]
        target = last["fixture"]["enemies"][0]["id"]
        reported = sum(
            e["amount"]
            for e in last["raw_events"]
            if e["kind"] == "enemy_damage" and e["entity_id"] == target
        )
        assert loss > 0 and reported == loss
        assert "ScrollEarthquake" not in names(last)
    else:
        raise AssertionError("Missing effect assertions: " + name)


def validate_case(case):
    states = [case["reset"], *case["steps"]]
    name = case["name"]
    if name.startswith("pickup:"):
        assert case["collected"] == (name[7:] not in NON_WALK_PICKUPS), (
            "Ground pickup behavior changed"
        )
        assert len(case["pickup_events"]) <= 1, "Repeated pickup credit for one ground object"
    if name.startswith("effect:"):
        validate_effect(case)
    if name.startswith("throw:"):
        target = name.split(":")[1]
        assert owner(states[0]) == target
        assert fact(states[0], target, "activation_active") is False
        assert fact(states[1], target, "activation_active") is True
        assert states[2]["position"] == states[0]["position"]
        assert not any(i["name"] == target for i in states[2]["equipment"]["items"])
    if name.startswith("reload:"):
        target = name.split(":")[1]
        assert fact(states[0], target, "ammo") == 0
        capacity = fact(states[0], target, "ammo_capacity")
        increment = fact(states[0], target, "ammo_per_reload")
        for before, after in zip(states[:-1], states[1:], strict=True):
            assert owner(before) == target
            assert fact(after, target, "ammo") == min(
                capacity, fact(before, target, "ammo") + increment
            )
        assert fact(states[-1], target, "ammo") == capacity
    for before, after in zip(states[:-1], states[1:], strict=True):
        assert after["sequence"] > before["sequence"]
        assert after["bridge"]["requested_action"] == after["action"]
    if name.startswith("dagger_") and name != "dagger_boots":
        assert owner(states[0]) == "WeaponDagger"
        assert fact(states[0], "WeaponDagger", "activation_active") is False
        assert fact(states[1], "WeaponDagger", "activation_active") is True
        if (states[1].get("equipment_observation_contract") or {}).get("profile") != "player-v1":
            assert states[1]["equipment_controls"] == [1, 1, 1]
        if name == "dagger_throw":
            assert owner(states[-1]) is None
            assert not any(i["name"] == "WeaponDagger" for i in states[-1]["equipment"]["items"])
            assert states[-1]["position"] == states[0]["position"]
        else:
            assert all(fact(s, "WeaponDagger", "activation_active") is True for s in states[1:])
    if name in {"crossbow_reload", "empty_crossbow_boots"}:
        assert owner(states[0]) == "WeaponCrossbow"
        assert fact(states[0], "WeaponCrossbow", "ammo") == 0
        assert fact(states[1], "WeaponCrossbow", "ammo") == 3
        assert fact(states[1], "WeaponCrossbow", "ammo_capacity") == 3
    if name in {"full_crossbow_boots", "empty_crossbow_boots", "dagger_boots"}:
        start = 1 if name == "empty_crossbow_boots" else 0
        assert owner(states[start]) == "FeetBootsLeaping"
        assert fact(states[start], "FeetBootsLeaping", "toggle_active") is True
        assert fact(states[start + 1], "FeetBootsLeaping", "toggle_active") is False
        if name == "dagger_boots":
            assert fact(states[-1], "WeaponDagger", "activation_active") is False
    if name == "holster_swap":
        assert all(len(s["equipment"]["containers"]) == 1 for s in states)
        ids = [s["equipment"]["containers"][0]["content_id"] for s in states]
        assert ids[0] and ids[0] != ids[1] and ids[0] == ids[2]
        assert all(fact(s, "WeaponCrossbow", "ammo") == 1 for s in states)
        if states[0].get("equipment_observation_contract") is not None:
            assert all(
                not any(e.get("kind") == "item_collected" for e in s["raw_events"])
                and s["reward_components"].get("new_item_type", 0) == 0
                for s in states[1:]
            ), "Swapping carried equipment must not create pickup evidence or new-item reward"
    if name in {"backpack_items", "bag_holding_items"}:
        action_slots = [s for s in states[0]["equipment"]["slots"] if s["name"] == "action"]
        assert len(action_slots) == 2
        assert len(states[-1]["equipment"]["items"]) < len(states[0]["equipment"]["items"])
    if name == "fireball_blood_cast":
        assert fact(states[1], "SpellFireball", "cooldown_kills") > 0
        assert fact(states[1], "SpellFireball", "spell_usable") is True
        assert states[1]["health"] - states[2]["health"] == fact(
            states[1], "SpellFireball", "blood_cast_damage"
        )
    if name == "war_drum":
        assert [fact(s, "WarDrum", "combo") for s in states] == [0, 1, 2, 0]
    if name == "blood_drum":
        assert states[0]["health"] - states[1]["health"] == fact(
            states[0], "BloodDrum", "activation_damage"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--equipment-observations", action="store_true")
    parser.add_argument("--weapon-mechanics", action="store_true")
    parser.add_argument("--effects", action="store_true")
    parser.add_argument("--pickup-sweep", action="store_true")
    parser.add_argument("--player-observations", action="store_true")
    parser.add_argument(
        "--exercise-catalog", action="store_true", help="Also attempt item controls, RIGHT and WAIT"
    )
    parser.add_argument(
        "--catalog-sweep",
        action="store_true",
        help="Inspect each slotted item in an isolated reset",
    )
    args = parser.parse_args()
    if args.pickup_sweep and (
        args.effects or args.catalog_sweep or args.weapon_mechanics or args.extended
    ):
        parser.error("--pickup-sweep is a separate suite")
    if args.effects and (
        args.catalog_sweep or args.extended or args.exercise_catalog or args.weapon_mechanics
    ):
        parser.error("--effects is a separate suite")
    if args.weapon_mechanics and (args.catalog_sweep or args.extended or args.exercise_catalog):
        parser.error("--weapon-mechanics is a separate suite")
    root, game = args.output.resolve(), args.game_dir.resolve()
    if root.exists():
        parser.error("Output must be fresh")
    root.mkdir(parents=True)
    host = root / "game-host"
    host.mkdir()
    for path in game.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(path, host / path.name)
    for name in ("versions", "dlc"):
        shutil.copytree(game / name, host / name)
    mod = host / "probe-mods/AutoDancer"
    source_mod = Path(__file__).resolve().parents[1] / "mods/AutoDancer"
    shutil.copytree(source_mod, mod)
    cfg = json.loads((game / "config.json").read_text())
    cfg["wos"]["game"]["assets"]["external"]["path"] = game.parent.as_posix()
    cfg["wos"]["game"]["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(cfg))
    entry = mod / "scripts/AutoDancer.lua"
    text = entry.read_text()
    cases = list(CASES) + (list(EXTENDED_CASES) if args.extended else [])
    fixture_items = {}
    fixture_reloads = {}
    effect_fixtures = {}
    if args.catalog_sweep:
        catalog = json.loads((source_mod.parents[1] / "docs/game-control-catalog.json").read_text())
        names = sorted(
            r["id"]
            for r in catalog["current_item_slots"]
            if r["slot"] not in {None, "", "innate", "shrine", "stash"}
        )
        fixture_items = {2100000 + i: name for i, name in enumerate(names)}
        cases = [(seed, "catalog:" + name, []) for seed, name in fixture_items.items()]
    if args.weapon_mechanics:
        cases = []
        for name in THROWABLES:
            for direction in (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT):
                seed = 2200000 + len(cases)
                fixture_items[seed] = name
                cases.append((seed, f"throw:{name}:{direction.name}", [Action.THROW, direction]))
        for name in RELOADABLES:
            seed = 2200000 + len(cases)
            fixture_items[seed] = name
            fixture_reloads[seed] = True
            cases.append(
                (seed, "reload:" + name, [Action.THROW] * (3 if name == "WeaponRifle" else 1))
            )
    if args.effects:
        cases = []
        for i, (name, spec, actions) in enumerate(EFFECT_CASES):
            seed = 2300000 + i
            effect_fixtures[seed] = spec
            cases.append((seed, "effect:" + name, actions))
    if args.pickup_sweep:
        from autodancer.equipment_state import equipment_registry

        cases = []
        for i, name in enumerate(sorted(equipment_registry()[0])):
            seed = 2400000 + i
            effect_fixtures[seed] = dict(pickup=name)
            cases.append((seed, "pickup:" + name, [Action.RIGHT, Action.WAIT]))
    fixture_lua = (
        "local fixtureItems = {"
        + ",".join(f"[{seed}]={json.dumps(name)}" for seed, name in fixture_items.items())
        + "}\n"
        + "local fixtureReloads = {"
        + ",".join(f"[{seed}]=true" for seed in fixture_reloads)
        + "}\n"
    )
    fixture_lua += "local effectFixtures = " + lua_literal(effect_fixtures) + "\n"
    fixture_lua += Path("tools/equipment_effects.lua").read_text() + "\n"
    marker = "    result.equipment_snapshot = Equipment.capture(player)"
    assert text.count(marker) == 1
    text = text.replace(
        "local function buildObservation()",
        fixture_lua + "local fixtureLevel\nlocal function buildObservation()",
    )
    entry.write_text(
        text.replace(
            marker, FIXTURE + "    applyEquipmentEffectFixture(player,result,sequence)\n" + marker
        )
    )
    originals = [game / "config.json", *source_mod.rglob("*.lua")]
    hashes = {str(p): sha256_file(p) for p in originals}
    result = {
        "passed": False,
        "training": False,
        "catalog_sweep": args.catalog_sweep,
        "exercise_catalog": args.exercise_catalog,
        "extended": args.extended,
        "weapon_mechanics": args.weapon_mechanics,
        "effects": args.effects,
        "pickup_sweep": args.pickup_sweep,
        "player_observations": args.player_observations,
        "equipment_observations": args.equipment_observations,
        "cases": [],
        "failures": [],
        "original_hashes": hashes,
        "qualification_hashes": {
            str(p): sha256_file(p)
            for p in [
                Path(__file__),
                Path("tools/equipment_effects.lua"),
                *mod.rglob("*.lua"),
                host / "config.json",
                game / "NecroDancer.wsp",
                game / "versions/v4.2.1.wsp",
                game / "lua51.dll",
                *Path("src/autodancer").rglob("*.py"),
                Path("src/autodancer/data/equipment_registry.json"),
                *sorted((host / "dlc").glob("*.necromod")),
            ]
        },
    }
    atomic_json(root / "result.json", result)
    shutil.copy2(__file__, root / "executed-qualifier.py")
    try:
        with PrivateSupervisor(
            SupervisorConfig(
                game_dir=host,
                mod_dir=mod,
                num_instances=1,
                startup_timeout=45,
                reset_timeout=30,
                max_turns=50,
                affinity_policy="none",
                steam_presence_worker=0,
                curriculum_commands_enabled=True,
                input_mode="engine-input-v1"
                if args.extended
                or args.exercise_catalog
                or args.equipment_observations
                or args.weapon_mechanics
                or args.effects
                or args.pickup_sweep
                or args.player_observations
                else "legacy",
                equipment_profile="player-v1"
                if args.player_observations
                else "internal-v1"
                if args.equipment_observations
                else "legacy",
                profile_root=root / "profiles",
                diagnostic_root=root / "diagnostics",
            )
        ) as supervisor:
            env = supervisor.environment(supervisor.worker_ids[0])
            accept_record = env._accept_record

            def accept_fixture_record(record):
                obs, info = accept_record(record)
                info["equipment_fixture"] = record["observation"].get("equipment_fixture")
                return obs, info

            env._accept_record = accept_fixture_record
            try:
                for seed, name, actions in cases:
                    obs, info = env.reset(seed=seed)
                    case = {"seed": seed, "name": name, "reset": snapshot(obs, info), "steps": []}
                    result["cases"].append(case)
                    if name.startswith("catalog:"):
                        target = name.removeprefix("catalog:")
                        case["retained"] = any(
                            item["name"] == target for item in info["equipment_state"]["items"]
                        )
                        if args.exercise_catalog:
                            selected = [
                                Action[c["action"]]
                                for c in info["equipment_state"]["controls"]
                                if c["owner"] == target
                            ]
                            actions = selected[:1] + [Action.RIGHT, Action.WAIT]
                    for action in actions:
                        obs, _, terminated, truncated, info = env.step(action)
                        if name.startswith(("catalog:", "pickup:")) and (terminated or truncated):
                            case["terminal"] = {
                                "status": info["episode_status"],
                                "action": int(action),
                                "equipment": info["equipment_state"],
                                "events": info["raw_events"],
                            }
                            break
                        if name == "bomb_terminal" and terminated:
                            assert info["episode_status"] == "dead" and not truncated
                            assert info["telemetry_timing"] == "terminal"
                            assert (
                                info["equipment_state"] and not info["equipment_state"]["current"]
                            )
                            case["terminal"] = {
                                "status": info["episode_status"],
                                "bridge": info.get("bridge"),
                                "equipment": info["equipment_state"],
                                "action": int(action),
                            }
                            break
                        assert not terminated and not truncated, "Unexpected episode end"
                        case["steps"].append({"action": int(action), **snapshot(obs, info)})
                    if name == "bomb_terminal":
                        assert "terminal" in case, "Bomb fixture did not produce a terminal"
                    if name.startswith("pickup:"):
                        events = [e for s in case["steps"] for e in s["raw_events"]]
                        events += case.get("terminal", {}).get("events", [])
                        pickups = [
                            e
                            for e in events
                            if e["kind"] == "item_collected"
                            and e.get("data", {}).get("item_name") == name[7:]
                        ]
                        case["pickup_events"] = pickups
                        case["collected"] = bool(pickups)
                    try:
                        validate_case(case)
                    except AssertionError as exc:
                        case["assertion_failure"] = str(exc) or "Effect assertion failed"
                        result["failures"].append(name)
                    atomic_json(root / "result.json", result)
                    if (
                        len(cases) < 80
                        or len(result["cases"]) % 25 == 0
                        or name in result["failures"]
                    ):
                        print(
                            json.dumps(
                                {
                                    "case": name,
                                    "completed": len(result["cases"]),
                                    "total": len(cases),
                                    "passed": name not in result["failures"],
                                }
                            ),
                            flush=True,
                        )
                obs, info = env.reset(
                    seed=1015008,
                    options={
                        "curriculum": {
                            "id": "equipment-floor-transition",
                            "start_level": 2,
                            "profile": "normal",
                        }
                    },
                )
                assert info["floor"] == 2 and info["bridge"]["kind"] == "GOTO"
                result["floor_transition"] = snapshot(obs, info)
                obs, _, terminated, truncated, info = env.step(Action.WAIT)
                assert not terminated and not truncated
                result["after_floor_transition"] = snapshot(obs, info)
                obs, info = env.reset(seed=1015001)
                assert info["floor"] == 1 and info["sequence"] == 0
                assert fact(snapshot(obs, info), "WeaponDagger", "activation_active") is False
            finally:
                env.close()
            handle = supervisor.workers[supervisor.worker_ids[0]]
            result["worker_restarts"] = handle.restart_count
            log_path = handle.log_path
            assert handle.restart_count == 0
        shutil.copy2(log_path, root / "game.log")
        result["passed"] = not result["failures"]
    finally:
        result["original_files_unchanged"] = all(
            sha256_file(Path(p)) == h for p, h in hashes.items()
        )
        result["passed"] = result["passed"] and result["original_files_unchanged"]
        atomic_json(root / "result.json", result)
    if not result["passed"]:
        raise SystemExit("Equipment qualification failed; see result.json")


if __name__ == "__main__":
    main()
