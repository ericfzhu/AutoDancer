"""Fixed-action world fixtures in a private game copy; never trains a policy."""

import sys
from pathlib import Path

import qualify_equipment as runner

from autodancer.constants import Action
from autodancer.world_observation import FACTS, world_registry


def entity(name, x=1, y=0, **components):
    return dict(name=name, x=x, y=y, fields=components)


CASES = [
    (
        "overlap",
        dict(
            worldEntities=[
                entity("Slime", health=dict(health=7, maxHealth=7)),
                entity("SpikeTrap"),
                entity("WeaponLongsword"),
                entity("PriceTagGold", priceTagCostCurrency=dict(cost=13)),
                entity("ChestLocked"),
            ]
        ),
        [Action.WAIT],
    ),
    (
        "prices",
        dict(
            worldEntities=[
                entity("PriceTagGold", priceTagCostCurrency=dict(cost=17)),
                entity("PriceTagDiamond", priceTagCostCurrency=dict(cost=3)),
                entity("PriceTagHealth", priceTagCostHealth=dict(cost=2)),
            ]
        ),
        [Action.WAIT],
    ),
    (
        "charge",
        dict(
            worldEntities=[entity("Minotaur", ai=dict(active=False), charge=dict(active=False))],
            worldChange={1: dict(charge=dict(active=True, direction=1))},
        ),
        [Action.WAIT],
    ),
    (
        "shield",
        dict(
            worldEntities=[entity("Armoredskeleton", ai=dict(active=False))],
            worldChange={1: dict(shield=dict(active=False))},
        ),
        [Action.WAIT],
    ),
    (
        "status",
        dict(worldEntities=[entity("Slime", freezable=dict(remainingTurns=5))]),
        [Action.WAIT],
    ),
    (
        "visibility",
        dict(
            worldEntities=[
                entity("Slime", sprite=dict(visible=False)),
                entity("Minotaur", x=12, ai=dict(active=False)),
            ]
        ),
        [Action.WAIT],
    ),
    (
        "boss_forms",
        dict(
            worldEntities=[
                entity(name, ai=dict(active=False))
                for name in [
                    "Deathmetal",
                    "DeathmetalPhase2",
                    "DeathmetalPhase3",
                    "DeathmetalPhase4",
                ]
            ]
        ),
        [Action.WAIT],
    ),
    (
        "shrine",
        dict(
            worldEntities=[entity("ShrineOfWar")], worldChange={1: dict(shrine=dict(active=True))}
        ),
        [Action.WAIT],
    ),
]
for tile in [
    "Water",
    "Ice",
    "HotCoals",
    "Lava",
    "Tar",
    "Ooze",
    "DoorHorizontal",
    "LockedDoorHorizontal",
    "LockedStairs",
    "Stairs",
]:
    CASES.append(("terrain:" + tile, dict(worldTiles=[dict(x=1, y=0, name=tile)]), [Action.WAIT]))


for name, component, before, after in [
    (
        "Blademaster",
        "parryCounterAttack",
        dict(active=False, direction=0),
        dict(active=True, direction=1),
    ),
    ("Ogre", "clonking", dict(active=False, direction=0), dict(active=True, direction=1)),
    (
        "Bell",
        "deadRingerBell",
        dict(rung=False, remainingTurns=0),
        dict(rung=True, remainingTurns=3),
    ),
    ("Gargoyle2", "stasis", dict(active=True), dict(active=False)),
    ("DeadRingerPhase2", "supercharge", dict(active=False), dict(active=True)),
]:
    CASES.append(
        (
            "phase:" + component,
            dict(
                worldEntities=[entity(name, **{component: before})],
                worldChange={1: {component: after}},
            ),
            [Action.WAIT],
        )
    )


def validate(case):
    name = case["name"].removeprefix("effect:")
    first, last = case["reset"], case["steps"][-1]
    for state in [first, *case["steps"]]:
        world = state["world"]
        assert world and world["current"]
        actual = {r["id"]: r for r in world["entities"]}
        for entry in state["fixture"]["world"]:
            expected = entry["visible"] and abs(entry["x"]) <= 10 and abs(entry["y"]) <= 10
            assert (entry["id"] in actual) == expected, entry
            if expected:
                assert actual[entry["id"]]["name"] == entry["name"]
        shown = [
            r
            for r in world["entities"]
            if r["displayed"] or state["world_observation_contract"]["profile"] == "internal-v1"
        ]
        assert sum(r[0] != 0 for r in state["world_tensors"]["world_entities"]) == len(shown)

    def item(state, identity):
        return next(r for r in state["world"]["entities"] if r["id"] == identity)

    if name == "overlap":
        ids = [r["id"] for r in first["fixture"]["world"]]
        assert len(ids) == 5
        assert {(item(first, i)["x"], item(first, i)["y"]) for i in ids} == {(1, 0)}
        assert item(first, ids[0])["facts"]["health.health"] == [7, 1, 1]
        assert item(first, ids[3])["facts"]["priceTagCostCurrency.cost"] == [13, 1, 1]
    elif name == "prices":
        rows = [item(first, r["id"]) for r in first["fixture"]["world"]]
        assert rows[0]["currency"] != rows[1]["currency"]
        assert rows[2]["facts"]["priceTagCostHealth.cost"] == [2, 1, 1]
        if first["world_observation_contract"]["profile"] == "player-v1":
            assert all(r["displayed"] for r in rows)
    elif name in {"charge", "shield", "shrine"}:
        identity = first["fixture"]["world"][0]["id"]
        field = name + ".active"
        assert item(first, identity)["facts"][field] != item(last, identity)["facts"][field]
    elif name == "status":
        row = item(first, first["fixture"]["world"][0]["id"])
        assert row["facts"]["freezable.active"] == [1, 1, 1]
        if first["world_observation_contract"]["profile"] == "player-v1":
            # The observation reveals frozen state, not the internal duration.
            assert all(
                r[FACTS.index("freezable.remainingTurns")][1] == 0
                for r in first["world_tensors"]["world_facts"]
            )
    elif name == "visibility":
        assert item(first, first["fixture"]["world"][0]["id"])["displayed"] is False
    elif name.startswith("phase:"):
        identity = first["fixture"]["world"][0]["id"]
        component = name[6:] + "."
        a, b = item(first, identity)["facts"], item(last, identity)["facts"]
        assert any(a[field] != b[field] for field in FACTS if field.startswith(component))
    elif name.startswith("terrain:"):
        assert (
            first["world_tensors"]["world_tiles"][10][11]
            == world_registry()[0]["tile_ids"][name[8:]]
        )


if __name__ == "__main__":
    runner.EFFECT_CASES = CASES
    runner.validate_effect = validate
    runner.EXTRA_EVIDENCE_FILES = [Path(__file__)]
    if "--failure-checks" in sys.argv:
        from qualify_world_failures import qualify_failures

        sys.argv.remove("--failure-checks")
        runner.POST_QUALIFICATION = qualify_failures
        runner.EXTRA_EVIDENCE_FILES.append(Path(__file__).with_name("qualify_world_failures.py"))
    sys.argv.extend(["--effects", "--world-observations"])
    runner.main()
