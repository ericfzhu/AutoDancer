"""Generate the readable catalog and compact factual inventory from local evidence."""
# The long literals below are generated Markdown paragraphs and table cells.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from autodancer.experiments.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs/game-control-catalog-current"

FAMILY_NOTES = {
    "Dagger": "THROW arms; direction releases. Starting dagger verified live; other variants not tested.",
    "Spear": "Throwable component; armed/directional semantics need a spear live check.",
    "Crossbow": "Reloadable; sampled base prototype starts with 3/3 ammo and reloads 3 per activation.",
    "Blunderbuss": "Reloadable; sampled prototype starts with 1/1 ammo and reloads 1 per activation.",
    "Rifle": "Reloadable; sampled prototype starts with 0/3 ammo and reloads 1 per activation.",
    "Rapier": "Directional attack with movement-related pattern behavior; do not assume a direction means one-tile walking.",
    "Cat": "Cat O Nine Tails; directional attack/movement pattern needs its own action interpretation.",
    "Flail": "Directional weapon attack; inspect knockback and target pattern separately from damage.",
    "Flower": "Special weapon definition; treat character rules separately from ordinary damaging weapons.",
    "GoldenLute": "Special weapon definition; character-specific directional/combat rules need separate validation.",
    "Axe": "Installed shape definition; not registered in the sampled training configuration.",
    "Cutlass": "Installed shape definition; not registered in the sampled training configuration.",
    "Harp": "Installed shape definition; not registered in the sampled training configuration.",
    "Staff": "Installed shape definition; not registered in the sampled training configuration.",
    "Warhammer": "Installed shape definition; not registered in the sampled training configuration.",
}

EXTRA_WEAPONS = [
    (
        "Trident",
        "mods/Sync/item/WeaponTrident.lua",
        "Synchrony",
        "Separate shape and attack behavior; validate special-action capabilities.",
    ),
    (
        "Plasma Cannon",
        "mods/Sync/item/WeaponPlasmaCannon.lua",
        "Synchrony",
        "Reloadable; source sets ammo 0, capacity 3, reload increment 1; attack depends on ammo.",
    ),
    (
        "Lantern",
        "mods/Sync/item/WeaponLantern.lua",
        "Synchrony / possession",
        "Source explicitly disables weaponThrowable and attaches possession-on-kill behavior.",
    ),
    (
        "Zweihänder",
        "mods/Sync/character/Klarinetta.lua",
        "Klarinetta",
        "Rotating sword; selected control mode can remap ITEM_2 and THROW to rotation.",
    ),
    (
        "Lance",
        "mods/Sync/character/Suzu.lua",
        "Suzu",
        "Source describes dash on kill; character charge/movement state matters.",
    ),
    (
        "Spring Onion",
        "mods/Coldsteel/Coldsteel.lua",
        "Hatsune Miku",
        "Character-specific weapon; base and phasing prototypes appeared in the sample.",
    ),
    (
        "Electric / Phasing Spring Onion",
        "mods/Coldsteel/ColdsteelEnchant.lua",
        "Hatsune Miku",
        "Explicit enchantment variants; electric variant absent from sampled prototypes.",
    ),
    (
        "Shovel Blade",
        "mods/Goldman/GoldmanEntities.lua",
        "Shovel Knight",
        "Hybrid character/shovel combat mechanism; not an ordinary weapon-slot-only abstraction.",
    ),
]

MECHANICS = [
    (
        "Armed throws",
        "Dagger, spear",
        "THROW then direction",
        "Throwable capability; armed state; direction",
        "Inventory lacks itemActivable.active; redundant THROW remains allowed",
        "Dagger live; spear definition",
    ),
    (
        "Reloading and charge",
        "Crossbow, rifle, blunderbuss, plasma cannon",
        "THROW / directional attack",
        "Ammo, capacity, reload increment, loaded attack",
        "Inventory does not expose weaponReloadable ammo",
        "Prototypes + installed code; no live gun test",
    ),
    (
        "Shared special-action resolution",
        "Weapons, toggle boots, convertible items",
        "THROW",
        "Which equipped item the game resolves for the action",
        "Mask checks weapon presence, rather than the resolved action item",
        "Installed ActionItem/MovementItem/Weapon code",
    ),
    (
        "Toggle movement equipment",
        "Boots of Leaping, Boots of Lunging",
        "Shared activation; directions",
        "Toggle state, distance, hazards, activation routing",
        "Toggle bit exists, but availability and movement assumptions need validation",
        "Current prototypes + MovementItem code",
    ),
    (
        "Weapon storage / swap",
        "Holster; bag/backpack behavior",
        "ITEM_2 in installed action-item handlers",
        "Container contents, slot ownership, swap state",
        "Flat inventory lacks stored contents; verify which holder owns the item action",
        "Holster prototype + ActionItem code; bag behavior not live-tested",
    ),
    (
        "Item conversion",
        "Items carrying itemConvertible",
        "THROW in conversion handler",
        "Current form, target form, eligibility",
        "Weapon-presence mask is not a general conversion-availability test",
        "Installed ActionItem code; no sampled convertible item asserted",
    ),
    (
        "Consumable slots",
        "Food, holy water, scrolls, heart transplants",
        "ITEM_1 / ITEM_2",
        "Item identity, charges, consumption, target rules",
        "Quantity is present; usability mostly means cooldown counters are zero",
        "Current item prototypes",
    ),
    (
        "Combo and health-cost items",
        "War Drum, Blood Drum",
        "Item activation then follow-up actions",
        "Combo progress, next-attack modifier, health cost",
        "Neither drum combo nor activation health cost is explicitly encoded",
        "Current prototypes",
    ),
    (
        "Spell recharge / blood magic",
        "Fireball, freeze, heal, bomb, shield, transmute, charm",
        "SPELL_1 / SPELL_2",
        "Kill/time recharge, blood-cast eligibility and health cost",
        "Cooldowns exist; mask may exclude valid blood casts and needs a live eligibility test",
        "Current spell-item prototypes",
    ),
    (
        "Directional or delayed spells",
        "Fireball; installed dash and berserk abilities",
        "Spell input plus facing/direction or later actions",
        "Aim/facing, pending cast, charge/duration",
        "No general pending-action state; exact controls require per-ability validation",
        "Installed spell definitions; behavior not live-tested",
    ),
    (
        "Bomb-slot substitution",
        "Bomb stacks, infinite bombs; Miku Sing",
        "BOMB",
        "Resolved slot action, quantity, effect and timing",
        "Current mask tests bomb-slot occupancy; occupied slot need not mean an explosive",
        "Current prototypes; Miku Sing occupies bomb slot",
    ),
    (
        "Character-specific remapping",
        "Klarinetta, Suzu, Chaunter, Miku, Shovel Knight",
        "Directions and shared special buttons",
        "Character, selected control mode, rotation, possession, charge, pogo/combo state",
        "Current environment trains Bard; its 11-action contract is not certified for these characters",
        "Installed character scripts",
    ),
    (
        "Rhythm / time control",
        "Bard; rhythm-based characters; heart transplant",
        "Every turn / timed item use",
        "Beat phase, missed-beat rules, temporary timing changes",
        "Bard experiments do not establish correctness for rhythm-based play",
        "Bridge character restriction + installed definitions",
    ),
    (
        "Contextual movement",
        "Digging, attacks, ice/sliding, confusion, knockback, stairs",
        "Directions",
        "Target cell, status effects, weapon mode, movement modifier",
        "Some grid/status fields exist; navigation masking must respect action context",
        "Existing observation/contract/outcome code",
    ),
    (
        "Pickup and transaction effects",
        "Shops, shrines, dropped gear, transmutation",
        "Movement/interactions",
        "Price, health/currency, inventory replacement and item bans",
        "Some prices/flags exist; action eligibility cannot be inferred from occupancy alone",
        "Existing rich observation and installed item rules",
    ),
    (
        "Passive automatic effects",
        "Potion, frost protection, movement-triggered equipment",
        "No dedicated button / triggered by another action",
        "Remaining item, trigger condition, effect duration",
        "Do not invent an active action for a passive item",
        "Current prototypes; exact triggers need dedicated tests",
    ),
]


def main():
    prototypes = json.loads((EVIDENCE / "active-prototypes.json").read_text())
    definitions = json.loads((EVIDENCE / "definitions.json").read_text())
    supplemental_path = EVIDENCE / "supplemental-definitions.json"
    supplemental = json.loads(supplemental_path.read_text()) if supplemental_path.exists() else {}
    all_definitions = {**definitions, **supplemental}
    weapons = []
    for row in prototypes:
        data = row["data"]
        if row["category"] != "item" or "weapon" not in data:
            continue
        family = (data.get("weaponType") or {}).get("name", "special")
        if row["name"] == "WeaponFlower":
            family = "Flower"
        weapons.append(
            {
                "id": row["name"],
                "name": data["friendlyName"]["name"],
                "family": family,
                "material": (data.get("weaponMaterial") or {}).get("name"),
                "throwable": "weaponThrowable" in data,
                "reload": data.get("weaponReloadable"),
                "activable": "itemActivable" in data,
                "control_components": {
                    k: v
                    for k, v in data.items()
                    if k.startswith("weapon")
                    or k in ("itemActivable", "itemToggleable", "itemConvertible", "itemSlot")
                },
                "pool_markers": [k for k in data if k.startswith("itemPool")],
                "ban_markers": [k for k in data if k.startswith("itemBan") or k == "itemGlobalBan"],
                "availability": "registered prototype; natural loot eligibility not established",
                "behavior_evidence": "live-tested starting dagger"
                if row["name"] == "WeaponDagger"
                else "prototype only",
            }
        )
    assert len(weapons) == 76 and len({w["id"] for w in weapons}) == 76
    families = []
    for path in sorted(n for n in definitions if "/weapon/shape/" in n):
        family = Path(path).stem
        families.append(
            {
                "family": family,
                "source": path,
                "archive": definitions[path]["archive"],
                "registered_variants": [w["id"] for w in weapons if w["family"] == family],
                "controls": FAMILY_NOTES.get(
                    family,
                    "Directional weapon attack; no throw/reload component on sampled variants.",
                ),
            }
        )
    assert len(families) == 19
    for _, source, _, _ in EXTRA_WEAPONS:
        assert source in all_definitions
    payload = {
        "schema_version": 1,
        "game_version": "v4.2.1-b5713",
        "steam_build": "22938426",
        "scope": "Installed weapon families plus registered item prototypes in the current training configuration; not a loot simulator",
        "evidence_sha256": {
            n: sha256_file(EVIDENCE / n)
            for n in (
                "active-prototypes.json",
                "definitions.json",
                "supplemental-definitions.json",
                "game.log",
            )
            if (EVIDENCE / n).exists()
        },
        "core_weapon_families": families,
        "additional_definitions": [
            {"name": n, "source": s, "scope": c, "controls": b} for n, s, c, b in EXTRA_WEAPONS
        ],
        "registered_weapon_items": weapons,
        "mechanics": [
            {
                "mechanic": m,
                "examples": e,
                "inputs": i,
                "required_state": s,
                "current_gap": g,
                "evidence": v,
            }
            for m, e, i, s, g, v in MECHANICS
        ],
        "current_item_slots": [
            {
                "id": r["name"],
                "name": (r["data"].get("friendlyName") or {}).get("name"),
                "slot": (r["data"].get("itemSlot") or {}).get("name"),
            }
            for r in prototypes
            if r["category"] == "item"
        ],
        "limitations": [
            "Only starting-dagger control sequences tested live.",
            "Prototype fields are a selected component/field inventory, not an exhaustive serialization.",
            "Registration is not spawnability, ownership, unlock status, or Bard eligibility.",
            "DLC-disabled entries were cataloged from definitions without enabling those DLCs.",
            "Custom player mods and every editor/PvP-only weapon combination are outside the general-game family list.",
        ],
    }
    (ROOT / "docs/game-control-catalog.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Weapon and control-mechanics catalog",
        "",
        "Installed build: **v4.2.1-b5713 / Steam 22938426**. Captured 2026-09-08.",
        "",
        "This catalog separates installed definitions, prototypes registered under the current training configuration, and behavior verified live. Registration does not establish that an item can appear as normal Bard loot. The production mod blacklists Amplified, DynChar, and Synchrony; some other DLC/special prototypes still register. No gameplay packages were enabled for this catalog.",
        "",
        "The private read-only probe enumerated **250 item prototypes and 13 playable-character prototypes**. Of the items, **76 carry a weapon component**; that includes character, tutorial, and special-purpose items. An additional weapon-bearing character prototype (Eli) is excluded from the item count. Only the starting dagger has a live input-sequence qualification.",
        "",
        "Machine-readable companion: [game-control-catalog.json](game-control-catalog.json). Full local component evidence: `runs/game-control-catalog-current/active-prototypes.json`.",
        "",
        "## Core weapon families",
        "",
        "These **19 shape definitions** occur in the installed core archives. A zero in the third column means no matching prototype was registered in this capture, not that the weapon does not exist in the game.",
        "",
        "| Family | Control considerations | Registered item variants |",
        "| --- | --- | ---: |",
    ]
    for f in families:
        display = {"Cat": "Cat O Nine Tails", "GoldenLute": "Golden Lute"}.get(
            f["family"], f["family"]
        )
        lines.append(f"| {display} | {f['controls']} | {len(f['registered_variants'])} |")
    lines += [
        "",
        "Source directory: `scripts/necro/game/data/item/weapon/shape/`, with effective patch overrides applied. The JSON records the source archive for each family.",
        "",
        "## Additional installed DLC and character weapons",
        "",
        "These are definition-backed entries; their controls were not exercised in this catalog run.",
        "",
        "| Weapon | Scope | Control considerations |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {n} | {c} | {b} |" for n, _, c, b in EXTRA_WEAPONS]
    lines += [
        "",
        "Spring Onion has separately declared base, electric, and phasing forms. The Shovel Blade is listed because it changes combat controls even though a weapon-slot-only search can miss it. Special registered items such as Mallet Of The Dad, Clonking Stick, Squishy Piston, and the two wands are listed in the exact inventory below; they are not asserted to be ordinary player loot.",
        "",
        "## Materials and variants",
        "",
        "Core material definitions: **base, titanium, obsidian, gold, blood, glass, jeweled, frost, phasing, electric**. Synchrony additionally defines **onyx**. Shape/material inclusion and exclusion rules exist: do not multiply every shape by every material to invent a weapon list.",
        "",
        "The sampled ordinary dagger/spear/broadsword/longsword/whip/rapier/bow/crossbow/flail/cat families include the basic six material forms. Dagger also has shard, jeweled, frost, and phasing definitions, plus separate Dad/tutorial identities. The exact registered names and IDs appear below. Materials can alter combat effects even when their input sequence remains the same; exact damage/effects are not inferred merely from the material name.",
        "",
        "## Other state-dependent controls",
        "",
        "| Mechanic | Examples / inputs | State the interface needs | Current coverage or question | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| {m} | {e}; **{i}** | {s} | {g} | {v} |" for m, e, i, s, g, v in MECHANICS]
    lines += [
        "",
        "Current action-slot item names: "
        + ", ".join(r["name"] for r in payload["current_item_slots"] if r["slot"] == "action")
        + ".",
        "",
        "Current spell-slot item names: "
        + ", ".join(r["name"] for r in payload["current_item_slots"] if r["slot"] == "spell")
        + ".",
        "",
        "Inventory currently exposes type, type ID, quantity, weapon damage, turn cooldown, kill cooldown, a derived usability bit, and itemToggleable state. It does not expose every state listed above. In particular, itemActivable armed state and weaponReloadable ammo are distinct from the already encoded toggle/cooldown fields.",
        "",
        "## Interface design implications",
        "",
        "Use the game's resolved action and equipped-item capabilities, together with mutable state, to determine availability. Do not equate THROW with a particular weapon or BOMB with explosives. Preserve the existing checkpoint contract while validating any replacement as a versioned observation/action interface.",
        "",
        "The first useful validation set is: dagger arm/release; spear arm/release; crossbow empty/full reload; rifle partial reload; a toggle boot; holster storage/swap; and a spell with blood-magic eligibility. Character remapping belongs to a separate qualification if training expands beyond Bard. Do not add positive rewards merely for activating, reloading, or toggling.",
        "",
        "## Exact registered weapon-item inventory",
        "",
        "This is the complete set of 76 weapon-bearing item prototypes returned by this capture, not a list of guaranteed spawnable loot. Empty pool markers and item bans are retained in the JSON for later eligibility auditing. Ammo columns describe prototype defaults, not an equipped live weapon.",
        "",
        "| Entity ID | Display name | Family | Capabilities found |",
        "| --- | --- | --- | --- |",
    ]
    for w in weapons:
        flags = []
        if w["throwable"]:
            flags.append("throwable")
        if w["activable"]:
            flags.append("activable")
        if w["reload"]:
            a = w["reload"]
            flags.append(
                f"ammo {a.get('ammo')}/{a.get('maximumAmmo')}, reload +{a.get('ammoPerReload')}"
            )
        lines.append(
            f"| `{w['id']}` | {w['name']} | {w['family']} | {', '.join(flags) or 'no throw/reload component recorded'} |"
        )
    lines += [
        "",
        "## Evidence and limits",
        "",
        "The collector only reset one private worker and read prototypes; it took no gameplay actions and trained no policy. The first attempt used an unavailable inspection API and was discarded. The successful probe's log needed an escaped-quote parser correction for Miner's Cap; all 263 records were then recovered and checked for unique IDs. No failed or missing record was silently omitted.",
        "",
        "Core/version/DLC archives were read locally. Component presence is stronger evidence than a matching word in bytecode, but prototype defaults still do not establish all runtime rules. Supplemental DLC control descriptions come from installed module definitions. Their paths and evidence hashes are in the JSON. Only dagger behavior is marked live-verified; see [THROW diagnostic](throw-semantics-diagnostic.md).",
        "",
        "The catalog covers core weapon families, explicit DLC/character additions found in the installed build, exact registered weapon items, and the control categories relevant to this interface. It does not claim exhaustive normal-loot eligibility, every editor/PvP custom combination, or behavior qualification for every item.",
        "",
    ]
    (ROOT / "docs/game-control-catalog.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "core_families": len(families),
                "registered_weapon_items": len(weapons),
                "mechanic_categories": len(MECHANICS),
            }
        )
    )


if __name__ == "__main__":
    main()
