# Weapon and control-mechanics catalog

Installed build: **v4.2.1-b5713 / Steam 22938426**. Captured 2026-09-08.

This catalog separates installed definitions, prototypes registered under the current training configuration, and behavior verified live. Registration does not establish that an item can appear as normal Bard loot. The production mod blacklists Amplified, DynChar, and Synchrony; some other DLC/special prototypes still register. No gameplay packages were enabled for this catalog.

The private read-only probe enumerated **250 item prototypes and 13 playable-character prototypes**. Of the items, **76 carry a weapon component**; that includes character, tutorial, and special-purpose items. An additional weapon-bearing character prototype (Eli) is excluded from the item count. Only the starting dagger has a live input-sequence qualification.

Machine-readable companion: [game-control-catalog.json](game-control-catalog.json). Full local component evidence: `runs/game-control-catalog-current/active-prototypes.json`.

## Core weapon families

These **19 shape definitions** occur in the installed core archives. A zero in the third column means no matching prototype was registered in this capture, not that the weapon does not exist in the game.

| Family | Control considerations | Registered item variants |
| --- | --- | ---: |
| Axe | Installed shape definition; not registered in the sampled training configuration. | 0 |
| Blunderbuss | Reloadable; sampled prototype starts with 1/1 ammo and reloads 1 per activation. | 1 |
| Bow | Directional weapon attack; no throw/reload component on sampled variants. | 6 |
| Broadsword | Directional weapon attack; no throw/reload component on sampled variants. | 6 |
| Cat O Nine Tails | Cat O Nine Tails; directional attack/movement pattern needs its own action interpretation. | 6 |
| Crossbow | Reloadable; sampled base prototype starts with 3/3 ammo and reloads 3 per activation. | 6 |
| Cutlass | Installed shape definition; not registered in the sampled training configuration. | 0 |
| Dagger | THROW arms; direction releases. Starting dagger verified live; other variants not tested. | 12 |
| Flail | Directional weapon attack; inspect knockback and target pattern separately from damage. | 6 |
| Flower | Special weapon definition; treat character rules separately from ordinary damaging weapons. | 1 |
| Golden Lute | Special weapon definition; character-specific directional/combat rules need separate validation. | 0 |
| Harp | Installed shape definition; not registered in the sampled training configuration. | 0 |
| Longsword | Directional weapon attack; no throw/reload component on sampled variants. | 6 |
| Rapier | Directional attack with movement-related pattern behavior; do not assume a direction means one-tile walking. | 6 |
| Rifle | Reloadable; sampled prototype starts with 0/3 ammo and reloads 1 per activation. | 1 |
| Spear | Throwable component; armed/directional semantics need a spear live check. | 6 |
| Staff | Installed shape definition; not registered in the sampled training configuration. | 0 |
| Warhammer | Installed shape definition; not registered in the sampled training configuration. | 0 |
| Whip | Directional weapon attack; no throw/reload component on sampled variants. | 6 |

Source directory: `scripts/necro/game/data/item/weapon/shape/`, with effective patch overrides applied. The JSON records the source archive for each family.

## Additional installed DLC and character weapons

These are definition-backed entries; their controls were not exercised in this catalog run.

| Weapon | Scope | Control considerations |
| --- | --- | --- |
| Trident | Synchrony | Separate shape and attack behavior; validate special-action capabilities. |
| Plasma Cannon | Synchrony | Reloadable; source sets ammo 0, capacity 3, reload increment 1; attack depends on ammo. |
| Lantern | Synchrony / possession | Source explicitly disables weaponThrowable and attaches possession-on-kill behavior. |
| Zweihänder | Klarinetta | Rotating sword; selected control mode can remap ITEM_2 and THROW to rotation. |
| Lance | Suzu | Source describes dash on kill; character charge/movement state matters. |
| Spring Onion | Hatsune Miku | Character-specific weapon; base and phasing prototypes appeared in the sample. |
| Electric / Phasing Spring Onion | Hatsune Miku | Explicit enchantment variants; electric variant absent from sampled prototypes. |
| Shovel Blade | Shovel Knight | Hybrid character/shovel combat mechanism; not an ordinary weapon-slot-only abstraction. |

Spring Onion has separately declared base, electric, and phasing forms. The Shovel Blade is listed because it changes combat controls even though a weapon-slot-only search can miss it. Special registered items such as Mallet Of The Dad, Clonking Stick, Squishy Piston, and the two wands are listed in the exact inventory below; they are not asserted to be ordinary player loot.

## Materials and variants

Core material definitions: **base, titanium, obsidian, gold, blood, glass, jeweled, frost, phasing, electric**. Synchrony additionally defines **onyx**. Shape/material inclusion and exclusion rules exist: do not multiply every shape by every material to invent a weapon list.

The sampled ordinary dagger/spear/broadsword/longsword/whip/rapier/bow/crossbow/flail/cat families include the basic six material forms. Dagger also has shard, jeweled, frost, and phasing definitions, plus separate Dad/tutorial identities. The exact registered names and IDs appear below. Materials can alter combat effects even when their input sequence remains the same; exact damage/effects are not inferred merely from the material name.

## Other state-dependent controls

| Mechanic | Examples / inputs | State the interface needs | Current coverage or question | Evidence |
| --- | --- | --- | --- | --- |
| Armed throws | Dagger, spear; **THROW then direction** | Throwable capability; armed state; direction | Inventory lacks itemActivable.active; redundant THROW remains allowed | Dagger live; spear definition |
| Reloading and charge | Crossbow, rifle, blunderbuss, plasma cannon; **THROW / directional attack** | Ammo, capacity, reload increment, loaded attack | Inventory does not expose weaponReloadable ammo | Prototypes + installed code; no live gun test |
| Shared special-action resolution | Weapons, toggle boots, convertible items; **THROW** | Which equipped item the game resolves for the action | Mask checks weapon presence, rather than the resolved action item | Installed ActionItem/MovementItem/Weapon code |
| Toggle movement equipment | Boots of Leaping, Boots of Lunging; **Shared activation; directions** | Toggle state, distance, hazards, activation routing | Toggle bit exists, but availability and movement assumptions need validation | Current prototypes + MovementItem code |
| Weapon storage / swap | Holster; bag/backpack behavior; **ITEM_2 in installed action-item handlers** | Container contents, slot ownership, swap state | Flat inventory lacks stored contents; verify which holder owns the item action | Holster prototype + ActionItem code; bag behavior not live-tested |
| Item conversion | Items carrying itemConvertible; **THROW in conversion handler** | Current form, target form, eligibility | Weapon-presence mask is not a general conversion-availability test | Installed ActionItem code; no sampled convertible item asserted |
| Consumable slots | Food, holy water, scrolls, heart transplants; **ITEM_1 / ITEM_2** | Item identity, charges, consumption, target rules | Quantity is present; usability mostly means cooldown counters are zero | Current item prototypes |
| Combo and health-cost items | War Drum, Blood Drum; **Item activation then follow-up actions** | Combo progress, next-attack modifier, health cost | Neither drum combo nor activation health cost is explicitly encoded | Current prototypes |
| Spell recharge / blood magic | Fireball, freeze, heal, bomb, shield, transmute, charm; **SPELL_1 / SPELL_2** | Kill/time recharge, blood-cast eligibility and health cost | Cooldowns exist; mask may exclude valid blood casts and needs a live eligibility test | Current spell-item prototypes |
| Directional or delayed spells | Fireball; installed dash and berserk abilities; **Spell input plus facing/direction or later actions** | Aim/facing, pending cast, charge/duration | No general pending-action state; exact controls require per-ability validation | Installed spell definitions; behavior not live-tested |
| Bomb-slot substitution | Bomb stacks, infinite bombs; Miku Sing; **BOMB** | Resolved slot action, quantity, effect and timing | Current mask tests bomb-slot occupancy; occupied slot need not mean an explosive | Current prototypes; Miku Sing occupies bomb slot |
| Character-specific remapping | Klarinetta, Suzu, Chaunter, Miku, Shovel Knight; **Directions and shared special buttons** | Character, selected control mode, rotation, possession, charge, pogo/combo state | Current environment trains Bard; its 11-action contract is not certified for these characters | Installed character scripts |
| Rhythm / time control | Bard; rhythm-based characters; heart transplant; **Every turn / timed item use** | Beat phase, missed-beat rules, temporary timing changes | Bard experiments do not establish correctness for rhythm-based play | Bridge character restriction + installed definitions |
| Contextual movement | Digging, attacks, ice/sliding, confusion, knockback, stairs; **Directions** | Target cell, status effects, weapon mode, movement modifier | Some grid/status fields exist; navigation masking must respect action context | Existing observation/contract/outcome code |
| Pickup and transaction effects | Shops, shrines, dropped gear, transmutation; **Movement/interactions** | Price, health/currency, inventory replacement and item bans | Some prices/flags exist; action eligibility cannot be inferred from occupancy alone | Existing rich observation and installed item rules |
| Passive automatic effects | Potion, frost protection, movement-triggered equipment; **No dedicated button / triggered by another action** | Remaining item, trigger condition, effect duration | Do not invent an active action for a passive item | Current prototypes; exact triggers need dedicated tests |

Current action-slot item names: Apple, Cheese, Drumstick, Ham, Holy Water, Lord Crown, War Drum, Blood Drum, Double Heart Transplant, Heart Transplant, Earthquake Scroll, Fear Scroll, Fireball Scroll, Freeze Enemies Scroll, Gigantism Scroll, Riches Scroll, Shield Scroll, Enchant Weapon Scroll, Scroll Of Need, Transmute Scroll, Enchant Leek Scroll, Descend Scroll.

Current spell-slot item names: Fireball Spell, Freeze Enemies Spell, Heal Spell, Bomb Spell, Shield Spell, Transmute Spell, Charm Spell.

Inventory currently exposes type, type ID, quantity, weapon damage, turn cooldown, kill cooldown, a derived usability bit, and itemToggleable state. It does not expose every state listed above. In particular, itemActivable armed state and weaponReloadable ammo are distinct from the already encoded toggle/cooldown fields.

## Interface design implications

Use the game's resolved action and equipped-item capabilities, together with mutable state, to determine availability. Do not equate THROW with a particular weapon or BOMB with explosives. Preserve the existing checkpoint contract while validating any replacement as a versioned observation/action interface.

The first useful validation set is: dagger arm/release; spear arm/release; crossbow empty/full reload; rifle partial reload; a toggle boot; holster storage/swap; and a spell with blood-magic eligibility. Character remapping belongs to a separate qualification if training expands beyond Bard. Do not add positive rewards merely for activating, reloading, or toggling.

## Exact registered weapon-item inventory

This is the complete set of 76 weapon-bearing item prototypes returned by this capture, not a list of guaranteed spawnable loot. Empty pool markers and item bans are retained in the JSON for later eligibility auditing. Ammo columns describe prototype defaults, not an equipped live weapon.

| Entity ID | Display name | Family | Capabilities found |
| --- | --- | --- | --- |
| `WeaponFlower` | Flower | Flower | no throw/reload component recorded |
| `WeaponDagger` | Dagger | Dagger | throwable, activable |
| `WeaponDaggerShard` | Glass Shard | Dagger | throwable, activable |
| `WeaponTitaniumDagger` | Titanium Dagger | Dagger | throwable, activable |
| `WeaponObsidianDagger` | Obsidian Dagger | Dagger | throwable, activable |
| `WeaponGoldenDagger` | Golden Dagger | Dagger | throwable, activable |
| `WeaponBloodDagger` | Blood Dagger | Dagger | throwable, activable |
| `WeaponGlassDagger` | Glass Dagger | Dagger | throwable, activable |
| `WeaponDaggerJeweled` | Jeweled Dagger | Dagger | throwable, activable |
| `WeaponDaggerFrost` | Dagger Of Frost | Dagger | throwable, activable |
| `WeaponDaggerPhasing` | Dagger Of Phasing | Dagger | throwable, activable |
| `WeaponBroadsword` | Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponTitaniumBroadsword` | Titanium Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponObsidianBroadsword` | Obsidian Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponGoldenBroadsword` | Golden Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponBloodBroadsword` | Blood Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponGlassBroadsword` | Glass Broadsword | Broadsword | no throw/reload component recorded |
| `WeaponLongsword` | Longsword | Longsword | no throw/reload component recorded |
| `WeaponTitaniumLongsword` | Titanium Longsword | Longsword | no throw/reload component recorded |
| `WeaponObsidianLongsword` | Obsidian Longsword | Longsword | no throw/reload component recorded |
| `WeaponGoldenLongsword` | Golden Longsword | Longsword | no throw/reload component recorded |
| `WeaponBloodLongsword` | Blood Longsword | Longsword | no throw/reload component recorded |
| `WeaponGlassLongsword` | Glass Longsword | Longsword | no throw/reload component recorded |
| `WeaponWhip` | Whip | Whip | no throw/reload component recorded |
| `WeaponTitaniumWhip` | Titanium Whip | Whip | no throw/reload component recorded |
| `WeaponObsidianWhip` | Obsidian Whip | Whip | no throw/reload component recorded |
| `WeaponGoldenWhip` | Golden Whip | Whip | no throw/reload component recorded |
| `WeaponBloodWhip` | Blood Whip | Whip | no throw/reload component recorded |
| `WeaponGlassWhip` | Glass Whip | Whip | no throw/reload component recorded |
| `WeaponSpear` | Spear | Spear | throwable, activable |
| `WeaponTitaniumSpear` | Titanium Spear | Spear | throwable, activable |
| `WeaponObsidianSpear` | Obsidian Spear | Spear | throwable, activable |
| `WeaponGoldenSpear` | Golden Spear | Spear | throwable, activable |
| `WeaponBloodSpear` | Blood Spear | Spear | throwable, activable |
| `WeaponGlassSpear` | Glass Spear | Spear | throwable, activable |
| `WeaponRapier` | Rapier | Rapier | no throw/reload component recorded |
| `WeaponTitaniumRapier` | Titanium Rapier | Rapier | no throw/reload component recorded |
| `WeaponObsidianRapier` | Obsidian Rapier | Rapier | no throw/reload component recorded |
| `WeaponGoldenRapier` | Golden Rapier | Rapier | no throw/reload component recorded |
| `WeaponBloodRapier` | Blood Rapier | Rapier | no throw/reload component recorded |
| `WeaponGlassRapier` | Glass Rapier | Rapier | no throw/reload component recorded |
| `WeaponBow` | Bow | Bow | no throw/reload component recorded |
| `WeaponTitaniumBow` | Titanium Bow | Bow | no throw/reload component recorded |
| `WeaponObsidianBow` | Obsidian Bow | Bow | no throw/reload component recorded |
| `WeaponGoldenBow` | Golden Bow | Bow | no throw/reload component recorded |
| `WeaponBloodBow` | Blood Bow | Bow | no throw/reload component recorded |
| `WeaponGlassBow` | Glass Bow | Bow | no throw/reload component recorded |
| `WeaponCrossbow` | Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponTitaniumCrossbow` | Titanium Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponObsidianCrossbow` | Obsidian Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponGoldenCrossbow` | Golden Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponBloodCrossbow` | Blood Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponGlassCrossbow` | Glass Crossbow | Crossbow | ammo 3/3, reload +3 |
| `WeaponFlail` | Flail | Flail | no throw/reload component recorded |
| `WeaponTitaniumFlail` | Titanium Flail | Flail | no throw/reload component recorded |
| `WeaponObsidianFlail` | Obsidian Flail | Flail | no throw/reload component recorded |
| `WeaponGoldenFlail` | Golden Flail | Flail | no throw/reload component recorded |
| `WeaponBloodFlail` | Blood Flail | Flail | no throw/reload component recorded |
| `WeaponGlassFlail` | Glass Flail | Flail | no throw/reload component recorded |
| `WeaponCat` | Cat O Nine Tails | Cat | no throw/reload component recorded |
| `WeaponTitaniumCat` | Titanium Cat | Cat | no throw/reload component recorded |
| `WeaponObsidianCat` | Obsidian Cat | Cat | no throw/reload component recorded |
| `WeaponGoldenCat` | Golden Cat | Cat | no throw/reload component recorded |
| `WeaponBloodCat` | Blood Cat | Cat | no throw/reload component recorded |
| `WeaponGlassCat` | Glass Cat | Cat | no throw/reload component recorded |
| `WeaponBlunderbuss` | Blunderbuss | Blunderbuss | ammo 1/1, reload +1 |
| `WeaponRifle` | Rifle | Rifle | ammo 0/3, reload +1 |
| `Coldsteel_WeaponLeek` | Spring Onion | special | no throw/reload component recorded |
| `Coldsteel_WeaponLeekPhasing` | Phasing Spring Onion | special | no throw/reload component recorded |
| `BellHammer` | Mallet Of The Dad | special | ammo 0/1, reload +1 |
| `Club` | Clonking Stick | special | ammo 0/1, reload +1 |
| `Piston` | Squishy Piston | special | no throw/reload component recorded |
| `WandConfusion` | Wand Of Inversion | special | no throw/reload component recorded |
| `WandWind` | Wand Of Attraction | special | no throw/reload component recorded |
| `WeaponDaggerDad` | Dagger | Dagger | throwable, activable |
| `WeaponDaggerTutorial` | Dagger | Dagger | no throw/reload component recorded |

## Evidence and limits

The collector only reset one private worker and read prototypes; it took no gameplay actions and trained no policy. The first attempt used an unavailable inspection API and was discarded. The successful probe's log needed an escaped-quote parser correction for Miner's Cap; all 263 records were then recovered and checked for unique IDs. No failed or missing record was silently omitted.

Core/version/DLC archives were read locally. Component presence is stronger evidence than a matching word in bytecode, but prototype defaults still do not establish all runtime rules. Supplemental DLC control descriptions come from installed module definitions. Their paths and evidence hashes are in the JSON. Only dagger behavior is marked live-verified; see [THROW diagnostic](throw-semantics-diagnostic.md).

The catalog covers core weapon families, explicit DLC/character additions found in the installed build, exact registered weapon items, and the control categories relevant to this interface. It does not claim exhaustive normal-loot eligibility, every editor/PvP custom combination, or behavior qualification for every item.
