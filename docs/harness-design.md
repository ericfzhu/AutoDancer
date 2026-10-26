# Bard harness design

Status: equipment and world-observation integration for the pinned solo Bard
configuration, updated 2026-10-26. Equipment uses snapshot v3 and the 250-type
registry. World snapshot v1 adds separate entities and exact visible terrain
through architecture 10; architecture 9 retains its equipment-only contract.
Collection now stops on protocol, transport, representation failures and game
aborts. The information-policy and task/reward audit is recorded below.
Qualification is summarized below. Training remains paused.

The harness should expose the decisions available to Bard and the information
needed to understand their consequences. Game rules stay in the game. The
policy learns which decisions are useful. Correct controls alone do not prove
that the observation is sufficient or that learning will succeed.

## Scope and completion definition

The first implementation target is ordinary solo Bard runs under the project's
current content configuration. Enumerate the actual enabled packages, unlocks,
mode and item restrictions rather than treating installed prototypes as loot.
The same design should accommodate additional Bard-accessible DLC equipment,
but other characters, custom modes and mods are separate support profiles.

"Bard equipment supported" means every reachable item in a declared content
profile has an identity, acquisition rules, observable state, action semantics,
interaction coverage and a declared qualification status. Counting definitions
is insufficient. Include items obtained from shops, transformations, shrines,
containers and drops, not just ordinary floor generation. A passive item's
effects still matter even when it introduces no new button.

The contracts below define the implementation target. The user authorized
focused live harness checks; training experiments remain paused. Qualification
of additional mechanics must retain explicit scope and evidence.

## Existing implementation and gaps

| Area | Current implementation | Required change |
| --- | --- | --- |
| Native transport | Session/run/command identities and acknowledgement checks in `live/protocol.py` and `envs/live.py` | Preserve; extend payload contracts rather than replace transport |
| Equipment scope | Exact-name dagger/basic-shovel/bombs allowlist in Lua | Replace with a versioned supported-content manifest after qualification |
| Inventory | Thirteen slots, eight features, first entity in each slot | Represent all owned items and container relationships; distinguish missing state from zero |
| Type identity | Names hashed into 4,095 nonzero IDs | Collision-free catalog IDs in the new representation; retain old hashes only for legacy checkpoints |
| Control state | Dagger-specific three-value telemetry and Python projection | General capability, pending-action and resolved-control records |
| Special masks | Mostly occupancy/readiness checks in Lua | Resolve the game's actual action handler, including shared buttons and conditional costs |
| Observation | Local grid, remembered terrain, player/inventory/context | Add equipment state and task context; audit co-located entities and world interactions |
| Outcomes | Mostly inferred from grid differences and events | Keep authoritative events and multi-effect action attribution; preserve uncertainty |
| Tasks | Fixed start/target/profile; `PlayerFeature.TASK` actually means boss identity | Explicit objective context, reset provenance and terminal semantics |

This table records the original baseline. The October implementation sections
below supersede its inventory, identity, control and telemetry descriptions;
the expanded data now has an opt-in actor/critic projection. The default legacy
policy still uses the original inventory tensors. The new equipment profiles and qualification are described below. This work
does not claim that all non-equipment gameplay or task-design work is complete.

Sources: `mods/AutoDancer/scripts/AutoDancer.lua`, `src/autodancer/constants.py`,
`src/autodancer/observation.py`, `src/autodancer/memory.py`,
`src/autodancer/training/action_contract.py`, `src/autodancer/outcomes.py`.
The earlier [equipment catalog](game-control-catalog.md) is discovery evidence;
its gap descriptions reflect its capture date, not current qualification.

## Responsibility boundaries

The transition path is game command → settled game outcome → validated telemetry
→ policy observation → task/reward accounting → learner sample.

1. **Lua bridge and exporter:** execute one requested input through the game's
   normal handler; acknowledge it once; export a coherent post-action snapshot
   and the effects that occurred. Export facts rather than action preferences.
2. **Python protocol decoder:** verify identity, sequence, schema, shapes, ranges,
   references and required capabilities. Never repair invalid telemetry into a
   plausible state. Keep transport/debug fields outside policy inputs.
3. **Observation adapter:** project permitted information into versioned tensors,
   retain explicit map/history knowledge, and apply the declared control
   contract. It must not consult hidden debug data.
4. **Task evaluator:** determine success/failure/truncation from a declared task;
   it does not invent movement advice or reward unsupported equipment as failure.
5. **Reward tracker:** score the transition under a separately versioned reward
   specification, with individually reported components.
6. **Collector:** store the exact projected observation, mask, action, recurrent
   context, reward and boundary used for learning. Reconstructing a mask later
   from a different equipment snapshot is forbidden.

Training, evaluation and diagnostics must use the same decoder and adapter.
Diagnostics may inspect more information, but those fields cannot enter the
actor, critic, masks or shaping accidentally.

## Equipment registry and inventory

Use a registry keyed by full engine type name and content-profile identity.
At build/export time allocate deterministic, collision-free integer IDs and
persist the mapping and its digest. Zero means absent; unknown is a separate
status. IDs must not be regenerated mid-run. Checkpoints bind to the exact map;
an expanded registry needs an explicit migration, not silent ID reassignment.

Each registry entry records:

- exact name, package/source, slot compatibility and known acquisition rules;
- weapon family/material and mechanics capabilities, derived from components;
- state fields needed by those capabilities, units and observability evidence;
- supported interactions and qualifications, with evidence paths/hashes;
- eligibility as `eligible`, `excluded` or `unresolved`, with a reason.

Registered, obtainable and behavior-qualified are three separate facts.
Unresolved eligibility prevents an exhaustive Bard item count.

Represent inventory as item instances plus slot/container relations. An instance
has type ID, occupied slot or parent container, quantity, active form, capability
flags, state values and per-field known/applicable flags. Instance IDs are for
transport/event association; arbitrary process IDs are not policy features.
Validate unique ownership, existing references and absence of containment cycles.
Stored weapons and swapped items must retain their state. Empty slots, unknown
contents, unavailable fields and a known numeric zero must remain distinguishable.

Use padded item tokens and validity masks in policy tensors. Derive capacity
from supported inventory/container rules during static inspection; do not pick
a convenient size and silently drop extra entries. Exceeding capacity is a
representation failure. Static item descriptions can be cached by registry ID;
only changing instance state needs per-action transport.

## Mechanics coverage

| Capability family | State required when applicable | Important interactions |
| --- | --- | --- |
| Directional weapons | Family, material/effects, effective damage, facing-dependent rules | Multiple targets, reach, knockback, armor, shields, move-on-attack |
| Throws | Armed/pending mode, owning item, permitted continuation inputs | Empty hand, dropped weapon, retrieval, weapon swap, shared activation |
| Reloadable weapons | Ammo, capacity, reload amount and phase | Partial reload, empty fire, throw/toggle button resolution |
| Digging | Dig strength, restrictions, durability/conditional effects | Wall types, equipment loss, combat/dig interaction |
| Passive equipment | Relevant visible modifiers and status changes | Damage prevention, breakage, healing, immunity, visibility, movement |
| Toggle movement gear | Active state, move pattern, activation owner | Weapon activation, blocked landing, hazards, attack versus movement |
| Consumables | Quantity, ownership, readiness and known cost | Consumption, target/direction, delayed effects, slot priority |
| Spells | Charges/cooldown, normal/blood-cast availability and cost | Facing, pending cast, health cost, recharge and overlapping effects |
| Drums/combo items | Combo stage, visible buff duration and activation cost | Next attack, interrupted combo, health changes |
| Bags/holsters | Contents, selected item, swap action and state retention | Multiple action owners, pickup when full, drops and replacement |
| Conversion/breakage | Current form, known conversion options and conditions | Shared controls, glass breaking, inventory and registry updates |

These are required representation categories, not assertions that every field
is currently available from a safe read-only game API. Inspect handler ordering
and field visibility before implementing each family. Shared-button cases need
interaction tests; testing each item in isolation cannot establish correctness.

## Action semantics and masks

Retain the current eleven input codes for the initial design. The existing
`THROW` code is better described to users as **special activation**: throwing,
reloading or another effect depends on the game's handlers and current gear.
Do not renumber historical codes. A new input is justified only if a legitimate
Bard decision cannot be expressed through the existing inputs.

For each input, describe current control state: action code, verified owner
where known, availability, known cost, current pending mode and mask reason.
Do not require the harness to predict the action's complete consequences or
next pending mode. Shared handlers may have conditional or multiple effects;
an unresolved owner is not evidence that no item will handle the input. Record
actual effects after acknowledgement, with evidence and inference labels.
Prefer pure queries of engine rules; where unavailable, use a small version-bound
adapter backed by inspected handler code. Never invoke a mutating handler to
ask if it is legal. No duplicate combat simulator or counterfactual action rollout
belongs in this interface.

One step is one requested input to the next acknowledged decision boundary.
It may move, attack, dig, interact or do several of these. It is not necessarily
one tile or one simulated beat. Record turn advancement and multi-effect events.
Do not automatically fire after aiming, reload to completion, pick a direction,
or perform several actions to finish an item interaction.

**Timing decision:** retain discounting per learner decision for the initial
implementation. Each accepted input has discount gamma, even if it advances
zero or multiple game turns. Export game turns advanced separately where a
reliable engine counter is available. Bridge sequence numbers and the existing
`PlayerFeature.TURN` export are not proof of engine turn advancement; the current
Lua field is populated from `sequence`. Unknown duration stays unknown.

A later game-turn objective would use gamma raised to measured duration, with
consistent accumulated rewards, potential shaping, bootstrapping and trace-decay
rules. Merely changing gamma in PPO is insufficient. Zero-turn decision loops
also need explicit handling under that objective. True termination still forces
zero bootstrap; discount factors do not override termination. No game-turn
objective is wired into the learner by this design change.

Separate three meanings that the current word "mask" can hide:

- **Executable:** the interface can dispatch the input in this state.
- **Semantically redundant:** verified equivalent to another retained input,
  including time advancement, costs and pending state.
- **Recommended:** strategically useful, safe or closer to a goal.

The trial-and-error mask may exclude non-executable inputs and explicitly
qualified redundancies. Recommendation never affects it. Wall attempts, waiting
and risky moves remain available. Failed item activation may consume a turn or
provide a wait, so empty-slot actions are not automatically invalid. Retain a
qualified wait action and document any equivalence used to suppress alternatives.

Unknown availability must not be silently masked as illegal. If the input can
be safely dispatched under a declared permissive contract, expose unknownness;
otherwise flag unsupported semantics and stop collection. Do not derive masks
from unseen enemy positions, undiscovered traps or hidden eligibility state.
All-masked nonterminal states are contract failures, never automatic WAITs.

## Observation sufficiency and information policy

Every new field needs a source, units/range, update timing, missing-value rule,
and an information classification. Human-equivalent information is an objective
choice, not a requirement of reinforcement learning. The original design
distinguishes these two information policies:

- `player`: static game knowledge, current visible cues and remembered player
  history. This preserves the existing intended information boundary.
- `privileged`: additionally permits declared internal state such as exact
  internal equipment counters or unseen current entities. This is a different
  learning environment and must be named in results and checkpoint metadata.

Neither profile includes future RNG, arbitrary engine IDs or transport/debug
internals as policy features. Both actor and critic use the same profile in the
initial implementation; a privileged critic would be a separate explicit design.
The implemented `internal-v1` and `player-v1` profiles do not map directly to
these two policies: both retain declared engine information. `player-v1` names
the HUD-filtering projection, not a whole-game human-equivalence certificate.
See the completed audit below. The exporter may supply internal diagnostics
separately, but excluded fields must not leak into masks or inputs. Separate static game knowledge,
currently visible state, remembered observations and diagnostic-only internals.
In the `player` profile, unseen enemies and unrevealed terrain remain excluded.
Exact cooldowns or internal charge state are not automatically permissible just
because Lua can read them; use visible cues or reproducible player history, or
classify the field as internal for the explicitly selected `privileged` profile.

Add named equipment-state tokens and pending-action context. Do not reuse an
unrelated inventory column as a permanent general-purpose solution. Continue
to expose the local geometry and persistent revealed map. Preserve explicit
validity and age for remembered facts; do not present stale dynamics as visible.

Audit multiple entities on one tile: actor, dropped weapon, price tag, trap and
explosion can coexist. Preserve them through a bounded entity/object stream or
document a lossless aggregation for the supported mechanics. Existing coarse
`OTHER` classes and a single per-cell type cannot be assumed sufficient.
Likewise inspect terrain effects, exits/locks, shrines, shops, pickups, status
ailments, enemy phases and turn timing; equipment is only one part of the harness.

Add task context separately from boss identity: objective kind, target level,
scope and, only for a true finite-horizon objective, remaining actions. A
collection time limit must not become an implicit survival objective. Fixed
single-task runs can retain the old unconditioned interface during migration.

## Transitions, rewards and episode boundaries

Associate every event with its causing command and affected instance. Preserve
damage source and target, item consumption, reload, mode changes, pickup/swap,
breakage, level transitions and turn advancement when authoritative evidence
exists. Mark inferred outcomes as inferred; allow multiple outcomes per action.
No displacement does not prove that an action was useless.

Success and game death terminate the declared task. Collection limits truncate
and bootstrap from the final pre-reset observation. True deadlines terminate
and require remaining-time context. Infrastructure errors and unsupported
semantics stop the run with an explicit invalid/incomplete result. They are not
negative-reward episodes, successful truncations or silently resampled seeds.
An unfinished fragment is retained diagnostically and not submitted as if valid;
earlier updates remain recorded and the run cannot be reported as completed.

Reward definitions remain independent of the equipment adapter. Do not reward
using each new capability merely to make it appear learnable. Preserve extrinsic
completion metrics separately from potential shaping and exploration bonuses.
Potential discounts must match the learner; task termination and truncation must
be handled consistently. Shaping sources and reward feedback must be declared
separately from actor inputs. The implemented symbolic task permits privileged
reward events; A10 excludes previous rewards from both actor and critic inputs.

## Compatibility, cost and failure handling

Version transport schema, catalog, policy observation, action contract, task and
reward specifications separately; bind their digests into run/checkpoint metadata.
A new equipment observation is not an exact resume of the old model, even if
some tensor dimensions match. Keep the historical bounded contract available;
use explicit model migration/warm-start rules for the new interface.

The production path should use cached static definitions, bounded dynamic
records and shared decoding, without full prototype enumeration every turn.
Preserve the existing native transport and worker concurrency. Measure payload
and projection cost later, when experiments resume; no speedup is assumed here.

Unknown items require a manifest update and a qualification decision. Do not
automatically accept them based on a matching component, suppress their pickups,
or reset until a supported run appears. If a restricted loot mode is introduced,
name it as a different environment and evaluate transfer separately.

## Implementation sequence and acceptance criteria

| Stage | Deliverable | Acceptance before proceeding |
| --- | --- | --- |
| 1. Content audit | Exact Bard eligibility registry for pinned enabled content | Every entry has eligibility evidence or explicit unresolved status; no invented total |
| 2. Contracts | Versioned schemas for registry, item instances, controls and task context | Unknown versus zero, handler ownership, capacities and migration rules specified |
| 3. Export/adapter | Lua read-only state extraction and common Python projection | Offline fixtures cover lossless round trips, bad references/ranges, overflow and hidden-state exclusion |
| 4. Ordinary gear | Passive equipment, digging and directional weapons | Pickups, effect changes, breakage and replacement covered; support declared per mechanic |
| 5. Stateful gear | Throws, reloads, toggles, consumables, spells and containers | Pending state and shared-action interactions covered, including conflicting loadouts |
| 6. Task integration | Shared observation/mask path across collectors and evaluation | Exact rollout inputs, terminal observations and objective conditioning preserved |
| 7. Qualification | Later bounded deterministic live mechanics checks | Evidence ties to exact source/content versions; scope and untested interactions explicit |

Offline tests check contracts and recorded fixtures; controlled live checks
qualify engine behavior separately. Equipment and world qualification are
recorded below, followed by the completed information-policy and task/reward
audit. Learning experiments remain paused.

For every mechanic, track discovery, source inspection, implementation, offline
verification and live qualification independently. Completion means the selected
content profile is covered, not that a few seeds avoided unsupported equipment.

The implementation supports the complete registered superset instead of imposing
an inferred Bard loot allowlist. Source-specific bans and unlock keys are audited
separately. A finite pickup sweep is not a proof of natural reachability. Storage
capacity is an explicit allocation budget, not a guessed game maximum; overflow
stops collection. No additional ordinary Bard input was needed in qualification.

## Implemented equipment contract

`Equipment.lua` reads engine state; the game continues to implement combat,
movement, spells, acquisition and passive effects. `equipment_state.py` validates
the payload. `equipment_observation.py` projects it into A9 inputs, and the same
checkpoint-bound contract is reconstructed for collection and evaluation.
`harness_contract.py` defines runtime information and task manifests as well as
reference types for information provenance and timing;
its conservative executable-mask helper is not the production permissive input
contract.

- **Identity and inventory:** 250 stable catalog IDs, all owned item instances,
  slot positions, holsters and ordered stash contents. Engine entity IDs are
  diagnostic join keys, never numeric policy features. Registry inclusion means
  registered, not obtainable. Integer inventory quantities accommodate infinite
  bombs without int16 overflow.
- **Mutable state:** arming, toggles, ammunition, cooldowns, combo state, temporary
  damage/dig/armor modifiers, invincibility, pickup bookkeeping, container state,
  shrine markers and the registered character-specific scalar state. Holder
  effects include regeneration progress, cursed health and temporary statuses,
  which can survive consumption of the source item.
- **Visible world:** every ground item in the existing visible 21×21 grid gets
  an exact catalog ID and relative position, including overlapping items.
  Charmed actors and active no-return floor markers are separate world records.
  Missing world evidence differs from an observed empty world.
- **Controls:** `ActionItem.getActionItem` supplies actual handler ownership.
  Ownership does not prove that an effect will succeed. `engine-input-v1` permits
  all eleven input attempts and preserves empty inputs, health-cost casts and
  failed attempts. It performs no automatic aiming, repeated reloads or strategic
  action pruning. Existing legacy contracts remain available for old checkpoints.
- **Timing:** observations are captured at the next input boundary, after the
  preceding action settles. Terminal equipment is marked stale and projects to
  empty, non-current tensors. Decision discounting remains unchanged; command
  sequence numbers are not relabeled as engine turns.

The A9 encoder combines item identity, state and container relations, holder
status, and visible equipment-world records. Actor and critic receive identical
information. Item/world ordering carries no meaning. Profile, capacity, field
order, registry digest and observation-contract version are checkpoint-bound;
incompatible checkpoints are rejected. The default item and visible-world
capacities are each 64, configurable up to 4096. Exceeding either budget raises
an error rather than dropping objects. This is a resource setting, not a promise
that arbitrary custom modes fit the default.

### Information profiles

`player-v1` queries the same equipment selectors as the HUD and the engine's
visual action-slot resolver. It exposes displayed identities and counters only
where the rendering rules establish them: activation prompts, toggles, stack
labels, cooldown labels, equipped ammo counters and injective ammunition sprite
frames. Stored identities can be displayed without exposing their hidden
counters. Other facts remain explicitly unknown, including their applicability.
Exact hidden status durations are withheld; visible active/inactive status cues
are separate from durations. Legacy inventory/control features are cleared in
this profile so they cannot bypass the equipment visibility filter. This is a
conservative symbolic equipment view, not a claim of complete pixel/audio parity
for the entire game. Recurrent state can remember earlier observations.

`internal-v1` exposes the declared current equipment and holder fields to both
heads, including hidden counters. It still does not reveal unseen ground items,
future RNG, arbitrary engine IDs or transport state. Internal diagnostics remain
available in `info` under either profile. The profiles are distinct learning
environments and are not interchangeable in checkpoints.

No training was started. When training is explicitly resumed, select architecture
9, an explicit equipment profile, and action contract `current`; the bounded
navigation masks are not the general equipment contract.

### Events and incentives

Pickup events come from successful `objectTryCollectItem` completion, preserving
the original identity before merge, consumption or conversion. Inventory deltas
remain a compatibility fallback for older telemetry. Swaps and artificial grants
are not reported as world pickups by the new event contract.

Damage events measure health reduction inside each hit, before later healing or
revival. This covers incoming attacks, environmental/equipment self-damage and
earthquakes. Overkill cannot manufacture extra damage credit. Healing later in
the same input does not erase a real damage event. New-item reward accounting
uses all carried items and stored contents, independently of HUD visibility, so
holster swaps do not create new-item credit. Existing reward coefficients and
objectives have not been tuned as part of this work.

Special actions with observed state changes are no longer described as having
"no effect" merely because Bard stayed in place. The classifier distinguishes
observed changes from strategic benefit or proof of a successful cast.

## Content and qualification evidence

The pinned build is v4.2.1-b5713 / Steam 22938426. Production blacklists Amplified,
DynChar and Synchrony gameplay packages; other registered base/extension handlers
are resolved through the engine. `resolve_bard_runtime.py --catalog-detail`
captures every item component used in the audit, all phase bans, handler identity
and priority, progression state and actual unlock keys. Material weapon variants
can share an unlock key with their base weapon; names are not used as substitutes
for those keys. Progression was disabled in this isolated training configuration.

The engine reports these Bard restrictions in the captured context:

| Operation | Blocked registered types |
| --- | --- |
| Pickup/equip/activate | Coldsteel_ScrollEnchantLeek |
| Item-pool/shrine-pool/transaction generation | Bomb, ShovelBasic, WeaponDagger, HeartTransplant, DoubleHeartTransplant, Coldsteel_ScrollEnchantLeek |
| Shrine drops | HeartTransplant, DoubleHeartTransplant, Coldsteel_ScrollEnchantLeek |
| Crate/fixed-shrine/level generation | Coldsteel_ScrollEnchantLeek |

Generation restrictions are not use restrictions. Starting equipment, fixed
sources, transformations and breakage products cannot be counted using a single
pool-membership flag. The runtime audit retains each pool's per-depth weights
and each phase's ban separately. Supporting all registered identities avoids
silently excluding rare equipment while an exact natural-loot census is absent.
No count of 250 or 249 naturally obtainable Bard equipment is claimed.

The qualification tool applies fixtures only in private mod/profile copies.
Every accepted report requires no training, zero worker restarts and unchanged
installed config and repository mod hashes over the run. Reports, host copies
and failed/discovery runs remain ignored under `runs/`. Reusable checks are in
`tools/qualify_equipment.py`; `tools/audit_equipment_evidence.py` combines accepted
reports and rejects incomplete pickup evidence or unknown applicable fields.

| Check | Coverage and evidence |
| --- | --- |
| Ordinary equipment and player view | 197 ordinary-slot types; selected special control where present, RIGHT and WAIT. `runs/equipment-completion-catalog-20261025-final/` |
| Shared controls and lifecycle | 16 cases: arming/repeated arming, reload/toggle priority, holster retention, both bags, blood casting, drums, empty inputs, death/reset/floor boundaries. `runs/equipment-completion-player-view-20261025-b/` |
| Weapon mechanics | All 17 throwable definitions in four directions and all 10 reloadables, including three rifle reloads: 78 player-view cases. `runs/equipment-completion-weapons-20261025-release/` |
| Effect assertions | 38 player-view cases covering melee reach/sweep, ranged hits, overkill, armor, glass breakage, digging, all seven spells, transmutation, recharge, two spell slots, boots, regeneration, revival, live/prototype stash contents and no-return behavior. `runs/equipment-completion-effects-20261025-release-b/` |
| Ground acquisition | All 250 registered types attempted independently: 238 successful fixture pickups and 12 expected non-walk objects. Merge and immediate consumption retain pickup events. `runs/equipment-completion-pickups-20261025-final/` |

The completion audit at `runs/equipment-completion-audit-20261025.json` combines
these 579 cases with the fresh runtime catalog in
`runs/equipment-completion-runtime-20261025-release/`. All five qualification
reports passed with zero worker restarts and unchanged source/installed files
during each run. The catalog records 159 positive item-pool candidates under the
captured progression and ban rules; this excludes other acquisition paths and
is not the count of obtainable equipment. The full automated suite passed 525
tests; the subsequent equipment/performance recheck passed 57 tests. Ruff passed.

The 12 non-walk objects observed are the Bard-banned Coldsteel scroll,
EnemyPlaceholder, FriendlyFireTeamChanger, ItemPlaceholder, ItemStash, Marv,
NoReturnModeSquare, NoReturnShrineSquare, ShrineMarkerRhythm, SoulLinkCoop,
SoulLinkDad and quirks_DummyItem. Marker/stash/follower state remains representable
when created through its actual engine path; a failed walk pickup is not a claim
that its mechanics are unsupported.

These checks qualify the equipment interface and the listed mechanic families.
They do not exhaust every combination, validate other character/mode profiles,
or establish learning performance. World integration is described below; task
conditioning and future game-turn objectives remain separate work.

## World observations and failure handling

`World.lua` captures one record per positioned game object on currently visible
tiles in the 21-by-21 local window, including intangible objects and inactive
objects that remain in the world. Held equipment stays in the equipment stream.
Multiple enemies, traps, items, chests and price tags on one tile remain separate.
The registry contains 621 registered positioned entity types and 75 tile types,
from `runs/world-runtime-20261025-final/`. These are content-registration counts,
not claims about ordinary Bard spawn pools. Unknown identities stop collection.

The declared state bindings cover 100 scalar fields, plus component markers and
derived status flags. They include health, facing, movement/AI phase counters,
charge and shield state, parry/counterattack, Ogre club attacks, dormancy,
summoning and boss counters, traps, prices, shrine activation and grab relations.
Each field distinguishes known, unknown and inapplicable. Arbitrary entity IDs
serve only as validated joins; the policy receives relative entity references.
Unseen relation targets do not disclose their identity or coordinates.

`internal-v1` receives the declared state of objects on visible tiles.
`player-v1` filters objects using sprite/display evidence and withholds internal
phase counters, exact status durations and uncertified health values. It retains
declared cues, prices and derived status flags. This conservative symbolic view
does not certify complete pixel/audio equivalence, disguise handling or every
visibility mechanic. The whole-game information audit below classifies these
inputs as privileged and explicitly rejects a human-equivalent claim. Boss-health
potential shaping is rejected with the world player profile because that profile
does not certify exact boss health; ordinary progress rewards remain available.

Only currently visible exact terrain enters `world_tiles`; absent cells are
unknown. Existing coarse revealed-map memory remains separate. The world path
clears the legacy hashed terrain/entity grid channels, preserves coarse geometry
and visibility, and counts visible enemy records rather than occupied tiles.
Action outcomes and boss-health accounting use the separate records where
applicable. No action recommendations or additional input pruning were added.

Architecture 10 combines equipment features with all world records, relation
features and exact terrain. Actor and critic receive the same representation.
Entity ordering and engine IDs are not numerical features. Capacity, field order,
registry digest and information profile bind to the checkpoint; collection,
evaluation and diagnostics reconstruct the same environment. Default world
capacity is 256, configurable from 1 to 4096. This is an allocation budget, not a
game maximum; exceeding it raises a protocol error without truncating entities.
Existing architecture 9 checkpoints retain their original shapes and semantics.
When explicitly resuming learning, the new path requires `--architecture 10`,
`--action-contract current` and an explicit `--equipment-profile`. No learning
run or optimizer update was performed for this work.

The failure policy is `stop-run-v1`. Vector reset/step, asynchronous collection,
evaluation (including prefix acquisition) and throughput measurement do not
restart workers, retry fragments or substitute seeds after protocol,
transport, timeout or map-capacity failures. A failed vector/collector becomes
invalid and rejects reuse. In-flight requests finish or fail before the inference
service closes; the incomplete batch is never returned to PPO. Diagnostics retain
the failed worker, seed/run/sequence, last observation and unfinished fragment.
The supervisor writes a diagnostic bundle without replacing the process. The
training entry point records invalid completion status in `run-status.json`,
retaining earlier checkpoints and their actual completed update counts.
Evaluation faults do not become completed episodes; throughput faults do not
produce a successful benchmark with failed steps discarded.

Qualification uses private hosts and fixed actions:

| Check | Evidence |
| --- | --- |
| Equipment interactions with world capture | 38 cases passed, zero restarts. `runs/world-equipment-effects-20261025/` |
| World player profile | 23 cases: overlapping objects, three price types, charge/shield/status changes, visibility/window exclusion, four boss forms, shrines, ten terrain types, parries, club attacks, bells, dormancy and supercharge. `runs/world-mechanics-20261025-release/` |
| World internal profile | The same 23 cases and failure checks. `runs/world-mechanics-internal-20261025-release/` |
| Collection failure boundary | A frozen two-transition collection followed by unknown-identity, capacity-overflow and timeout injections. Each failure retains one completed transition, aborts collection, rejects reuse and writes a diagnostic bundle. The world-profile reports record unchanged weights and zero optimizer updates/restarts. |

Private hosts, raw reports and failure bundles remain ignored under `runs/`.
Reusable fixtures are in `tools/qualify_world.py` and
`tools/qualify_world_failures.py`. Tests cover malformed data, capacity, hidden
fields, co-location, handle/order invariance, model sensitivity and checkpoint
reconstruction, terminal/truncation snapshots, and failure behavior. These checks
qualify the declared interface and mechanic families; they are not an exhaustive
test of every enemy interaction or proof of learning performance.

## Information-policy and task/reward audit — 2026-10-26

The audit is complete for the current solo Bard symbolic harness. It does not
certify a screen/audio-only agent. Both existing information profiles expose
privileged current engine state; the misleading implication that `player-v1`
means human-equivalent has been removed. This is a valid declared RL problem,
but success on it must not be reported as screen-only play.

`harness_contract.information_contract()` records the actual boundary. A10 binds
it and `task_contract()` into its architecture specification; incompatible or
missing contracts are rejected when reconstructing the model or environment.
Training records both contracts and the reward-accounting revision in run and
checkpoint metadata. Actor and critic receive the same projected observation.

| Input or signal | Actual source and audit conclusion |
| --- | --- |
| Local terrain and map memory | Lua `Vision.isVisible/isRevealed`, `Tile.getInfo/isSolid`, and Python visit history. Coarse geometry can refresh on revealed but off-screen cells; map items reveal more geometry. This is engine map knowledge, not a guarantee of remembered pixels. |
| Exact world identity and terrain | Registry IDs from engine names on visible cells. Names can distinguish disguises and walls containing loot. Both profiles retain this privilege. |
| World facts and relations | Internal profile includes the 100 scalar fields, component markers and grab relations. Player profile hides excluded scalar values and applicability, but display filtering and the remaining markers, relations and selected facts are not a renderer-equivalence proof. |
| Equipment | Internal profile includes current ownership, storage and counters. Player profile uses the qualified HUD selectors and unknown sentinels; engine-derived holder effects and world records remain part of the declared symbolic interface. Legacy inventory bypasses are cleared in this profile. |
| Player vector | Engine health, gold, groove, position, level, boss type, music clock/song state and calculated shop volume. These are not all certified screen/audio observations. A10 replaces bridge sequence with the count of acknowledged episode actions; GOTO setup commands do not increment it. Prefix handoff preserves this history. |
| Inputs and masks | Eleven engine input attempts. Missing equipment or a failed effect may produce a no-op. A9/A10 training and evaluation CLIs require `current`; navigation priors, wall pruning and dagger-only masks are different historical action contracts. |
| Task | Explicit objective and target level, independently of boss identity. No remaining collection budget is exposed as a survival deadline. |
| Recurrent feedback | Previous action is available. A10 ignores previous reward at the shared model encoder, covering rollout, evaluation, prefix replay and direct model calls. Hidden engine events cannot enter recurrent state through this numeric feedback channel. Learning targets can still use declared privileged rewards. Historical models retain reward feedback. |
| Diagnostics | Seed, run/session/launch IDs, raw events and raw entity handles stay outside policy tensors. Handles become relational row references, with identity-renaming and permutation tests. No future RNG is exported. |

A strict human-equivalent profile would require a separate rendering/disguise,
terrain-reveal and audio contract. It is not silently approximated here, and is
not required to study learning in this explicitly privileged symbolic task.

### Task and boundary decisions

A10 now receives `task_context = [objective, target_level]`: `[1, 0]` means win
the run; `[2, N]` means reach sequential level N. Start-level/health assistance
remains reset-distribution metadata, not a hidden change to the objective.
Different target tasks require A10 conditioning in the training entry point;
historical unconditioned architectures remain usable for a fixed objective.
Levels must be integers representable in the observation; fractional, boolean
and string values are rejected instead of silently coerced by reset JSON.

Winning or reaching the declared target terminates successfully; death
terminates unsuccessfully. `completed` continues to mean a full-run win;
`curriculum_completed`, `task_success` and episode status distinguish a subtask
success. A target success is credited by level progression, without an extra
full-run victory bonus. Collection limits truncate and retain the final state
for bootstrapping. Discounting is per acknowledged decision, not per inferred
beat or wall-clock second. True finite-horizon tasks are not implemented.

An engine `aborted` outcome now raises a protocol failure before adaptation or
reward scoring. The collector invalidates the fragment and stops the run; it
cannot bootstrap from a stale terminal snapshot or treat interruption as a
negative-reward episode. The historical `aborted` coefficient remains readable
for old configuration metadata but is not scored by the current harness.

### Reward accounting and incentive conclusions

Coefficients and existing reward configuration files are unchanged. Accounting
is separately versioned by `reward_semantics()` revision 2. New checkpoints bind
this revision, so an old run is not an exact resume merely because its weight
file matches. Evaluation rejects a declared incompatible accounting revision
unless explicitly used as a source reference; older missing metadata is marked
as an accounting mismatch in evaluation reports. A9 tensor shapes remain unchanged.

| Reward family | Definition, limits and interpretation |
| --- | --- |
| Progress, victory and death | Credit new maximum level/zone progress, full-run victory and death. Returning to a previous floor cannot re-earn progression. Health assistance and later-level starts remain distinct task distributions. |
| Exploration and carried item novelty | Episode novelty history plus persistent per-floor budgets. Returning to a floor cannot replenish its budget. Item novelty includes carried storage contents from engine state, not every consumed pickup; it is explicitly privileged shaping. |
| Combat | Local-player-attributed health loss and kills, per-target damage limits, kill deduplication and a per-floor budget. The exporter classifies non-player characters with health; this is not a hostility/team filter, so neutral characters can also qualify. Optional boss scopes narrow the targets. Handles are scoped to floors. These bonuses can prefer fighting over avoidance and do not preserve a pure completion objective. |
| Player damage | Actual health lost, including intentional blood equipment costs, before healing or revival. This can discourage a useful self-damaging action; it remains a deliberate incentive to evaluate, not a harness defect or a new equipment penalty. |
| Stair and boss potentials | `gamma * next_potential - previous_potential`, with zero potential at death, victory or target completion; collection truncation retains it. Tests check discounted cancellation at all true task terminals. Boss potential is the historical visible minimum-health heuristic, not a general damage counter across all boss phases; world `player-v1` still rejects it because exact boss health is excluded there. |
| Currency and containers | Currency now shares its cap across all events in one decision. Container IDs are deduplicated per floor. Neither stream has an episode-wide budget; renewable positive-credit loops remain possible if enabled. Both weights are zero by default. |
| Turn/revisit and historical V2 | Positive turn/revisit bonuses can reward stalling; V2 retains unbounded historical shaping. The current default has zero turn/revisit reward. These settings are explicit alternative objectives, not generally safe defaults. |

Lua events now carry their zone/floor at capture time. A final kill on the
departing floor cannot spend the arrival floor's budget; events actually on the
arrival floor use that floor's account. Historical events without attribution
use the departing floor. Malformed partial/noninteger attribution is rejected.
Unsupported negative bounded-credit weights and noninteger count limits are
also rejected instead of being silently clipped or coerced.

The default exploration, combat and item budgets total 1.75 per floor, compared
with 5 for a normal floor completion. This local bound is useful, but does not
prove that the shaped discounted objective always prefers eventual victory.
Only the potential terms have the tested telescoping property; novelty, combat,
item and damage terms change preferences. Keep completion, depth, death and
subtask success separate from shaped return when training resumes. No reward
recipe or hyperparameter was tuned during this audit.

### Verification

- `runs/information-task-audit-20261026-player/`: 23 live world fixtures and seven
  task/feedback/failure checks passed. Both task contexts round-trip; changing
  reward feedback leaves frozen logits, values and recurrent state identical.
  Unknown identities, capacity overflow, timeout and abort each stop collection
  after one retained diagnostic transition, with no retry or optimizer update.
- `runs/information-task-audit-20261026-events/`: 38 existing equipment interaction
  fixtures passed with world capture and event-time level attribution checks.
- Both private-game runs report zero worker restarts and unchanged original
  game/mod files. Reports, private hosts and diagnostic bundles remain ignored
  under `runs/`; the reusable tests and qualifier changes belong in source control.
- The full regression suite passed: **574 tests**, with Ruff and `git diff --check`
  also passing. Tests cover hidden-diagnostic and reward-feedback invariance, task
  sensitivity and gradients, checkpoint mismatches, floor reentry, reused event
  handles, currency aggregation, potential cancellation and abort handling.

Training remains paused. These results qualify the interface and accounting;
they do not demonstrate that a policy can learn to beat the game.
