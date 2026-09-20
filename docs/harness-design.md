# Bard harness design

Status: implementation design with offline contract scaffolding, grounded in the current code on
2026-09-20. This document does not claim new equipment support. No live game,
training, performance or learning experiment was run to prepare it.

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

The immediate deliverable is the contract and implementation plan below.
Implementation and offline checks can follow without launching experiments.
Live qualification remains pending while the user wants experiments paused.

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
choice, not a requirement of reinforcement learning. Define two explicit profiles:

- `player`: static game knowledge, current visible cues and remembered player
  history. This preserves the existing intended information boundary.
- `privileged`: additionally permits declared internal state such as exact
  internal equipment counters or unseen current entities. This is a different
  learning environment and must be named in results and checkpoint metadata.

Neither profile includes future RNG, arbitrary engine IDs or transport/debug
internals as policy features. Both actor and critic use the same profile in the
initial implementation; a privileged critic would be a separate explicit design.
No switch to privileged observations is made here. The exporter may supply
internal diagnostics separately, but they must not leak into masks or inputs. Separate static game knowledge,
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
be handled consistently. Fields used by shaping must follow the declared
information policy, including any explicitly remembered history.

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

Stages 1–6 are design/implementation work. Offline tests can check contracts and
known recorded fixtures; they cannot prove live engine behavior. Stage 7 and all
learning experiments remain pending under the current instruction.

For every mechanic, track discovery, source inspection, implementation, offline
verification and live qualification independently. Completion means the selected
content profile is covered, not that a few seeds avoided unsupported equipment.

Remaining source-inspection questions: exact Bard loot restrictions and unlocks;
shared activation handler precedence; observable versus hidden ammo/cost state;
maximum supported inventory/container occupancy; co-located entity limits; and
whether any ordinary Bard decision requires an additional input. These are
engineering questions to resolve from code before scheduling experiments.


## Offline reference contract and remaining integration

`src/autodancer/harness_contract.py` makes the following decisions executable:

- `StateFact`: known zero, unknown and inapplicable are different states; numeric
  data must be finite and carries an information-source classification.
- `project_facts`: validates that requested fields belong to the selected
  information policy. It rejects leakage rather than silently dropping fields
  and changing the schema. Classification itself still needs source inspection.
- `ControlState` / `executable_mask`: one descriptor per existing input, no
  strategic or redundancy pruning, unknown availability stops collection, and
  an all-masked nonterminal state is an error. A known-executable input can have
  an unknown owner; ownership alone is not an action legality test.
- `TransitionTiming`: decision and game-turn discounts are distinct; unknown
  engine duration cannot be replaced by the command sequence number.
- `ObservedEffect`: post-action effects carry command association and evidence;
  multiple authoritative or inferred effects may describe the same transition.
  Association identifies the transition window, not proof that Bard directly
  caused every world event within it.

These are reference validators, not a new Lua payload, model input schema or
production action contract. They are deliberately not connected to current
training: doing so without engine-backed descriptors would fabricate guarantees.
The legacy bounded contract and its redundant-arming rule remain unchanged.

Next implementation dependencies are engine-backed field classifications and
control descriptors, exact registry IDs and inventory relations, wire decoding,
policy tensor projection, then shared collector/evaluation integration. Support
and eligibility must remain unresolved wherever source evidence is incomplete.
Offline tests validate the new invariants without claiming live qualification.
