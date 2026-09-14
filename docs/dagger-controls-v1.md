# Dagger control interface v1

Implemented and verified on 2026-09-08. This is the first equipment-control
increment following the [THROW diagnostic](throw-semantics-diagnostic.md).
No training or policy promotion was performed.

Follow-up: [bounded Bard mechanics qualification](bounded-bard-mechanics.md) adds
an enforced loadout boundary, preserves contextual directions, and records fresh
controller and gameplay qualification. The original results below describe the
earlier dagger-only increment.

## Interface

The Lua observation includes `equipment_controls = [version, starting_dagger, armed]`.
Version 1 reads `itemActivable.active` from an equipped `WeaponDagger` with
`weaponThrowable`. It does not infer armed state from action history. Other weapon
variants, reload weapons and character-specific controls are outside this increment.

This is an additive extension to bridge schema 10. Legacy records without the field
decode to `[0, 0, 0]` (unknown), not a claimed unarmed dagger. The new action contract
rejects missing extension telemetry. Invalid versions, shapes and flag combinations
are rejected by the protocol decoder.

Raw inventory, engine action availability, and input mapping retain their existing
semantics. The policy opts in using `--action-contract dagger-controls-v1` in the
training or baseline CLI. That contract:

- Projects dagger armed state into the weapon row's final inventory feature
  (Python `[0, 7]`). The network already consumes this feature, so model dimensions
  remain unchanged. Other equipment retains its original toggle feature.
- Uses the existing map-navigation-prior-v1 when the dagger is unarmed.
- Bypasses navigation preferences and learned walking-wall exclusions while armed.
  It preserves the engine's directional availability and WAIT.
- Masks repeated THROW only for the bounded loadout containing weapon, shovel and
  bombs. If additional equipment is present, THROW remains available because boots,
  holsters and conversions can share that button. This conservative exception is
  covered by a synthetic test; those equipment combinations are not live-qualified.

The first THROW remains a tactical policy choice. The harness does not combine
preparation and direction into a macro or prescribe where to aim.

## Outcomes and rewards

Outcome telemetry now distinguishes `throw_prepared`, `throw_already_armed`, and
`weapon_thrown`. Combat and interaction retain category precedence, with the weapon
mechanic also recorded in `equipment_action`. A throw into empty space is no longer
reported as a walking-wall failure or an unchanged direction. Preparation and
release no longer inflate the baseline accumulator's special-no-effect or repeated
walking-direction counters.

These are mechanic labels, not claims of tactical benefit. Reward policies are
unchanged; preparation and an empty-space throw do not receive a new reward.

## Compatibility and qualification

Existing action-contract names keep their policy inventory and mask behavior.
The model ignores the additive control field unless the new contract projects it.
CPU/CUDA inference tests cover both legacy and packed transfer paths with the new
field present. Checkpoints retain their dimensions, and the selected action contract
is recorded through existing checkpoint metadata.

Switching an old checkpoint to `dagger-controls-v1` is an explicit intervention:
its observation semantics and masking change. It must not be described as an
unchanged-checkpoint baseline. Historical diagnostic category counts also should
not be compared without accounting for the corrected labels.

The repository Lua changed, so previous controller qualification hashes are stale.
Before a training experiment, run fresh controller qualification and requalify any
demonstration/handoff evidence affected by the new telemetry or action contract.
The bounded check below verifies dagger behavior; it does not replace that broader
controller qualification. The shared installed mod was not deployed by this check.

## Verification

- 194 targeted regression tests passed across protocol, action contracts, baseline,
  vector environments, inference transport, policy runner, asynchronous collection,
  training and qualification. After extending transport coverage, the 16 transport
  and dagger tests also passed.
- Final private-host live check: 16 cases, 36 actions, seeds 92008 and 214001,
  player20 assisted Bard, level 4 to 5; 21.562 seconds, zero worker restarts.
- Cases cover normal movement, preparation, repeated THROW, all four release
  directions, WAIT retaining armed state, and reset clearing state. The script
  deliberately sends redundant actions through raw engine availability to verify
  the distinction between engine input and policy masking.
- Source hashes remained unchanged during verification. Game workers were closed.
- Ruff and `git diff --check` passed.

Evidence: `runs/dagger-controls-v1-verification-final/protocol.json`, `result.json`,
and `game.log`. The protocol records the tested Lua and Python source hashes.
Reproduce with a fresh output directory:

```powershell
.venv\Scripts\python.exe tools/verify_dagger_controls.py --output runs/dagger-controls-v1-new-check
```

The next learning experiment can compare the old interface against this explicit
intervention after fresh qualification. No learning or throughput gain has yet been
measured. Reload/ammo support remains a separate subsequent increment.
