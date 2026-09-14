# THROW semantics diagnostic

Date: 2026-09-08. Follow-up to EXP-0035, without changing that experiment's
protocol, decision, checkpoint, or reward.

The bridge sends the correct THROW command. With Bard's starting dagger, THROW
arms a directional throw; it does not immediately throw the weapon. Repeating
THROW while armed leaves the weapon armed. A subsequent directional command
throws the dagger while the player stays in place.

## Live evidence

`tools/probe_throw_semantics.py` ran one worker in a private game/mod copy.
The only mod change was logging the real weapon's `itemActivable.active` state,
its `weaponThrowable` component, and inventory/mask/position identity alongside
normal observations. Input mapping, game pace, and action handling were unchanged.

The test used seeded level-4 `player20` starts on 92008 and 214001. Eight sequences
were tested on each seed, each from a fresh reset: 16 cases and 42 actions total.
The live portion took 21.75 seconds with zero restarts. Both seeds agreed.

| Input sequence | Observed behavior |
| --- | --- |
| RIGHT | Player moves right; dagger stays equipped and unarmed. |
| THROW six times | First command arms dagger; all later commands leave it armed. Player stays still. |
| THROW, then any cardinal direction | Dagger leaves weapon slot; player stays still; THROW becomes masked because no weapon is equipped. |
| THROW, THROW, RIGHT | Second THROW does not cancel arming; RIGHT still throws the dagger. |
| THROW, WAIT, RIGHT | WAIT does not cancel arming; RIGHT still throws the dagger. |

The tested dagger is genuinely throwable. This is not evidence that the bridge
selected a nonexistent action or that THROW should always be disabled.

## Representation and diagnostic gaps

1. **Arming is absent from the inventory observation.** The real component changes
   from `itemActivable.active=false` to `true`. The eight-element weapon inventory
   row remains unchanged. `encodeInventoryItem` currently exports
   `itemToggleable.active` in its last column, a different component.
2. **Redundant THROW remains selectable.** The base mask enables THROW whenever
   the weapon slot has an item. It does not distinguish armed from unarmed state.
3. **Outcome labels miss the state transition.** The first valid arming command
   is labeled `special_no_effect`. A successful directional throw into empty space
   is labeled `unchanged_direction`, despite the weapon leaving inventory.
4. **Directions have state-dependent meaning.** RIGHT means movement when unarmed
   and a throw when armed. The map-navigation action contract needs to account for
   that distinction before applying movement-specific restrictions.

The entire model input is not claimed to be identical across these transitions:
sequence counters, clocks, memory ages, and recurrent previous-action input can
change. The LSTM could in principle infer arming from history. The demonstrated
gap is the absence of an explicit authoritative armed-state feature, coupled with
availability of repeated redundant commands and misleading outcome categories.

The EXP-0035 count of `special_no_effect` remains a correct count of its recorded
labels, but those labels must not be interpreted as proof that every command had
no game-state effect. This probe confirms that the first THROW changes state.

## Concrete correction to test next

Use a versioned observation/action-contract change that exposes the armed state,
masks redundant arming when appropriate, and interprets directional actions as
throws while armed. Correct the outcome diagnostics to distinguish arming,
throwing, and actual no-ops. Do not automatically add positive reward for arming
or throwing: neither by itself establishes progress toward winning.

Preserve the legacy path for checkpoint comparisons. Other weapons require
separate checks: the installed game's effective Weapon module also handles reload
behavior on the THROW action, and ActionItem includes conversion behavior. A
global rule that simply requires `weaponThrowable` could remove valid actions.

This diagnostic does not measure how much these corrections would improve
training or boss success. No production observation, mask, reward, or checkpoint
was changed, and no policy was trained or promoted.

## Sources and verification

- Bridge mapping: `mods/AutoDancer/scripts/Bridge.lua`, `LOGICAL_TO_ENGINE[8]`.
- Inventory encoding and mask: `mods/AutoDancer/scripts/AutoDancer.lua`,
  `encodeInventoryItem` and `mask[9]`.
- Outcome classification: `src/autodancer/outcomes.py`.
- Installed v4.2.1 archive: effective `scripts/necro/game/item/Weapon.lua` comes
  from `versions/v4.2.1.wsp`; inspected constants include `prepareThrow`,
  `itemActivable`, `weaponThrowable`, and `handleThrowAttack`. No extracted game
  module was executed outside the game. The live test supplies the behavioral
  evidence; bytecode strings alone do not establish control flow.
- Ignored evidence: `runs/throw-semantics/live/{protocol,result,summary}.json` and
  `game.log`; inspection artifacts under `runs/throw-semantics/inspection/`.

The summary validator checks episode/probe alignment, real activation state,
weapon removal and stationary position after directional throws, unchanged
inventory encoding across repeated arming, and zero restarts. Original installed
executable/archive/config and repository Lua hashes matched after the test. The
private host remains in the ignored evidence directory; its worker was closed.
