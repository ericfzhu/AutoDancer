# Bounded Bard mechanics contract

This increment scopes the next learning-interface experiment to assisted Bard
(`player20`), direct Death Metal floor starts (level 4), and completion at level 5.
It does not reopen EXP-0035 or its held-out test pool. No PPO training is included
in this mechanics qualification.

## Enforced scope

Use `--action-contract bounded-bard-v1`. The bridge compares exact entity names
and slots, rather than lossy hashed type IDs, against this allowlist:

| Slot | Allowed equipment |
| --- | --- |
| Weapon | `WeaponDagger`, or empty after a throw |
| Shovel | `ShovelBasic`, or empty |
| Bomb | `Bomb` / `Bomb3` finite stacks, or empty |
| All other slots | Empty |

The additive `bounded_loadout = [version, allowed]` field is checked on reset,
each policy observation, and after transitions including terminal transitions.
Missing telemetry is an error. Extra equipment raises `UnsupportedLoadoutError`;
the collector propagates this and does not record it as a completion, death, or
controller recovery. An experiment encountering this error is invalid and must
stop. Do not discard such episodes and continue training on the survivors.

This is a runtime boundary, not proof that arbitrary movement can never encounter
other loot. The game is not modified to prevent pickups. If a chosen seed/policy
escapes the boundary, extend and qualify support or explicitly redesign the
environment before resuming that experiment. Do not select replacement seeds by
their learning outcomes.

## Action semantics

All engine-available directions and WAIT are preserved. The previous walking
navigation prior and learned wall exclusions are absent from this new contract,
so aiming, attacking, digging, confusion and other contextual direction meanings
are not incorrectly excluded by a walking heuristic. Existing contracts retain
their original behavior.

Dagger armed state is projected into the existing weapon active feature. THROW
prepares a throw; the next direction releases it. Repeated arming is suppressed
within the bounded loadout. WAIT retains armed state. After release, the weapon
slot can be empty; retrieval must restore the dagger and its action availability.

Bomb input remains available according to the authoritative finite stack. Bomb
placement diagnostics require consumption plus a visible lit bomb; delayed
explosion is not classified as an immediate no-effect action. Existing visible
bomb and beat-delay channels remain the observation representation. No extra
reward is awarded for preparing, throwing, or placing a bomb.

Other weapons, spells, boots, holsters, stored food/scrolls, alternate characters,
and equipment combinations are outside this experiment. They remain cataloged
work, not implicitly supported mechanics. Ordinary health, enemy damage, item
collection and floor-transition telemetry still comes from the existing harness.

## Qualification layers

`tools/qualify_bounded_gameplay.py` checks the exact assisted boss setup, known
completion traces twice with new normalized observation digests, bomb consumption
and detonation, dagger retrieval, a fixed 128-action audit on each of the 16
already-selected training seeds, and the actual collector with the new contract.
It does not play development or final-test seeds. The training-seed audit is
coverage sampling; the runtime guard supplies the continuing scope enforcement.

The general controller qualification independently exercises movement, walls,
digging, attacks, kills, damage, item collection, traps, death, level boundaries,
same-seed replay, forced worker recovery and collector integration. The refresh
uses 8 workers and 4,096 transitions per worker. Memory is sampled every 128
transitions, producing 32 samples per worker. Qualification now rejects fewer
than 20 samples instead of treating an unmeasured short run as stable memory.
This is a short functional/stability refresh, not another million-transition
endurance qualification or proof of long-run memory behavior.

The original demonstration bank remains immutable. A replay check is mechanics
evidence; it does not automatically create a new hash-bound guide bank for
training. Prefer direct starts for the next interface experiment; any trace-tail
training must use separately regenerated, qualified guide artifacts for its
declared action contract.

Changing from `map-navigation-prior-v1` changes both observation/action semantics
and navigation assistance. Name that combined intervention explicitly in the next
experiment; do not attribute any improvement solely to the dagger feature.

## Results — 2026-09-08

Both qualification layers passed against the current repository Lua.

| Check | Result |
| --- | --- |
| Targeted regression suite | 202 passed |
| General controller qualification | All six phases passed; artifact freshness validated |
| Eight-worker short soak | 32,768 transitions, 269.969 seconds, zero natural restarts or infrastructure events |
| Worst worker action p99 | 110 ms (250 ms acceptance threshold) |
| Largest measured sustained RSS growth | 2.56% (5% threshold); short-soak evidence only |
| Known boss completion traces | Seeds 92008, 92096, 92116 reproduced twice; 95, 98, 82 turns respectively |
| Bombs | Placement, one-bomb stack consumption, delayed disappearance and exhausted mask passed |
| Dagger | Arming/direction/WAIT/reset checks plus live retrieval passed |
| Training-seed scope audit | All 16 seeds, 1,950 actions, zero scope violations; two ordinary deaths |
| Scoped collector | 16 × 1 rollout passed; no optimizer updates |
| Scoped gameplay qualification | 22 cases, 2,510 scripted actions plus 16 collector actions, 75.531 seconds, zero restarts |

The scoped runs observed enemy damage/kills, player damage, digging, item
collection, and six successful floor transitions in addition to the dedicated
equipment checks. These observations are coverage evidence, not exhaustive proof
over every reachable game state. The exact-name runtime guard remains active.

The raw trace for seed 92116 contains redundant THROW inputs at zero-based action
indices 50 and 52, which conflict with the new mask. Its unchanged engine replay
succeeds, but it cannot be imported unchanged as a policy-valid guide for this
contract. The other two traces had no policy-mask conflicts. None of these checks
replaces generation of a fresh guide bank if guide training is later selected.

Evidence:

- `runs/controller-qualification-bounded-bard-v1/qualification.json`
- `runs/bounded-bard-dagger-verification/result.json`
- `runs/bounded-bard-gameplay-qualification-final/protocol.json`
- `runs/bounded-bard-gameplay-qualification-final/result.json`
- `runs/bounded-bard-gameplay-qualification-final/transitions.jsonl`

The first scoped attempt at `runs/bounded-bard-gameplay-qualification` is preserved
as failed. Its script constructed a second adapter for an existing worker,
restarting command IDs, then obscured that failure during cleanup. The final
script reuses one vector adapter and captures diagnostics before closing it. The
complete corrected attempt passed; the failed attempt is not qualification evidence.

All workers were closed. Shared mod deployment occurred through the normal
controller qualifier. No final-test gameplay, training run, or policy promotion
was performed. Use the new qualification report and the explicit bounded contract
for the next direct-start experiment; the original EXP-0035 decision is unchanged.
