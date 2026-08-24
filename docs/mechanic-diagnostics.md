# Live mechanic diagnostics

Reward and architecture experiments are paused until the live action and
mechanic ladder is trustworthy. `autodancer-diagnose` exercises the same hidden
worker, named-pipe bridge, observation decoder, and authoritative engine turns
used by training.

## Action contract

Logical action IDs remain fixed:

| ID | Action | Live availability |
| ---: | --- | --- |
| 0–3 | Up, right, down, left | Always legal while Bard is running |
| 4 | Wait / engine idle | Always legal while Bard is running |
| 5 | Bomb | Bomb slot exists |
| 6–7 | Action items 1–2 | Slot exists and is ready |
| 8 | Throw | Weapon slot exists |
| 9–10 | Spells 1–2 | Slot exists and is ready |

Every action transition must echo the requested logical ID and must contain an
engine action observed by SYNCHRONY that exactly equals the injected engine
action. A running record is rejected if directions or wait are unavailable, or
if any special-action mask bit disagrees with its inventory slot.

## Outcome classifier

The classifier compares the authoritative observation before and after the
action plus emitted events. Its mutually exclusive primary categories are:

- `move`, `wait`, `combat`, `interaction`, `dig`, or `floor_transition`;
- `combat_attempt` when Bard targets a visible enemy without a damage event;
- `wall_attempt` when a direction targets a wall without movement or a visible
  terrain change;
- `unchanged_direction` for another directional no-displacement outcome; and
- `special_no_effect` for a legal non-directional action with no classified
  effect.

`wall_attempt` is deliberately not called “blocked”: a real dig can require
multiple hits, and an unchanged attack can be legitimate. The structured result
keeps target terrain, target actor, event kinds, movement, and floor-change
flags so later analyses do not have to infer those facts from a single label.

## Reproducible mechanic ladder

The default diagnostic uses contract seed `46000` and exploration seeds
`46001–46003`. It resets before each isolated action check, then runs the
deterministic `LiveExplorer` against normal Bard All Zones floors. The report
records first evidence for movement, wall attempts, digging, combat,
interactions, visible-stair routing, floor transitions, and an immediate return
to a previously occupied position.

Actions are paced at 120 ms by default. A single diagnostic worker otherwise
outpaces the engine's future-turn input buffer—unlike training, where inference
and eight independent environments naturally limit each worker's action rate.

This is a diagnostic, not a training environment. It does not inject items,
enemies, rooms, or hidden state. Consequently, an item or spell action that
never becomes legal is reported as `masked_unavailable`, and a mechanic absent
from the fixed runs is reported as `observed: false`. Such a result identifies
a missing deterministic fixture; it is not counted as a mechanic pass.

## First measured run

The initial full run on game `v4.2.1-b5713` / Steam build `22938426` completed
3,011 mechanic turns across seeds `46001–46003`, plus the isolated action
checks, with zero worker restarts.

- The engine acknowledged all actions that became legal, including exact
  engine-action equality for four directions, wait, bomb, and throw.
- The normal starting loadout did not make action-item or spell slots legal,
  so IDs 6, 7, 9, and 10 remain `masked_unavailable` rather than verified.
- The run captured movement, wait, combat, a no-damage combat attempt, digging,
  item collection, wall attempts, unchanged directional actions, and a return
  to a previously occupied position.
- None of the three deterministic explorations revealed a staircase, so stair
  routing and floor-transition coverage remain open fixture gaps.

The machine-readable report is written to
`runs/mechanic-diagnostics/report.json`. Runtime reports remain ignored by Git;
this measured summary is the durable experiment record.
