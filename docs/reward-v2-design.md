# Reward-v2: staircase navigation

Reward-v1 improved survival, combat, pickups, and local exploration but never
reached floor 1-2 in held-out evaluation. Reward-v2 targets that measured
failure without adding generic survival reward or changing the observation,
action, or model architecture.

## New signal

When a staircase first appears in the revealed symbolic grid, the episode gets
a one-time `stairs_discovered` reward. From the following transition onward,
`stair_progress` is the signed change in Manhattan distance to the nearest
known staircase:

`reward = 0.05 × (previous_distance - current_distance)`

The distance change is capped at four tiles for teleports and traps. A normal
step toward stairs is +0.05; reversing it is -0.05. Discovery establishes a new
potential baseline so newly revealed information is not double-counted as
movement. Floor changes clear the potential.

## Objective hierarchy

Floor, zone, and victory rewards increase to 5, 10, and 50 respectively. The
existing turn cost, novelty bounds, combat credit deduplication, player-damage
penalty, death penalty, and truncation penalty remain unchanged. A -0.01
`revisit` component makes safe coordinate loops negative instead of merely
withholding further novelty reward. Checkpoints
store reward profile version 2 and reject reward-v1 resumes.

## Acceptance

- Stair discovery appears in live rollout component metrics.
- Moving toward and exactly back from known stairs has zero net stair reward.
- No distance credit crosses floor boundaries.
- Fixed-seed evaluation must compare furthest floor, staircase discovery,
  deaths, kills, and pickups; shaped return alone is not success.
