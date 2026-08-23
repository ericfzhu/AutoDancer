# Reward strategy

AutoDancer uses reward profile version 4. The objective is to complete a Bard
All Zones run; shaping exists to make that sparse objective learnable, not to
replace it. Rewards are computed in Python from validated live telemetry and
each transition records a `reward_components` breakdown.

The rationale, evidence, and next hypothesis for every profile are maintained
in [`reward-history.md`](reward-history.md). V4A is the default implementation;
V4B changes only the strength of known-stair potential for a controlled pilot.

## Default weights

| Component | Reward | Anti-exploit rule |
| --- | ---: | --- |
| Turn | 0 | Legitimate play is not charged merely for lasting longer |
| First visit to a position | +0.005 | Exploration and tile credit share a +0.5 per-floor budget |
| Revisit | 0 | Familiar coordinates are neutral |
| Newly revealed tile | +0.001 | Once per coordinate; at most 25 per turn and within the exploration budget |
| Enemy damage | +0.01 per point | At most 16 points per enemy; shares the combat budget |
| Enemy kill | +0.15 | Once per entity; damage and kills share a +0.75 per-floor budget |
| Player damage | -0.15 per point | No cap beyond available health |
| New inventory type | +0.10 | Once per exact type; at most +0.50 per floor |
| Currency | 0 | Removed because collection can distract from progression |
| Container opened | 0 | Removed because it is weakly aligned with progression |
| Stair potential | `0.99 × phi(next) - phi(current)` | `phi` is bounded to [0, 0.5] from known-stair distance |
| Floor completed | +5.0 | Paid for every floor advancement, including a zone boundary |
| Zone completed | +10.0 | Additional payment on a positive zone transition |
| Victory | +50.0 | Terminal win only |
| Death | -2.0 | Terminal death only |
| Abort or turn limit | -1.0 | Truncation only |

The hierarchy is intentional. Exploration, combat, and equipment shaping are
positive but strictly bounded per floor, so they cannot be farmed indefinitely
or outweigh repeated dungeon advancement. Returning to known positions is
neutral rather than punitive. Stair guidance uses the same `gamma = 0.99` as
PPO with `phi(s) = 0.5 × (1 - min(distance, 20) / 20)` once stairs are known.
V4B uses `1.0` instead of `0.5`, restoring an effective `0.05` per-tile signal.
Discovery changes that bounded potential naturally; floor and terminal changes
close the previous potential rather than dropping its debt. Generic `success`
and `failure` events are not separately rewarded, avoiding double payment for
victory, death, or truncation.

Each transition records `extrinsic_reward` for floor, zone, terminal, and
victory outcomes plus `shaping_reward` for all auxiliary guidance. Episode
metrics report both separately.

## Auditing and tuning

Training metrics include totals such as `reward_turn`, `reward_new_position`,
`reward_enemy_kill`, and `reward_floor_complete` for every rollout. The local
dashboard shows the component breakdown for each worker's latest transition.
Evaluation continues to report unshaped gameplay outcomes—completion, floor
progress, kills, damage, and pickups—so a higher shaped return alone is never
treated as evidence of better play.

Override any subset of defaults with a JSON object:

```json
{
  "enemy_kill": 0.15,
  "max_combat_reward_per_floor": 0.75,
  "stair_potential_max": 1.0
}
```

Then pass `--reward-config .\weights.json`. Unknown fields are rejected. The
resolved profile and version are saved in `config.json` and every checkpoint;
exact resume fails if the reward profile differs.

Weights should be changed only after fixed-seed evaluation shows a specific
failure mode. In particular, do not raise survival or damage rewards merely
because training return is low: compare floor progress, completion, deaths,
and behavior on the dashboard first.
