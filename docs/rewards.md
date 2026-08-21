# Reward strategy

AutoDancer uses reward profile version 1. The objective is to complete a Bard
All Zones run; shaping exists to make that sparse objective learnable, not to
replace it. Rewards are computed in Python from validated live telemetry and
each transition records a `reward_components` breakdown.

## Default weights

| Component | Reward | Anti-exploit rule |
| --- | ---: | --- |
| Turn | -0.005 | Always applied; passive survival has negative return |
| First visit to a position | +0.015 | Once per world coordinate per episode |
| Newly revealed tile | +0.001 | Once per coordinate; at most 25 per turn |
| Enemy damage | +0.025 per point | At most 16 points per enemy entity |
| Enemy kill | +0.25 | Once per enemy entity |
| Player damage | -0.15 per point | No cap |
| New inventory type | +0.15 | Once per exact type per episode |
| Currency | +0.002 per unit | At most 25 units per turn |
| Container opened | +0.05 | Driven by the engine event |
| Floor completed | +3.0 | Positive floor transition only |
| Zone completed | +5.0 | Positive zone transition only |
| Victory | +25.0 | Terminal win only |
| Death | -2.0 | Terminal death only |
| Abort or turn limit | -1.0 | Truncation only |

The hierarchy is intentional. Repeated waiting loses reward. Exploration can
help early learning but is exhausted on a visited map. Combat is useful but a
floor transition is worth substantially more than ordinary enemies. Victory
dominates every shaping event on its transition. Generic `success` and
`failure` events are not separately rewarded, avoiding double payment for
victory, death, or truncation.

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
  "turn": -0.0075,
  "enemy_kill": 0.3,
  "floor_complete": 4.0
}
```

Then pass `--reward-config .\weights.json`. Unknown fields are rejected. The
resolved profile and version are saved in `config.json` and every checkpoint;
exact resume fails if the reward profile differs.

Weights should be changed only after fixed-seed evaluation shows a specific
failure mode. In particular, do not raise survival or damage rewards merely
because training return is low: compare floor progress, completion, deaths,
and behavior on the dashboard first.
