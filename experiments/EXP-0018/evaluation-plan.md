# EXP-0018 conditional mixed-replay evaluation

This plan applies because the frozen player6 gate failed. It is fixed before the
first optimizer trial completes.

Select the ascending first 24 telemetry-confirmed Death Metal seeds from
`74001-74256`. Evaluate the unchanged parent and all three final checkpoints on
both `boss1hp-player6` and `boss1hp-player10`, with 500 turns per episode and:

- deterministic argmax (`policy_seed=0`);
- stochastic `policy_seed=104001`;
- stochastic `policy_seed=104002`.

The two sampled streams are the primary policy statistic; argmax is a deployment
diagnostic. All reports must use the same ordered seeds, eight workers, Reward
V4A, A2, `map-navigation-prior-v1`, and the exact qualified controller with zero
infrastructure contamination.

For each checkpoint, compute sampled player6 completion/death, distinct
successful seeds, player10 completion, and deterministic diagnostics. The arm
passes only if:

1. mean sampled player6 completion across its three checkpoints is at least 50%;
2. at least two checkpoints improve sampled player6 completion over the matched
   frozen parent;
3. at least one checkpoint individually has player6 completion at least 50%,
   player6 death at most 50%, and player10 completion at least 80% of the
   parent's matched-seed player10 rate;
4. all three training trials have finite losses, changing parameters, exact
   worker capacity, and zero natural recovery.

Among individually eligible checkpoints, select lexicographically by sampled
player6 distinct successful seeds, completion rate, negative death rate,
player10 retention, deterministic player6 completion, then median success turns.
Shaped return cannot select or rescue a checkpoint. If the arm fails, retain the
EXP-0017 parent and add an intermediate player-health profile rather than
changing reward or architecture post hoc.
