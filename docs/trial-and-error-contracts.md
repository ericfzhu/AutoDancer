# Trial-and-error tasks and incentives

This setup learns exclusively from the learner's own actions. It uses no
demonstrations, imitation loss, guide checkpoint, replayed prefix or navigation
heuristic. It is a bounded Bard/dagger setup, not full-game qualification.

## Fixed tasks

Use each JSON with the existing `--curriculum-mixture` argument, in a separate
run. Each file has exactly one entry: task identity cannot change invisibly
between resets. The observation's `PlayerFeature.TASK` means **boss identity**,
not the training objective. Do not mix objectives at the same observed state
without adding explicit task conditioning to the policy.

| Config | Start | Success | Suggested collection limit |
| --- | --- | --- | --- |
| `configs/task-first-floor-v1.json` | Ordinary level 1 reset | Enter level 2 alive | 256 actions |
| `configs/task-zone-one-v1.json` | Ordinary level 1 reset | Enter level 5 alive | 1,024 actions |
| `configs/task-boss-floor-v1.json` | Native level skip to level 4, normal profile | Enter level 5 alive | 512 actions |

First-floor is the initial learning task. Zone-one tests whether the skills
compose. Boss-floor isolates combat and exiting: it does not establish that the
agent can reach the boss. Native level skipping is an artificial start
distribution; it does not replay a teacher or award rewards for skipped floors.
No profile boosts player health or weakens the boss. Boss-floor is not restricted
to Death Metal unless the experiment separately verifies the seed's boss identity.

Task success is `curriculum_complete`; it is not a full-game win. Death is a true
terminal failure. The action limit is a collection truncation and retains value
bootstrapping, not a newly defined finite-horizon task. A deadline-based task
would need remaining-time observations and different terminal handling.

Use `bounded-bard-v1`. It exposes the dagger's armed state and suppresses redundant
arming; it leaves walking, waiting and wall attempts to the learner. Unsupported
equipment invalidates the bounded experiment rather than becoming a death or a
silently filtered reset. Normal floors can contain such equipment, so longer
tasks need additional equipment qualification before large-scale training.

## Paired reward profiles

`configs/reward-trial-sparse-v5.json` rewards actual progress: 5 per floor,
10 extra per zone, 50 for winning. It has no survival, currency, item, damage,
kill, death or abort reward. Infrastructure failure is not a gameplay objective.

`configs/reward-trial-exploration-v5.json` has the same objective rewards, plus:

- 0.005 per new position and 0.001 per newly revealed local tile, sharing a
  **0.5 total budget per floor**. Revisiting and waiting cannot refill it.
- Stair-distance potential up to 0.5, using only revealed stairs from the local
  grid or persistent map. Distance is Manhattan distance, not a route planner;
  walls and locked stairs can make it misleading.
- Visible boss-health progress potential of 0.2 per health lost, capped at 9.
  This is the existing V5 signal, not repeated credits for boss adds.

Both use gamma 0.99. Existing checks reject a PPO gamma that differs from the
shaping discount. Potential is cancelled on true termination and retained at
time limits. Novelty is an intentional auxiliary objective, **not**
policy-invariant shaping. Its cap does not prove that farming early exploration
cannot outweigh a distant, discounted success. Compare held-out task completions
and repeat the comparison without novelty before promoting a policy. A high
shaped return alone is not progress.

Do not select a reward profile just because it produces more nonzero rewards.
Report task completions, deaths, time limits, floor reached, boss contact/damage,
wall attempts, environment actions and wall-clock time separately. Use identical
training seed pools and held-out evaluation seeds across the two reward arms;
do not tune on evaluation failures. Keep architecture and optimizer fixed for
that comparison. Previous stable fine-tuning used LR 3e-6 and target KL 0.02;
that does not establish the right rate for a fresh randomly initialized model.

## Harness corrections

- Telemetry integers are checked against storage bounds **before** narrowing to
  int8/int16/int32. Overflow now fails instead of wrapping into plausible data.
- Reward configuration rejects NaN and infinite numeric values.
- Stair shaping can use revealed stairs already present in the agent's map,
  even beyond the local grid. Unknown terrain is not stair evidence.
- The Death Metal episode tracker counts the finishing-hit event when boss
  identity clears on the same floor, then closes attribution. Events on later
  floors or another boss cannot contaminate that counter.

Tensor shapes and checkpoint architecture are unchanged. No Lua changes were
needed. Historical reports retain their original hashes and measured outcomes;
the old missing finishing-hit counts are not retroactively rewritten. Dynamic
off-screen entities remain absent from map memory. These fixes do not establish
that every gameplay-relevant variable is observable or that every weapon works.

## Reproducible checks

The dedicated smoke command runs a random masked policy, with one native worker
at a time. It validates raw and policy observations, finite/decomposed rewards,
and equality of the two reward profiles' extrinsic objectives on identical
transitions. It saves source/config/mod hashes and partial results on failure.
Its 64-action episode limit deliberately exercises truncation. It is neither
training nor a learning-performance experiment.

The first run passed 576 native transitions (192 per task), including 28 deaths
and 4 time-limit truncations. There were no task completions; no claim of
learnability follows from this random-policy check. Finishing-hit attribution
and successful task boundaries are covered by regression tests rather than this
smoke. Result: `runs/trial-error-harness-qualification-v1/result.json`.
All 452 repository tests pass; targeted Ruff checks pass. The full-suite run also
required correcting an older imitation-data test fixture to use policy input
keys rather than all transport telemetry fields; imitation remains unused here.

```powershell
.venv\Scripts\python.exe tools/qualify_trial_and_error.py `
  --game-dir 'X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64' `
  --output runs/trial-error-harness-qualification-v1 --steps 192
```

For a training run, the first-floor task arguments are:

```text
--curriculum-mixture configs/task-first-floor-v1.json
--reward-config configs/reward-trial-exploration-v5.json
--action-contract bounded-bard-v1 --gamma 0.99 --max-turns 256
```

Pass these to `python -m autodancer.training.train` along with its required game
directory, worker count, total steps and new run directory. Use a separately
declared sparse arm rather than changing a running experiment's rewards.
