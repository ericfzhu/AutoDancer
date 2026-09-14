# Bounded direct-start learning pilot — EXP-0036

The pilot completed with valid controller, scope, numerical and artifact checks,
but showed no improvement in its paired final evaluation. No checkpoint was
selected or promoted. The decision is **inconclusive**: this is one short
optimizer trial, not evidence that the task cannot be learned.

## Results

| Pool | Frozen completion | Final completion | Mean boss damage, before → after |
| --- | --- | --- | --- |
| Training seeds | 0 / 32 | 0 / 32 | 0 → 0 |
| Development seeds | 0 / 16 | 0 / 16 | 0 → 0 |

Both evaluations used `bounded-bard-v1`, assisted Bard `player20`, direct Death
Metal starts at level 4, a level-5 completion target, a 500-turn cap, the same
16 training and 8 development seeds, and stochastic streams 215001 and 215002.
The final-test pool remained untouched. All 96 evaluation episodes were valid.

Wall attempts occupied 20,209 / 23,679 frozen turns (**85.35%**) and 20,273 /
22,732 final-policy turns (**89.18%**). The recurring failure was pressing into
walls, not repeated dagger arming. Frozen evaluation recorded three deaths;
final evaluation recorded four; the remaining episodes reached their turn cap.

The single fine-tune used optimizer seed 217001, A8, DeathMetalPotentialV5,
fresh optimizer state, unchanged source model/critic weights at initialization,
no frozen parameters, and 16 updates of 1,024 transitions each. It completed
exactly **16,384 transitions**. Of 55 completed training episodes, 13 dealt boss
damage, totaling 34 damage, but none completed the task. These training-time
encounters did not translate into boss damage in the fixed final evaluation.
Training metrics cover changing policies and sampling streams, so they are not
interchangeable with the frozen-checkpoint evaluation.

Core optimization metrics remained finite and no workers recovered or restarted.
Policy updates were sometimes substantial: approximate KL reached 0.2553 at
update 9. Entropy varied from 0.0846 at update 1 to a peak of 0.8984 at update 3
and ended at 0.3302. This is diagnostic evidence of changing action concentration,
not proof that an optimizer setting caused the evaluation failure.

## Time

| Stage | Seconds |
| --- | --- |
| Frozen evaluations | 459.672 |
| Training, including startup | 498.344 |
| Final evaluations | 464.702 |
| Other elapsed time, including the directory repair interval | 164.265 |
| Total measured execution | 1,586.983 — 26.45 minutes |

Training took 8.31 minutes, below its 20-minute cap. Total measured execution
remained below the 45-minute cap. The initial metadata-only launch failure
preceded this execution clock and performed no gameplay. Its artifacts are
preserved. The later directory-identity failure and repair are included in the
reported total. Evaluation consumed more wall-clock time than this short training
pilot; that is relevant when budgeting future diagnostics.

## Interpretation and next experiment

The interface fixes are qualified correctness changes. This pilot does **not**
show that they independently improve learning, because the bounded contract also
removes the earlier navigation assistance. The source policy had been trained
with that assistance and guided handoffs. Both pre- and post-training policies
here spent most turns attempting walls and never damaged the boss in evaluation.

The next focused comparison should retain corrected dagger state and the loadout
guard, and vary only the earlier navigation assistance. First evaluate frozen
policies, before buying another training run. That would distinguish a navigation
dependency from a combat-learning problem without changing rewards, architecture
or game speed. Do not interpret this pilot as justification for an engine port,
a large hyperparameter sweep, or a claim that longer training cannot help.

## Artifacts and audit

- Contract: `experiments/EXP-0036/experiment.yaml`.
- Current results: `runs/bounded-learning-exp0036-complete/validated-analysis.json`.
- Runtime inputs and timings: `manifest.json` and per-stage invocation files in
  the same directory.
- Evaluations: root-level `frozen-215001.json`, `frozen-215002.json`,
  `pilot-215001.json`, `pilot-215002.json`; fresh tracked evaluations have separate
  `evaluations/<stage>/` directories.
- Training: `training/config.json`, `metrics.jsonl`, `episodes.jsonl`, `final.pt`.
- Final checkpoint SHA-256:
  `c593fa9452461db9bde4e80d9c9760914fde7b588e75dd7f62e5ffb1294faa85`.

The first successful frozen stream was reused after a tracked-output-directory
error. Its report hash, original command and elapsed time remain recorded;
model/controller/reward source hashes and the immutable protocol were unchanged.
There was no outcome-based seed or checkpoint replacement. The analyzer verified
all 16 updates, exact seed coverage, unchanged source hashes, checkpoint identity,
zero recoveries, finite core optimization metrics, and both execution caps.

Run `tools/analyze_bounded_learning_pilot.py --run-dir
runs/bounded-learning-exp0036-complete` with the repository Python environment to
repeat the audit. Experiment registration and Ruff checks passed. All game
workers were closed after execution.
