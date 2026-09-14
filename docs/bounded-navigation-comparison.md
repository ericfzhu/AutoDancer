# Frozen navigation comparison — EXP-0037

This diagnostic varies navigation assistance while holding the corrected dagger
controls and bounded equipment guard fixed. It uses the original pre-pilot A8
checkpoint, not the fine-tuned EXP-0036 checkpoint. No training takes place.

## Design

- Reuse the 48 completed EXP-0036 frozen evaluations as the control.
- Run 48 new evaluations: the same 16 training and 8 development seeds, each with
  stochastic policy streams 215001 and 215002. Final-test seeds remain untouched.
- Both arms use eight workers, Bard player20, direct level 4 to level 5,
  500-turn episodes, and DeathMetalPotentialV5 reward and policy feedback.
- The new `bounded-bard-navigation-v1` contract wraps `dagger-controls-v1` with the
  existing loadout guard. It restores remembered invalid-wall masking and the
  exploration/stair prior. Armed dagger directions bypass navigation masking.
- The twenty-minute new-evaluation cap and ten-minute per-stream cap are hard
  limits. Scope violations, recovery, source drift or incomplete coverage abort
  the attempt rather than becoming gameplay losses.

The control reports are reused to save evaluation time. The runtime manifest
records their hashes and their original source manifest. The only changed
runtime source is the additive action-contract wrapper; existing no-navigation
behavior remains covered by regression tests. This is a paired historical-control
comparison, not a fresh interleaved A/B run. Two policy streams are repeated
measurements of the same 24 seeds, not 48 independent level samples.

## Correctness evidence

The live mechanics refresh passed all 22 cases, including repeated successful
raw trace replays, bomb consumption/detonation/exhaustion, dagger retrieval,
random bounded-loadout audits on all 16 training seeds, and the actual collector.
That refresh exercises the base bounded mechanics. The new navigation wrapper
has additional unit checks for delegation equivalence, slot-local wall memory,
reset, batch projection, armed-direction preservation, and terminal scope failure.
Its live integration is checked by the frozen evaluations themselves. The
combined targeted regression suite passed 205 tests.

The old third raw trace still contains two repeated THROW actions excluded by the
corrected policy mask; its successful raw replay does not qualify it as a new
policy demonstration. No demonstrations are used in this comparison.

## Results

All 48 new episodes were controller-valid, with zero recoveries and no loadout
scope violations. New evaluation elapsed time was 385.078 seconds (6.42 minutes),
with streams taking 189.375 and 195.437 seconds. The mechanics refresh is separate
from this evaluation time. All game workers were closed after the comparison.

| Metric | No navigation (reused control) | Navigation restored |
| --- | ---: | ---: |
| Completed episodes | 0/48 | 0/48 |
| Episodes with boss damage | 0/48 | 8/48 |
| Total boss damage | 0 | 17 |
| Wall attempts / all turns | 85.35% | 30.54% |
| Mean distinct positions per episode | 11.90 | 37.73 |
| Deaths | 3/48 | 40/48 |
| Turn-limit endings | 45/48 | 8/48 |

Training-seed evaluations produced 15 boss damage in 7/32 episodes; development
produced 2 damage in 1/16. Both pools remained at zero completions. Stream 215001
produced 13 damage and stream 215002 produced 4; thus contact was not confined to
one sampling stream, but development evidence is still weak.

The restored exploration/stair prior was active for 7,870 of 13,780 turns
(57.1%). There were 4,209 discovered invalid-wall states and 4,209 wall attempts.
Wall-state memory is deliberately sensitive to observed state changes, so a
blocked direction can reopen after its context changes. This comparison does
not establish whether a coarser safe signature would improve learning.

## Interpretation

This supports navigation dependence of the frozen source: restoring the existing
assistance substantially reduces wall attempts and creates some boss contact.
It does not demonstrate learned direct-start competence or faster learning.
The increased deaths show that exploration brings the agent into fights it
cannot yet reliably handle; surviving by repeatedly attempting walls was not
useful task progress.

The next bounded learning test should retain this navigation assistance and the
corrected controls, repeat the same short training budget from the original
source, and evaluate the final checkpoint on the same development protocol.
That would ask whether the increased combat exposure produces learning. Keep
architecture and reward fixed for that comparison. A change to a game port or
a broad hyperparameter search is not supported by this diagnostic.

No checkpoint was promoted, and no claim is made that the current harness covers
all weapons or equipment. Support remains the explicitly guarded Bard loadout.

## Artifacts

- Immutable design: `experiments/EXP-0037/experiment.yaml`.
- Results and input hashes: `runs/bounded-navigation-exp0037/`.
- Live mechanics refresh: `runs/bounded-navigation-mechanics-refresh/`.
- Reproduction runner: `tools/run_bounded_navigation_comparison.py`.
- Artifact audit: `tools/analyze_bounded_navigation_comparison.py`.

The exact executed runner is retained as `executed-runner.py` in the run folder.
The audit permits only an AST-identical formatting cleanup of that runner;
model, contract, game and evaluation source hashes must still match exactly.
