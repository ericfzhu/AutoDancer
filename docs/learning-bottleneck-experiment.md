# EXP-0035: Bounded learning-bottleneck experiment

Date: 2026-09-08. Protocol: `experiments/EXP-0035/experiment.yaml`.
Evidence: `runs/learning-bottleneck-exp0035/`.

Follow-up: [THROW semantics diagnostic](throw-semantics-diagnostic.md) confirmed
that the first THROW arms the dagger, a subsequent direction throws it, and
repeated THROW leaves it armed. The `special_no_effect` label does not capture
that initial activation. EXP-0035's outcome counts and decision are unchanged.

The frozen-policy preparation screen failed the predeclared training gate.
The six optimizer runs were therefore not launched. This is a completed
preparation diagnosis, not a completed comparison of learning algorithms or
learning curves. The fixed-versus-varied learning question remains inconclusive.

## What was fixed before gameplay

- Frozen A8 checkpoint: `runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt`.
  Its SHA-256 was recorded in `manifest.json` before the first game launch.
- Reward and recurrent reward feedback: DeathMetalPotentialV5.
- Action contract: `map-navigation-prior-v1`; eight workers; normal recurrent carry.
- Direct curriculum start: Bard, level 4, target level 5, `player20`, at most 500 learner turns.
- Current controller qualification: `runs/controller-qualification-steam-parity/qualification.json`.
  The older memory-controlled qualification did not match the current mod; the
  newer Steam-parity report did. No qualification check was bypassed.
- Candidate game seeds: 214000 through 214255. Selection used reset boss identity
  only. Of 256 seeds, 52 contained Death Metal. The first 40 sorted Death Metal
  identities were partitioned into 16 training, 8 development, and 16 final-test seeds.
- Two fixed stochastic action streams: 215001 and 215002. These are evaluation
  sampling streams, not optimizer replications.
- Gate: 10--90% direct-start success on the 16 training seeds across those two
  streams. Development scores could not change difficulty or select seeds.
- Positive control: qualified tail-60 handoffs on familiar seeds 92008, 92096,
  and 92116, with exact live prefix replay and warm recurrent state.
- One-hour preparation cap and seven-hour total cap. On a failed preparation
  gate, finish the narrower frozen-policy diagnosis without spending the training budget.

The final-test pool received reset-only identity screening, but no policy actions
or policy outcome evaluation. The new pools were not used for this checkpoint's
most recent finite-pool training. An exhaustive audit of every ancestral training
transition was not performed, so this report does not certify complete ancestral
non-exposure.

## Primary stochastic results

| Start condition | Seed pool | Completions | Mean learner turns | Mean boss damage |
| --- | --- | ---: | ---: | ---: |
| Direct | 16 training seeds, two streams | 0/32 | 372.5 | 0.56 |
| Direct | 8 development seeds, two streams | 0/16 | 402.4 | 0.25 |
| Direct | 3 historically familiar seeds, two streams | 0/6 | 324.8 | 0.17 |
| Qualified tail-60 handoff | Same 3 familiar seeds, two streams | 6/6 | 21.7 | 1.67 |

All 60 scored episodes had valid controller reports and zero worker restarts.
The direct-start episodes contained 31 deaths and 23 turn-limit endings.
Only 13 of 54 direct-start episodes recorded any boss damage. Across 20,307
direct-start actions, 4,435 were classified as special actions with no effect and
4,538 as wall attempts. These are behavioral diagnostics; discovering a wall can
provide information, and the counts do not establish a causal action-mask defect.

The comparison on familiar seeds is particularly informative: changing game seed
is not necessary to observe failure. The policy succeeds after the guide but
fails from the earlier direct start even on those same seeds.

## What the positive control actually establishes

The guide executed 35, 38, and 22 actions for the three familiar seeds. At handoff,
the reported initial boss health was 1, 2, and 2 HP, respectively; direct-start
reports recorded 9 HP. The learner then finished in 16, 21, and 28 turns in each
stochastic stream.

"Tail 60" describes how much of the stored demonstration remains after the
handoff. It does not mean the policy performed the first 60 actions of the fight,
nor does it require the policy to take 60 actions to finish.

The two start conditions differ in game state, remaining task difficulty,
previous action/reward inputs, and recurrent history. This test does not isolate
which difference causes failure. It supports a gap between the learned handoff
task and direct boss starts; it does not prove seed memorization, insufficient
network capacity, or inability of PPO to learn the full encounter.

## Secondary execution-mode check

The predeclared deterministic check produced 0/27 direct-start completions and
3/3 qualified-handoff completions. All 27 direct episodes reached the 500-action
limit without recording boss damage. Of 13,500 direct actions, 12,636 were THROW
commands; 12,663 actions (93.8%) were classified as `special_no_effect`.

That classification means the action did not produce the movement or relevant
events used by the outcome classifier. It does not prove every internal game
state variable remained unchanged. Nevertheless, the repeated commands produced
no measured encounter progress.

Code inspection identifies a concrete follow-up: `AutoDancer.lua` currently sets
the THROW mask from whether the weapon inventory slot is nonempty. This is not
an explicit check of whether the current weapon/state can execute THROW. Verify
the real game's THROW prerequisites and input semantics, then test an appropriate
action-contract correction separately. No action mask was changed in this study,
and this observation does not establish that masking alone solves the fight.

Across both stochastic streams and the deterministic check, all 90 scored
episodes had valid controller reports and zero restarts. The deterministic
result did not change the stochastic gate or select replacement seeds.

## Cost and next decision

The primary live preparation pipeline took 729 seconds: 106.6 seconds for identity
screening, 470.3 seconds for the two direct evaluations, and 152.0 seconds for the
two handoff controls. The deterministic direct and handoff evaluations added
254.1 and 76.9 seconds. Including the gap between primary and secondary commands,
the live preparation elapsed time was 1,112.3 seconds (18.5 minutes). Protocol and
runner preparation preceded that live timer; the whole preparation remained
within its one-hour cap. These times include game startup and cleanup. They are
not a breakdown of pure engine, inference, or reset costs.

No new PPO updates, demonstrations, or simulator work were performed. Therefore
there are no learning curves, optimizer seed comparisons, or new PPO timing
fractions to report. The previously proposed 4.5-hour training allocation remains
unused.

The next diagnostic should first verify why THROW remains selectable during
these unproductive repetitions, then target the transition from guided finishing
to handling earlier encounter states. Before comparing seed diversity, establish
an intermediate task with comparable starts across seeds and measurable source
competence. An earlier-handoff study should report actual boss state and guide
dependence, not just the nominal demonstration tail length. Faster simulation
could make that work cheaper, but this screen does not establish it as the
dominant learning bottleneck.

## Reproducibility and validation

`tools/run_learning_bottleneck_preparation.py` runs the registered screen into a
fresh output directory and refuses to overwrite it. Its `--deterministic` mode
adds the optional secondary check within the original preparation deadline.
`tools/analyze_learning_bottleneck_preparation.py` verifies source/report hashes,
seed selection, episode coverage, controller health, execution modes, assistance,
reward feedback, and absence of final-test gameplay before producing `analysis.json`.

Validation passed for all six frozen-policy reports and the identity selection.
Ruff checks and registry/spec validation passed. All owned game workers exited.

No policy is promoted by this diagnostic. Pre-existing performance reports and
engine-pacing probe files are preserved.
