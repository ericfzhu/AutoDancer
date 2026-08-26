# Reward policy history

This document records why each AutoDancer reward profile was introduced, what
was measured, what was learned, and what the next version is intended to test.
It is an experiment ledger: measured results are kept separate from hypotheses
and shaped return is never treated as gameplay competence by itself.

The invariant objective is to win normal Bard All Zones runs on unseen seeds.
Reward shaping is only a learning aid for that sparse objective. Gameplay is
evaluated with floor and zone progress, victories, deaths, damage, turns,
kills, and pickups rather than with the training reward alone.

## Timeline

| Profile | Status | Trigger | Intended improvement | Result |
| --- | --- | --- | --- | --- |
| Legacy event reward | Retired | First end-to-end live trainer | Prove that live PPO could learn from engine events | Learned survival/passivity, no held-out progression |
| Reward V1 | Retired | Legacy policy rarely explored or fought | Add bounded progress, exploration, combat, and item signals | Better survival, combat, pickups, and exploration; still no stairs |
| Reward V2 | Retained baseline | V1 never left floor 1-1 | Add staircase discovery and distance guidance; suppress safe loops | Strongest local competence and floor progress so far; 24 training floor transitions, no zone completion; penalties dominated return |
| Reward V3 | Rejected | V2 learned local competence but not sustained progression | Make progression authoritative and shaping positive, bounded, and potential-based | Reduced timeouts but replaced them with unsafe activity; no held-out progress gain |
| Reward V4A/V4B | Rejected | V3 lost local competence | Restore bounded combat and item competence; separately test stronger stair potential | Became safer but overwhelmingly passive; no held-out progress gain |

## Before versioning: direct event reward

### Why it existed

The first reward was deliberately small and direct. Its job was to prove that
the Lua bridge, live environment, recurrent trainer, and engine events formed a
working learning loop before investing in reward design.

It paid `-0.001` per turn, `+1` for generic success, `-1` for generic failure,
`+0.05` per point of enemy damage, `+0.1` per kill, `-0.1` per point of player
damage, and `+0.001` per revealed tile. It had no stateful deduplication, no
explicit floor hierarchy, and no item or navigation model.

### What was expected

Frequent damage and reveal events were expected to give PPO a detectable signal
before the very sparse end-of-run outcome. No claim was made that these weights
represented a final objective.

### What happened

The 8,192-transition live baseline changed every model tensor and remained
numerically stable. On 16 held-out seeds, its deterministic policy died much
less often than masked random (6.25% versus 93.75%) and lasted longer (121.44
versus 19.19 turns), but recorded no kills, pickups, floor transitions, or
completions. It had found a strong survival/passivity policy.

### Decision

Survival was not sufficient. The next profile needed stateful, bounded credit
for useful exploration and local competence, explicit progression milestones,
and component-level auditing. This led to Reward V1. The complete experiment is
recorded in [`baseline.md`](baseline.md).

## Reward V1: progress-first stateful shaping

### Why it was chosen

The legacy policy demonstrated that negative damage and terminal signals could
teach the agent not to die without teaching it to play. V1 was designed to make
new positions, revealed space, combat, equipment, and dungeon progress visible
while preventing the easiest repeatable reward loops.

### Design

V1 introduced an episode-local `RewardTracker` and profile versioning. Its key
weights were:

| Component | V1 weight |
| --- | ---: |
| Turn | -0.005 |
| First visit to a world position | +0.015 |
| Newly revealed tile | +0.001, capped at 25 per turn |
| Enemy damage | +0.025 per point, capped at 16 per entity |
| Enemy kill | +0.25, once per entity |
| Player damage | -0.15 per point |
| New inventory type | +0.15, once per episode |
| Currency | +0.002 per unit, capped at 25 per turn |
| Container opened | +0.05 |
| Floor / zone / victory | +3 / +5 / +25 |
| Death / abort | -2 / -1 |

The hypothesis was that one-time exploration credit and deduplicated combat
credit would produce active but non-farmable behavior, while larger milestone
rewards would preserve the intended hierarchy.

### What happened

An 8,192-transition, eight-worker run produced 110 completed episodes, 104
deaths, 10 kills, and 2 currency pickups. On held-out seeds, V1 reduced the
death rate from 81.25% for masked random to 43.75%, increased mean survival from
33.50 to 82.75 turns, recorded 6 kills and 6 pickups, and took 23 fewer points
of damage. It still never progressed beyond floor 1-1.

### Interpretation and decision

V1 fixed the strongest legacy failure: the policy now explored, fought, and
collected items rather than merely surviving. The measured bottleneck moved to
navigation. V2 would therefore preserve V1's observation, action, architecture,
and local shaping while adding an explicit signal for finding and approaching
stairs. See [`reward-v1-baseline.md`](reward-v1-baseline.md).

## Reward V2: staircase navigation and loop suppression

### Why it was chosen

V1 showed local competence but no held-out floor transition. The next
hypothesis was that the agent lacked a learnable connection between discovering
a staircase and moving toward it over several turns.

### Design

V2 raised floor, zone, and victory rewards to `+5`, `+10`, and `+50`. It added:

- `+0.5` once when a staircase coordinate was discovered;
- `0.05 × (previous_distance - current_distance)` for signed Manhattan-distance
  progress toward the nearest known staircase, capped at four tiles per turn;
- `-0.01` whenever the player revisited an episode-local world coordinate.

Discovery established a new distance baseline, distance was never compared
across floors, and reversing a movement toward stairs charged back its distance
reward. The revisit penalty was added after an equal-budget precursor produced
safe movement loops and never activated stair shaping. The design and precursor
validation are in [`reward-v2-design.md`](reward-v2-design.md) and
[`reward-v2-baseline.md`](reward-v2-baseline.md).

The working hypothesis was that signed stair progress would solve navigation
without rewarding oscillation, while the revisit penalty would make passive
coordinate loops unprofitable.

### Short validation

The final-profile smoke test completed 2,048 transitions and two finite PPO
updates. Revisit penalties appeared in both rollouts, proving the component was
active, but no staircase was encountered. This justified a longer competence
run while leaving floor completion as an unproven hypothesis.

### Full 250k experiment

The eight-worker `reward-v2-250k` run completed on 2026-08-22:

- 250,880 transitions and 245 PPO updates;
- 17.46 end-to-end steps/second and 96.38 final collection steps/second;
- 603 completed episodes and deaths;
- 1,156 enemy kills and 826 item pickups;
- 24 floor transitions, maximum Zone 1 Floor 2;
- no zone completion and no victory;
- 6 worker replacements, all recovered at fixed capacity.

Behavior changed substantially over training:

| Metric | First 50 updates | Last 50 updates |
| --- | ---: | ---: |
| Deaths | 227 | 129 |
| Enemy kills | 37 | 473 |
| Item pickups | 16 | 347 |
| Floor transitions | 1 | 11 |
| Mean episode return | -5.15 | -5.18 |
| Policy entropy | 1.31 | 1.07 |

The similar mean return concealed much better local gameplay. Aggregate reward
components explain why:

| Component | Full-run total |
| --- | ---: |
| Revisit | -2,292.93 |
| Turn | -1,254.40 |
| Death | -1,206.00 |
| Player damage | -405.30 |
| New position | +323.81 |
| Enemy kill | +322.50 |
| Floor completion | +120.00 |
| New tile | +54.15 |
| Enemy damage | +49.18 |

### Interpretation and decision

V2 succeeded at local competence and improved floor transitions late in the
run, but it did not solve sustained progression. Its universal turn and revisit
penalties accumulated throughout the long episodes required for success. They
also punished necessary backtracking, combat positioning, and searching. At
`gamma = 0.99`, future milestone rewards were already discounted, so the turn
cost imposed an additional preference for short episodes.

Combat and item signals were learnable but weakly aligned with the remaining
navigation objective. The signed stair delta resembled potential shaping but
did not use PPO's discount factor. V3 should therefore keep progression
milestones, remove renewable negative pressure on legitimate play, and strictly
bound every auxiliary positive signal.

Artifacts are in the ignored `runs/reward-v2-250k/` directory. The resolved
configuration and raw rollout metrics are `config.json` and `metrics.jsonl`.

## Reward V3: bounded progression-first proposal

**Status: retired after held-out evaluation on 2026-08-23.** Values below
are the current defaults and experiment specification.

### Intended question

Can V2's learned local competence be retained while making deeper progression
the only renewable source of substantial reward?

### Proposed task reward

- `+5` for every floor transition;
- an additional `+10` for a zone transition;
- an additional `+50` for victory;
- `-1` for death and `-1` for abort;
- no universal turn cost.

Task and shaping returns must be recorded separately. Checkpoint selection and
evaluation must use gameplay progress rather than total shaped return.

### Proposed shaping

- First visit to a world position: `+0.005`.
- Newly revealed tile: `+0.001`.
- Combined exploration credit: at most `+0.5` per floor.
- Revisited position: `0`; familiarity with a tile is neutral.
- Enemy kill: `+0.05`, at most `+0.5` per floor.
- Player damage: `-0.05` per point.
- New inventory type: `+0.05`, at most `+0.25` per floor.
- Enemy damage, currency, and container rewards: `0`.

Known-stair guidance becomes potential-based. With distance `d` to the nearest
known staircase:

`phi(s) = 0.5 × (1 - min(d, 20) / 20)`

and each transition receives:

`gamma × phi(next_state) - phi(state)`, with `gamma = 0.99`.

The implementation must define staircase discovery, floor boundaries, and
terminal potentials as part of the augmented reward state and test that loops
cannot manufacture return. There is no separate staircase-discovery payment;
discovery changes the potential naturally.

### Proposed experiment

1. Evaluate the final V2 checkpoint deterministically on held-out seeds.
2. Initialize V3 from V2's encoder, recurrent model, and actor; reset the critic
   and optimizer because value targets changed.
3. Run a 50,000-transition pilot rather than another immediate 250k run.
4. Evaluate V2 and V3 on the same seeds and step limits.
5. Continue V3 only if held-out floor or zone progress improves without a
   material collapse in survival.

The launched pilot uses independent training seeds `31001`, `31002`, and
`31003`, 51,200 transitions per seed, eight sequential live workers per run,
and `runs/reward-v2-250k/final.pt` as the common warm-start checkpoint. Runs are
written under `runs/reward-v3-directional/`.

Periodic evaluation should use 16 held-out episodes with a 3,000-step cap; the
final comparison should use 64 episodes with a 5,000-step cap. Save `best.pt`
lexicographically by victory rate, mean zones completed, mean floors completed,
then lower median turns. Keep `latest.pt` separately for exact recovery.

RND, behavioral cloning, hard curricula, reward annealing, and model changes are
deferred so this experiment isolates the reward change.

### Directional-pilot result

Three V3 checkpoints each completed 51,200 transitions and 50 PPO updates with
no worker restarts. Training trajectories looked favorable, but deterministic
evaluation on the same 30 unseen seeds rejected the hypothesis:

| Held-out metric per episode | V2 | V3 checkpoint mean |
| --- | ---: | ---: |
| Floor progress | 1.067 | 1.067 |
| Death rate | 40.0% | 54.4% |
| Step-limit rate | 60.0% | 45.6% |
| Enemy kills | 1.67 | 1.10 |
| Item pickups | 1.40 | 0.76 |
| Player damage | 1.80 | 2.49 |

V3 reduced passive timeouts but converted them into unsafe activity without an
aggregate progress gain. The large simultaneous reductions in combat, item,
damage-avoidance, and death signals prevented attributing the regression to one
component. V3 is not promoted; V2 remains the initialization and comparison
baseline. Seeds `41001` through `41030` are now development history and must
not be reused as unseen evaluation seeds.

## Reward V4: bounded competence restoration

**Status: rejected after the two-arm directional pilot on 2026-08-23.**

### Intended question

Can V3's reduction in passivity be combined with V2's local competence while
keeping all positive auxiliary rewards bounded below the floor milestone? A
second arm separately tests whether V3's stair gradient was too weak.

### Shared V4 design

- Keep turn and revisit reward at zero.
- Keep first-position and new-tile credit at `+0.005` and `+0.001`, sharing a
  `+0.50` per-floor exploration cap.
- Restore `+0.01` enemy-damage credit, limited to 16 points per enemy, and
  `+0.15` per unique kill. Damage and kills share a `+0.75` per-floor cap.
- Restore player damage to `-0.15` per point and terminal death to `-2`.
- Reward a new inventory type with `+0.10`, capped at `+0.50` per floor.
- Keep currency and container rewards at zero.
- Keep floor, zone, victory, and abort at `+5`, `+10`, `+50`, and `-1`.

V4A retains stair-potential maximum `0.5`. V4B changes only that maximum to
`1.0`; both use a 20-tile horizon and `gamma=0.99`. No inactivity penalty is
added because strategic Bard waits must remain valid and the V3 result already
showed that activity alone is not competence.

### Experiment and predeclared decision

Both arms warm-start policy weights from V2 with fresh critics and optimizers.
Each uses paired training seeds `32001`, `32002`, and `32003`, 51,200
transitions per checkpoint, and eight workers. V2 and all six checkpoints are
then evaluated on new seeds `42001` through `42030` with a 3,000-turn cap.

An arm passes only if its aggregate improves floor progress, death rate is at
most 50%, kills and items each retain at least 80% of V2 per episode, idle or
step-limit behavior decreases, and at least two checkpoints improve progress.
If both pass, choose by progress, death rate, idle rate, kills, then items. If
neither passes, retain V2. Shaped return is never a selection metric.

### Two-arm pilot result

All six V4 runs completed 51,200 transitions and 50 finite PPO updates. Neither
arm passed the predeclared gameplay gates on seeds `42001` through `42030`:

| Held-out metric | V2 | V4A aggregate | V4B aggregate |
| --- | ---: | ---: | ---: |
| Mean floor progress | 1.00 | 1.00 | 1.00 |
| Death rate | 23.3% | 3.3% | 1.1% |
| Step-limit rate | 76.7% | 96.7% | 98.9% |
| Enemy kills / episode | 1.37 | 0.24 | 0.18 |
| Item pickups / episode | 1.00 | 0.16 | 0.14 |

V4 made the policy safer but overwhelmingly passive. Neither arm discovered a
staircase or improved floor progress, and both lost most of V2's combat and item
competence. The decision was `retain_v2_reject_v4`; no V4 250k continuation was
started.

These checkpoints used architecture 2. Subsequent observation work added the
65x65 player-visible map memory, tactical state, hazards, interaction context,
and shopkeeper audio cue in architecture 6. V4 therefore does not answer
whether those missing inputs caused the navigation failure.

## Architecture 6 with Reward V2: representation-isolation experiment

**Status: rejected after the completed overnight experiment on 2026-08-24.**

This experiment changes the observation/model architecture while restoring the
exact Reward V2 calculation. Three architecture-6 policies warm-start from
`runs/reward-v2-250k/final.pt`, with fresh critics and optimizers, and train for
51,200 transitions on seeds `33001`, `33002`, and `33003`. V2 and the three
pilots are evaluated deterministically on new seeds `43001` through `43030`.

The architecture passes only if aggregate mean floor progress improves, at
least two pilots improve progress independently, all training artifacts are
finite and valid, and evaluation has no worker restarts. A passing experiment
continues the strongest gameplay-ranked checkpoint to 250,880 total
transitions. Reward V5 remains deferred until this representation question is
answered.

### Result

All three Architecture-6 pilots completed 51,200 transitions and 50 finite PPO
updates. Each checkpoint was evaluated deterministically on seeds `43001`
through `43030`, alongside the V2 reference:

| Held-out metric | V2 Architecture 2 | Architecture 6 aggregate |
| --- | ---: | ---: |
| Mean floor progress | 1.033 | 1.011 |
| Death rate | 30.0% | 32.2% |
| Step-limit rate | 66.7% | 67.8% |
| Enemy kills / episode | 1.47 | 0.83 |
| Item pickups / episode | 1.30 | 0.53 |
| Unchanged-position rate | 74.3% | 91.0% |
| Mean longest unchanged-position streak | 1,278 | 1,813 |
| Unique positions per 100 turns | 0.668 | 0.654 |
| Staircase discovery rate | 0% | 0% |

Architecture 6 did not improve aggregate progression, and only one pilot ever
reached floor 2. It lost substantial combat and item competence and produced
much longer stationary sequences. All recorded evaluation actions were
directional; explicit wait, item, bomb, throw, and spell actions were zero.
Consequently, the 91% unchanged-position rate does not describe passive
`WAIT` actions. It most likely contains repeated directional inputs into a
persistent obstruction, mixed with legitimate stationary attacks and digging.
The current diagnostic cannot yet separate those cases.

The predeclared decision was `retain_v2_architecture_2`. No Architecture-6
250k continuation was started. This result does not motivate a reward change:
Reward V2 already charged most unchanged turns `-0.015` through its turn and
revisit terms, yet the blocked-action behavior persisted. A stronger generic
stationary penalty could also punish combat and digging without solving the
underlying action-selection failure.

### Interpretation and next architectural question

Architecture 6 has 6.48 million parameters versus 5.95 million for Architecture
2. It retains the same 512-unit LSTM and actor/critic heads while adding the
65x65 map CNN and expanded tactical, inventory, player, interaction, hazard,
and audio inputs. Each pilot received only 51,200 new transitions, whereas the
V2 source checkpoint had accumulated 250,880.

More information should not reduce the best representable policy, but this
upgrade was not behavior-preserving. Adding the 128-value map representation
expanded the fusion input from 896 to 1,024 values. The old fusion matrix could
not be transferred and was randomly initialized; the new map encoder and new
feature columns were also fresh, while the transferred LSTM and actor suddenly
received a different latent distribution. The experiment therefore establishes
that this partially warm-started A6 did not recover and improve within 51,200
steps. It does not establish that explicit map memory is harmful or that a
fully trained A6 cannot outperform A2.

Before another reward version or a larger A6 run:

1. classify unchanged-position turns as productive combat, digging,
   interaction, changed-target movement, or repeated blocked movement;
2. stop advertising provably impossible movement as legal while preserving
   attacks and diggable walls;
3. add new sensory information through a zero-initialized residual adapter so
   the upgraded policy initially reproduces A2 logits exactly;
4. verify that equivalence before training, then evaluate learning curves at
   fixed checkpoints through a longer budget.

This will distinguish insufficient training time from an ineffective map
representation without using reward changes to compensate for an observation
or optimization problem.

## Architecture 7 with Reward V2: function-preserving sensory adapter

**Status: rejected after the completed paired pilot on 2026-08-24.**

### Question and causal hypothesis

Architecture 7 tests whether Architecture 6 failed because its partial warm
start destroyed the useful Architecture-2 latent representation, rather than
because persistent map and tactical information are intrinsically unhelpful.
It retains the complete Architecture-2 policy as a base and injects only the
schema-9 information unavailable to Architecture 2 through a bounded residual
adapter. The adapter gate starts at exactly zero. Before training, the upgraded
model must therefore reproduce the source checkpoint's logits, values,
recurrent states, and deterministic actions within numerical tolerance even
when the new inputs are perturbed.

This is an architecture experiment, not Reward V5. Reward V2, observation
schema 9, PPO settings, live workers, and the existing action mask remain
unchanged. In particular, the separately identified unavailable-`WAIT` mask is
not repaired inside this experiment because doing so would change the effective
action space and confound the A2/A7 comparison.

### Pre-training acceptance gate

The pilot may not start unless an automated parity test demonstrates all of the
following against `runs/reward-v2-250k/final.pt`:

1. actor logits, critic values, and both LSTM state tensors match at absolute
   tolerance `1e-6` over batched single steps and reset-containing sequences;
2. deterministic actions are identical;
3. arbitrary changes to map memory and schema-9-only grid, player, and
   inventory fields cannot affect the zero-gated A7 output;
4. every A2 tensor, including its critic, is loaded into the preserved base,
   while only the adapter is freshly initialized.

Training diagnostics must subsequently show finite PPO losses, a healthy fixed
worker count, a bounded finite gate, a changing gate, and eventually non-zero
adapter parameter gradients. A gate that remains functionally zero means the
new information was not learned and fails the hypothesis even if gameplay is
unchanged.

### Directional pilot and seeds

Three A7 checkpoints warm-start from the same A2 source with fresh optimizers
and train for 51,200 transitions on seeds `35001`, `35002`, and `35003` using
eight workers. A2 and all A7 checkpoints are evaluated deterministically on
new seeds `44001` through `44030` with a 3,000-turn cap. These seeds are fixed
before any A7 result is observed and become development history afterward.

Evaluation additionally separates unchanged-position outcomes into productive
combat or interaction, explicit waits, special-action no-ops, and unchanged
directional attempts. Repeated same-direction attempts at the same position
are recorded as a blocked-movement proxy; they are not claimed to be an exact
game-engine legality label.

### Predeclared gameplay decision

A7 passes only if its three-checkpoint aggregate:

- improves mean floor progress over A2, with the direction reproduced by at
  least two of three independently trained checkpoints;
- produces staircase evidence: discovery, discovery-to-exit conversion, or
  completed exits must improve, and zero discoveries is an automatic failure;
- keeps death rate no more than five percentage points above A2;
- retains at least 80% of A2 enemy kills and item pickups per episode;
- reduces unchanged directional attempts by at least 20%, excluding productive
  stationary combat and interactions; and
- does not worsen step-limit rate or the mean longest unchanged-position
  streak.

Selection is based on unshaped gameplay outcomes, never shaped return. A pass
authorizes a separately seeded 250,880-transition continuation and 64-seed
acceptance evaluation. A failure retains A2 and distinguishes three cases:
the gate never learns (optimization failure), the gate learns without gameplay
gain (information not useful at this budget), or richer inputs improve local
diagnostics without progression (navigation/control remains the bottleneck).

### Result

The real V2 checkpoint passed the mandatory initialization proof with exactly
zero measured error—not merely error below tolerance—for logits, critic values,
next LSTM state, reset-containing sequences, deterministic actions, and changes
to every A7-only input. All three pilots then completed 51,200 transitions and
50 finite PPO updates. The adapter gates changed and remained bounded, but
finished extremely close to closed: `-0.001276`, `+0.000265`, and `-0.000030`.

Deterministic evaluation on seeds `44001` through `44030` produced:

| Held-out metric | A2 | A7 aggregate |
| --- | ---: | ---: |
| Mean floor progress | 1.000 | 1.033 |
| Death rate | 46.7% | 40.0% |
| Step-limit rate | 53.3% | 58.9% |
| Enemy kills / episode | 1.53 | 1.51 |
| Item pickups / episode | 1.30 | 0.92 |
| Unchanged directional attempts | 76.7% | 91.7% |
| Repeated same-direction attempts | 67.5% | 89.6% |
| Mean longest unchanged-position streak | 1,341 | 1,693 |
| Unique positions per 100 turns | 0.813 | 0.815 |
| Staircase discoveries | 0 | 0 |

Two pilots improved mean floor progress independently: seed `35001` reached
1.033 and seed `35003` reached 1.067, while seed `35002` remained at 1.000.
The aggregate also improved survival and retained 98.6% of A2's kills. Those
are positive directional signals, but they do not satisfy the experiment.
There was no staircase evidence; item retention fell to 71%; step-limit and
stationary-streak metrics worsened; unchanged directional attempts increased
rather than falling by 20%; and evaluation required two worker replacements in
each of the latter two candidate waves. All evaluation actions again belonged
to the four directional actions.

The predeclared decision is `retain_v2_architecture_2`. No A7 continuation is
started. The result is classified primarily as an adapter-optimization failure,
not evidence that richer observations are harmful: the bounded scalar gate
never opened beyond magnitude 0.0013, so the new branch had almost no influence
while ordinary PPO updates could still move the preserved A2 base. The modest
progress gain therefore cannot be attributed confidently to map or tactical
information.

The next architecture experiment should retain exact A2 initialization while
allowing useful new features to receive gradients immediately—for example, a
zero-output final residual projection with a nonzero path, or a staged phase
that freezes the A2 base while training the adapter. It must also isolate the
directional-action collapse and the unavailable live `WAIT` mask before another
long representation pilot. Reward V5 remains deferred.

### Post-pilot representation diagnostic

After repairing the action contract and adding mechanic-level outcome
classification, the fixed representation test measured each observation group
in the A2 source and all three final A7 checkpoints. It used isolated
counterfactual perturbations plus encoder gradient norms, with a fixed
materiality floor of 1% relative to the median established A2 input path. The
threshold was added after A7's gameplay evaluation, so this A7 analysis is
descriptive; the gate is predeclared for future architecture candidates.

All six established input groups were material in every checkpoint. All four
A7-only groups were merely trace-active in every A7 checkpoint: their largest
relative sensitivity was `4.8768e-4`, and no new-input perturbation changed an
argmax action. The largest relative gradient was `4.3876e-2`, but that path's
output influence remained below the sensitivity gate. Thus zero of twelve
candidate/group pairs passed both requirements.

This upgrades the A7 conclusion from an inference based mainly on its tiny
scalar gate to a direct representation result. The richer observations were
connected, but did not acquire enough influence for the gameplay pilot to test
their usefulness. Future architecture candidates must pass this diagnostic—or
an equivalent controlled representation-learning gate—before broad live
training. Exact results and methodology are in
`docs/representation-diagnostics.md`.

### Architecture 8 controlled-learning protocol (predeclared)

Reward V2 remains fixed while Architecture 8 tests the representation-learning
failure identified above. The experiment includes two A2 controls: exact A2
fine-tuning under V2's legacy policy-side no-WAIT contract, and exact A2
fine-tuning under the repaired current 11-action contract. A8 uses the current
contract, preserves the complete A2 function through a zero-output residual
projection, and freezes the A2 base for its first 10,240 transitions.

All three arms use training seed `36001`, a 30,720-transition budget, and saved
curve points at 0, 10,240, 20,480, and 30,720. Deterministic curve evaluation
uses seeds `45001–45016`. A8 must pass material new-input tests at warmup and
final plus predeclared local-gameplay harm bounds before any broad integration
test. If it passes, final policies use seeds `46001–46030` with a 3,000-turn
cap. Exact criteria are in `docs/architecture8-controls.md`.

### Architecture 8 curve result

All three 30,720-transition arms and all eleven required curve reports
completed. A8 preserved exact initial A2 behavior, and all four new groups—
tactical grid, map memory, extended player state, and extended inventory—were
material at both 10,240 and 30,720 transitions. The residual projection norm
grew from `1.7694` to `2.1708`; this resolves A7's representation-learning
failure rather than merely showing a connected but trace-active branch.

At the final 16-seed curve point, fixed-contract A2 reached mean progress
`1.0625`, death rate `0.375`, step-limit rate `0.625`, `1.0625` kills per
episode, and `0.8125` items per episode. A8 reached mean progress `1.125`, death
rate `0.3125`, step-limit rate `0.6875`, `1.5625` kills per episode, and
`1.1875` items per episode. It therefore passed every predeclared
representation and local-gameplay harm criterion.

The overall curve gate nevertheless failed exactly as declared because both A2
control training runs recorded one worker restart; A8 recorded zero. Broad
gameplay was not run, so A8 is not promoted and A2 remains the baseline. The
result is `stop_before_broad_gameplay`, with positive candidate evidence but an
invalid control-health condition. The next valid step is an infrastructure-only
retry of the two controls, preserving the protocol and A8 artifacts, rather
than changing the architecture, reward, or decision thresholds.

During result inspection, the comparator was found to reuse fixed-A2 reports
for all A8 curve points instead of only the shared step-zero point. The stored
A8 reports were unaffected. The comparator and its regression test were fixed,
and the decision above uses the corrected reports.

### Architecture 8 qualified replication and horizon result

EXP-0007 repeated the A8 comparison with the schema-10-qualified controller.
A8 again proved exact A2 initialization parity and material use of all four
rich observation groups, but its broad 30,720-transition result tied both A2
controls on floor progress and missed the declared death gate. A2 remained the
promoted baseline.

EXP-0008 then isolated the remaining adaptation-horizon hypothesis by exactly
resuming the qualified A2 and A8 states—including model, critic, optimizer,
counters, and random state—to 250,880 transitions. Both training runs were
finite and controller-valid. The final held-out results were:

| Policy | Mean progress | Furthest floor | Death rate | Kills | Items |
|---|---:|---:|---:|---:|---:|
| Frozen A2 | 1.125 | 2 | 0.4375 | 42 | 33 |
| Continued A2 | 1.000 | 1 | 0.4375 | 73 | 55 |
| Continued A8 | 1.000 | 1 | 0.5000 | 38 | 28 |

Longer adaptation therefore did not rescue A8. The new inputs remained
material, but neither continued arm improved held-out progression, and A8 lost
local combat and item competence relative to frozen A2. The predeclared result
is `retain_a2_after_long_horizon`; architecture scale and observation richness
are not the next isolated variables.

One unresolved execution mismatch remains: PPO trains by sampling a categorical
policy with an entropy bonus, while promotion evaluation has used per-state
argmax. Stochastic training reached Zone 1 Floor 3, whereas deterministic
held-out evaluation reached only Zone 1 Floor 2. EXP-0009 therefore freezes A2
and A8 checkpoints and compares argmax with two turn-keyed, reproducible
stochastic sample streams on the same 24 unseen game seeds. It passes only with
repeatable Zone 2 progress across multiple seeds. The wider environment audit
and next causal interventions are recorded in `docs/rl-environment-audit.md`.

### EXP-0009 partial execution result and learning-integrity correction

The three completed frozen-A2 trials confirmed a large execution-mode effect
without satisfying the Zone 2 gate. Argmax remained on Zone 1 Floor 1. Both
turn-keyed stochastic streams reached Floor 2 on the same seeds `57004`,
`57005`, and `57009`; policy stream `91002` also reached Floor 3 on seed
`57019`, with one recorded stair discovery and exit. No trial reached Zone 2.
The active continuation evaluation was mistakenly interrupted after its owned
game workers were queried under the wrong executable name. There is no evidence
of a launcher failure. The completed reports are retained as partial diagnostic
evidence rather than promoted as a complete EXP-0009 comparison.

The result rejects argmax collapse as the sole progression blocker. Before any
Reward V5 or architecture change, commit `d12b791` corrected four confounds:

- client turn limits now bootstrap the real final observation and stop GAE only
  across the reset boundary, without an abort/failure reward;
- player damage is scored from bounded before/after health loss rather than raw
  lethal-overkill magnitude;
- PPO records explained variance and value/return/advantage/reward/gradient
  scale diagnostics;
- floor transitions are attributed to stairs, trapdoors, or unknown sources.

The next training comparison must hold A2 and Reward V2 fixed so the effect of
these corrections is measurable. Reward V2 remains the historical baseline,
but its unbounded turn/revisit penalties remain a separately declared objective
defect; they should not be changed in the same causal run.

### EXP-0010 corrected A2 replication (predeclared)

EXP-0010 holds Architecture A2, Reward V2, the observation slice, action
contract, PPO hyperparameters, and eight-worker runtime fixed. It warm-starts
all compatible non-critic tensors from `runs/reward-v2-250k/final.pt`, resets
the critic and optimizer, and trains for 250,880 transitions with seed `39001`.
This is a full-horizon causal replication rather than a short smoke pilot.

Final and curve checkpoints are evaluated on unseen seeds `60001–60024` under
argmax and policy streams `92001`/`92002`. Passing requires Zone 2 on at least
three distinct seeds, with at least two successes repeated across both sampled
streams, plus valid controller and critic-health evidence. Failure earns a
separately predeclared floor curriculum; it does not justify an architecture or
reward change inside EXP-0010.

## Rules for future entries

Every new reward version must record:

1. the measured failure that motivates it;
2. the exact changed components and anti-exploit bounds;
3. the behavioral hypothesis before training;
4. training budget, seeds, checkpoint provenance, and evaluation protocol;
5. component totals and unshaped gameplay outcomes;
6. evidence for and against the hypothesis;
7. the decision to retain, revise, or retire the profile.

Do not introduce a new version solely because shaped return is low. Change the
reward only after held-out gameplay or component auditing identifies a specific
failure mode.

## Research references

- Ng, Harada, and Russell, [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Küttler et al., [The NetHack Learning Environment](https://arxiv.org/abs/2006.13760)
- Bukaty and Kanne, [Using Human Gameplay to Augment Reinforcement Learning Models for Crypt of the NecroDancer](https://bbukaty.github.io/CoNBot/report.pdf)
