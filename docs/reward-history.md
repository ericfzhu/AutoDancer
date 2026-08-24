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
