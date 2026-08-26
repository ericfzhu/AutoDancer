# RL environment audit: path to repeatable Zone 2

## Acceptance target

The next promoted agent must reach Zone 2 on multiple previously unseen game
seeds. A single training trajectory or one lucky sampled evaluation is evidence
of possibility, not success. Gameplay progress, rather than shaped return, is
the primary outcome.

## Evidence from the current system

- Stochastic PPO collection has reached Zone 1 Floor 3.
- Deterministic held-out evaluation has reached Zone 1 Floor 2.
- No recorded policy has reached Zone 2.
- The promoted A2 checkpoint still has policy entropy around 1.18 nats, while
  the A8 continuation is around 1.31. Training samples those distributions, but
  promotion evaluation has always selected per-state argmax.
- Extending both A2 and A8 to 250,880 transitions did not improve their final
  held-out mean progress beyond Zone 1 Floor 1. Richer A8 observations remained
  materially connected to the policy, so merely adding more capacity or more
  of the same training is not the next justified intervention.

## Design flaws and risks

### 1. Training and evaluation execute different policies

PPO optimizes a sampled categorical policy with an entropy bonus. Converting it
to argmax can collapse several useful near-tied actions into one repeated action.
This may explain why stochastic rollout progress exceeds deterministic progress.
EXP-0009 therefore evaluates frozen checkpoints under both argmax and two
turn-keyed stochastic sample streams. It changes no learned component.

### 2. Time-limit truncation is treated as terminal failure

The live environment distinguishes `terminated` and `truncated`, but the rollout
collector currently combines both into one `done` flag before GAE. That prevents
value bootstrapping across a client-imposed time limit. Gymnasium's step-API
guidance specifically distinguishes termination from truncation because most RL
algorithms should bootstrap after a truncation. A max-turn cutoff also currently
produces an `aborted` gameplay penalty, conflating an experiment boundary with a
game outcome. See the [Gymnasium terminated/truncated rationale](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).

### 3. Reward V2 has unbounded negative shaping

V2 charges `-0.005` every turn and an additional `-0.01` on revisits, while a
floor completion is only `+5`. At 3,000 or 10,000 turns, accumulated living cost
can dwarf the task milestone. Death can then stop future negative rewards,
creating an objective that is not cleanly aligned with survival and progression.
This shaping is not potential-based. Potential-based shaping is the general form
that preserves the optimal policy under the assumptions in
[Ng, Harada, and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

### 4. Sparse procedural exploration is not solved by recurrence alone

The agent has local memory and A8 has a persistent map, but neither guarantees
that rare useful frontier states will be revisited enough for PPO to learn from
them. [Go-Explore](https://arxiv.org/abs/1901.10995) addresses hard exploration
by explicitly remembering promising visited states, returning to them, and
exploring onward. Direct arbitrary-state restoration is not currently available
in the live game, so an applicable analogue would be a curriculum over normal
floor starts or a bounded episodic novelty objective, not a claim that the live
game can restore snapshots.

Intrinsic-motivation approaches such as
[Random Network Distillation](https://arxiv.org/abs/1810.12894) and the
[Intrinsic Curiosity Module](https://proceedings.mlr.press/v70/pathak17a.html)
show that prediction-based novelty can help sparse-reward exploration. They are
candidate interventions only after the execution-mode diagnostic and objective
corrections, because adding curiosity now would change multiple causal factors.

### 5. Evaluation uncertainty is understated

Most experiments use one training seed and report means across a modest game-seed
set without confidence intervals. Deep-RL comparisons can be unstable with few
runs; [RLiable](https://arxiv.org/abs/2108.13264) recommends interval estimates
and robust aggregate performance profiles. Procedural generalization also needs
held-out level seeds, as emphasized by the
[Procgen benchmark](https://arxiv.org/abs/2103.15332). AutoDancer already uses
fresh procedural seeds, but promotion should report per-seed successes and
bootstrap intervals rather than only a mean.

### 6. PPO details can dominate nominal architecture changes

Seemingly minor on-policy implementation choices can materially change results,
as documented by [What Matters in On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990).
Before another architecture search, we should verify truncation bootstrapping,
advantage construction, action masking, entropy, and evaluation semantics against
the intended task.

## Staged causal program

1. **EXP-0009 — execution calibration:** frozen A2/A8 checkpoints, 24 unseen
   seeds, argmax plus two reproducible stochastic sample streams. Pass only with
   Zone 2 on at least three distinct seeds and two successes repeated across both
   stochastic streams.
2. **Environment correction if EXP-0009 fails:** preserve bootstrapping on
   time-limit truncation and remove infrastructure/client-horizon penalties from
   gameplay reward. Add regression tests before retraining.
3. **Objective experiment:** make floor/zone milestones dominant; eliminate
   unbounded renewable penalties; keep task reward separate from bounded shaping.
   Use potential-based guidance where a valid potential exists.
4. **Exploration experiment:** only if objective correction still stalls, test
   one bounded episodic novelty mechanism or curriculum at a time.
5. **Promotion:** require repeatable Zone 2 on unseen seeds, controller-valid
   reports, multiple policy/training seeds, and uncertainty intervals. Do not
   select by shaped return.

This order is intentionally falsifiable: each stage either explains the current
failure or earns the right to introduce the next mechanism.

## Why a floor curriculum is likely necessary

The live transition budget is extremely small relative to published hard-
exploration game results. MiniHack's own examples use roughly 100,000 steps for
a basic 5×5 room and 1,000,000 for ordinary RLlib examples, while BeBold reports
120 million steps for its hardest procedurally generated MiniGrid tasks. These
are not directly comparable benchmarks, but they make one point clear: 250,880
live turns are a useful controlled pilot, not a generous budget for discovering
an entire multi-floor game strategy from scratch. See the
[MiniHack reference implementation](https://github.com/facebookresearch/minihack)
and [BeBold](https://arxiv.org/abs/2012.08621).

The strongest next exploration intervention is therefore a start-state
curriculum, not a larger renewable movement bonus. Reverse Curriculum
Generation shows that sparse goal tasks can become learnable when training
starts near a known goal and progressively expands to earlier states
([Florensa et al., 2017](https://proceedings.mlr.press/v78/florensa17a.html)).
MiniHack similarly treats complex roguelike competence as a collection of
controlled navigation and skill-acquisition environments rather than expecting
one sparse end-to-end task to teach every prerequisite at once
([MiniHack environment design](https://github.com/facebookresearch/minihack/blob/main/docs/getting-started/interface.md)).

AutoDancer already has a qualified, sequential `GOTO` mechanism that can move a
fresh seeded All Zones run to the next real generated level. A training-only
curriculum can use that mechanism to practice, in reverse order:

1. Zone 1 boss entry → Zone 2;
2. Zone 1 Floor 3 → boss entry;
3. Zone 1 Floor 2 → Floor 3;
4. normal Zone 1 Floor 1 → Floor 2 and onward.

This would still use the authoritative game, normal mechanics, Bard, seeded
procedural levels, and the same policy. Evaluation must always start from a
normal unassisted All Zones reset. Curriculum starts must be explicitly tagged
in telemetry and excluded from normal-start success statistics. Each stage
should retain some later-stage starts to prevent catastrophic forgetting, and
promotion still requires complete normal-start Zone 2 runs on unseen seeds.

A curriculum experiment should compare the same corrected reward and optimizer
under normal versus staged start distributions. That isolates whether sparse
credit assignment—not model capacity—is the limiting factor. Intrinsic novelty
is a later fallback because procedural worlds weaken global state-count methods;
episode-level exploration methods are more plausible than lifetime novelty when
every seed generates a new dungeon.
