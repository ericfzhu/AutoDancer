# Live RL environment design audit

Updated: 2026-08-27

## Objective and acceptance boundary

The current objective is not a high shaped return or a one-off descent. A promoted
agent must enter Zone 2 from a normal Bard All Zones reset on at least three
distinct unseen game seeds, with the direction reproduced by at least two fixed
stochastic policy-sampling streams. Infrastructure faults, trapdoor accidents,
and privileged curriculum starts do not count as final acceptance evidence.

The controller itself is qualified separately by the schema-10 one-million-turn
soak. This audit concerns the learning problem exposed through that controller.

## Evidence-backed defects

### 1. The task horizon is much longer than the useful policy-gradient horizon

PPO uses `gamma=0.99`, `gae_lambda=0.95`, 128-turn rollouts, and 32-turn
truncated recurrent chunks. The direct GAE trace multiplier is `0.9405^k`: about
0.140 at 32 turns, 0.0197 at 64, and 0.000389 at 128. A critic can propagate
value across rollout boundaries, but procedural seeds give it little exact-state
revisitation. The only observed Zone 1 Floor 3 entry took 2,024 turns. A flat
policy therefore receives extremely weak causal credit from a Zone 2 milestone
for decisions made early in a run.

This is a curriculum/hierarchy problem before it is a model-size problem.
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
shows why sparse goal tasks benefit from starts near the goal that expand
outward. Hierarchical imitation/RL similarly reduces exploration cost by
separating high-level subgoals from low-level control
([Le et al. 2018](https://proceedings.mlr.press/v80/le18a.html)).

### 2. Unbounded one-off training seeds undermine both learning and diagnosis

Earlier training sampled effectively unbounded procedural seeds. That maximizes
novelty but minimizes useful revisitation while the policy is still learning the
basic task. The repository now supports a checkpointed finite training seed
pool, but it has not yet been paired with a difficulty-aware sampler.

The correct split is a finite, diverse development pool plus permanently unseen
evaluation seeds. Procgen found substantial generalization gaps from limited
training levels and evaluated generalization on held-out levels
([Cobbe et al. 2020](https://proceedings.mlr.press/v119/cobbe20a.html)).
Prioritized Level Replay improves sample efficiency by revisiting levels with
high estimated learning potential rather than sampling every level uniformly
([Jiang et al. 2021](https://proceedings.mlr.press/v139/jiang21b.html)).

### 3. A2 has local recurrence but omits durable map and tactical state

The strongest current checkpoint is Architecture 2: a CNN/entity-attention
encoder, player and inventory encoders, a 512-unit LSTM, actor head, and critic
head (5,953,167 parameters). Its LSTM state is carried across turns, but gradients
are truncated every 32 turns and it receives no persistent map tensor. Expecting
that recurrent state to learn thousands of turns of mapping is an avoidable
informational disadvantage.

A2 is also a compatibility architecture: it ignores grid channels 11--28,
player features 16--20, inventory slots 9--13, and inventory fields 5--8. Thus
the current best policy cannot directly observe facing, beat delay/interval,
charge and shield direction, status duration, tell animation, explosive state,
music timing, or complete item cooldown/readiness information even though the
controller supplies them. This is especially relevant on a boss floor, where
reacting to multi-turn tells and phases is part of the task. A function-preserving
rich-observation adapter remains a justified boss-curriculum control even though
the earlier normal-start A8 experiment did not improve progression.

Architecture 8 added explicit terrain/reveal/visit/recency map channels, but the
map omits remembered static traps, containers, items, shrines, and other
landmarks that a player can remember after seeing them. Dynamic off-screen
enemies should remain excluded. Future map architectures should add only
human-observable persistent facts and ablate them in coherent groups.

### 4. Flat action choice conflates navigation and tactics

The same 11-way policy chooses exploration direction, staircase routing,
digging, combat, item use, and waiting. EXP-0012 tested a minimal navigation
prior, but it disengaged whenever *any* enemy was visible in the 21x21 window.
It activated on only 0.87% of deterministic turns, produced zero staircase
exits, and did not improve progress in two of three policy modes. It also used a
local least-visited rule rather than planning to a frontier.

The next navigation contract should use:

- local-threat gating, not any-visible-enemy gating;
- shortest paths to known stairs;
- shortest paths to informative frontiers when stairs are unknown;
- remembered static hazards and authoritative wall/dig outcomes;
- a tactical policy override only near immediate threats.

This is a small hybrid hierarchy: symbolic memory/path selection supplies a
macro-direction and the learned policy retains tactical control. Difficult
partially observed dungeon games provide strong evidence that hierarchy and
hybrid agents outperform merely scaling a flat neural policy
([NetHack Challenge report](https://proceedings.mlr.press/v176/hambro22a.html),
[NetHack Is Hard to Hack](https://arxiv.org/abs/2305.19240)).

### 5. The existing heuristic is not an expert teacher

The schema-10 qualification soak used `LiveExplorer` for one million live
transitions. Every worker's maximum was Zone 1 Floor 2. It is useful as a
navigation component, but its adjacent-enemy combat rule is too weak to provide
whole-policy demonstrations. Behavior cloning from it would likely inherit its
ceiling.

Behavior cloning remains viable once a competent teacher or human telemetry is
available. The game-specific CoNBot work found that human-play pretraining
reduced wall/corner loops and made exploration qualitatively more useful, while
training from scratch frequently timed out against walls
([project](https://bbukaty.github.io/CoNBot/),
[report](https://bbukaty.github.io/CoNBot/report.pdf)). Its need to train RL on
a fixed seed is additional evidence that our original unbounded-seed schedule
was too difficult for early skill acquisition.

### 6. Progress and competence metrics had causal ambiguities

Two defects were found after EXP-0012:

- Episode code independently maximized zone and floor. A trajectory through
  Zone 1 boss to Zone 2 Floor 1 could be represented as the impossible pair
  `(Zone 2, Floor 4)`. Progress tracking now keeps the deepest lexicographic
  zone/floor pair from one real state.
- Raw item pickup counts include repeated inventory swaps. In EXP-0012 seed
  62023, one looping episode recorded 2,485 pickups but only one newly acquired
  item type. Reports now expose repeated item transactions; selection should use
  unique item types and gameplay outcomes, not raw pickup traffic.

Floor entry must also be attributed. Trapdoor and unknown descents are useful
outcomes but do not prove navigation. Staircase discovery, staircase exit, and
discovery-to-exit conversion are the causal navigation metrics.

### 7. The default training horizon under-samples the declared seed pool

The live environment correctly implements client-side time-limit truncation and
the collector correctly bootstraps the critic from the real final observation.
The remaining problem is the default scale: training permits 10,000 turns per
episode, while controlled evaluations generally use 1,000--5,000. At 250,880
total transitions and eight workers, each slot contributes only 31,360 turns.
An indefinitely looping Bard policy can therefore complete only three full
10,000-turn episodes per slot, making a declared 32-seed training pool mostly
nominal rather than actually sampled.

Every curriculum experiment must declare an episode horizon appropriate to its
subtask and report distinct seeds seen, episode count, and truncation rate. The
boss-to-Zone-2 stage uses 1,000 turns. Earlier-floor stages may increase the
horizon only from measured completion-time quantiles. A time limit remains an
MDP truncation: it receives no death/abort reward and bootstraps value from the
real terminal observation. This is partial-episode bootstrapping as prescribed
for training-only time limits by
[Pardo et al. 2018](https://proceedings.mlr.press/v80/pardo18a.html).

### 8. Several shaping terms depend on hidden reward history

V2 decides whether a position, tile, staircase, or item type is new by consulting
episode/floor history inside `RewardTracker`. It can therefore assign different
rewards to the same apparent observation and action depending on facts that A2
cannot reconstruct exactly. Floor-level caps in later rewards create the same
issue. The game is already partially observable, but adding a second hidden
reward state makes value learning harder for reasons unrelated to gameplay.

Future reward work should either expose a compact, human-compatible reward-memory
state (for example visited-map state and acquired-item history), express guidance
as a proper potential on observable augmented state, or remove the history-based
term. Reward machines formalize this idea by compressing non-Markovian event
history into state that can be composed with the observation
([Bourel et al. 2023](https://proceedings.mlr.press/v206/bourel23a.html)).
Potential shaping is attractive where applicable because it preserves the task
policy rather than allowing intrinsic return to replace task progress
([Skalse et al. 2023](https://proceedings.mlr.press/v202/skalse23a.html)). The
current boss curriculum intentionally leaves V2 fixed; boss damage, death, and
Zone 2 entry are short-horizon signals and let us isolate representation first.

## Next controlled sequence

1. Add a curriculum-only real-game level start that performs sequential `GOTO`
   transitions after a normal seeded reset, clears reward/map/recurrent state
   after the final jump, and never awards progress for the jump itself.
2. Measure the frozen A2 policy from the Zone 1 boss across a fixed development
   seed set. This establishes whether local competence is already sufficient to
   produce any Zone 2 transitions when early-floor exploration is removed.
3. Train boss-to-Zone-2 first with an explicit 1,000-turn episode horizon, then
   expand starts backward to Floor 3, Floor 2, and finally normal Floor 1.
   Curriculum success terminates the subtask, while final evaluation always
   begins from a normal All Zones reset.
4. In parallel only after isolated frozen-policy evidence, test a frontier-based
   navigation prior v2. Do not combine it with curriculum until each component
   has an independently measured effect.
5. Add prioritized seed replay after the uniform finite-pool baseline, using
   learning-progress/TD-error evidence and retaining unseen evaluation seeds.

The promotion gate remains normal-start repeatable Zone 2 entry. Curriculum
success is training evidence, not acceptance evidence.
