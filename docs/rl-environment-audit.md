# RL environment audit: path to repeatable Zone 2

## Acceptance target

The next promoted agent must reach Zone 2 on multiple previously unseen game
seeds. A single training trajectory or one lucky sampled evaluation is evidence
of possibility, not success. Gameplay progress, rather than shaped return, is
the primary outcome.

For the first normal-start Zone 2 qualification, "multiple" means at least three
distinct successes from a seed bank fixed before evaluation, with the direction
reproduced by at least two independent policy/action samples. That bank must be
unfiltered with respect to generated boss type; assisted starts and a suite
selected to contain only a mastered boss are subskill evidence, not end-to-end
generalization. Reports must stratify attempts and successes by official boss
type so favorable seed composition is visible. Broader boss robustness remains
a subsequent requirement across all four Zone 1 boss types, but it must not be
conflated with the initial multi-seed normal-start milestone.

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

### 2. Resolved: time-limit truncation was treated as terminal failure

The previous collector combined `terminated` and `truncated` into one `done` flag
before GAE, preventing value bootstrap across a client-imposed time limit. The
2026-08-27 correction now stores true terminations and per-transition truncation
values separately: a time limit bootstraps its real final observation but stops
the GAE trace across reset. Client turn limits use `time_limit`, emit no failure
event, and receive no `aborted` gameplay penalty. This follows the
[Gymnasium terminated/truncated rationale](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).

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

### 7. The discount and advantage horizons are shorter than a floor

The current PPO configuration uses `gamma=0.99` and `gae_lambda=0.95`. A reward
500 turns in the future is multiplied by about `0.0066`; at 1,000 turns it is
about `0.000043`. The direct GAE trace decays even faster because
`gamma * lambda = 0.9405`: after 100 turns its weight is roughly `0.0022`.
Measured failed episodes routinely last 3,000–5,000 turns. Floor rewards provide
intermediate subgoals, but even those are often hundreds or thousands of actions
away from the early decisions that must discover them.

The effective-horizon approximation `1 / (1 - gamma)` makes the mismatch plain:
`gamma=0.99` expresses a horizon near 100 turns. Zhang et al. describe how
discounting effectively forgets consequences beyond this scale and why simply
raising gamma trades that bias for higher variance
([average-reward on-policy RL](https://proceedings.mlr.press/v139/zhang21q.html)).
Pohlen et al. used `gamma=0.999` plus stabilization specifically to extend Atari
planning by an order of magnitude
([Observe and Look Further](https://arxiv.org/abs/1805.11593)).

This does not justify changing gamma alone: a longer horizon can destabilize the
critic, and potential-based shaping must use the same discount as the learning
objective to retain its invariance guarantee. It does justify treating discount,
GAE lambda, rollout length, value normalization, and stage-level rewards as one
explicitly versioned objective/optimization block. A floor curriculum shortens
the experienced distance to each milestone without first taking on the variance
of an almost-undiscounted full-run critic.

### 8. A 32-turn recurrent gradient window cannot teach arbitrary long memory

The policy carries its LSTM state continuously and stores the correct initial
state for every recurrent PPO chunk, but gradients are truncated at the
32-transition chunk boundary. This is not equivalent to forgetting every 32
turns—the hidden state still persists during play—but it biases learning toward
dependencies that can be improved through short local gradients. Truncated BPTT
is computationally practical precisely because it cuts longer gradient paths;
the resulting estimator is biased toward short-term dependencies
([Tallec and Ollivier, 2017](https://arxiv.org/abs/1705.08209)) and adaptive
truncation work explicitly treats length as a gradient-bias control
([Aicher et al., 2020](https://proceedings.mlr.press/v115/aicher20a.html)).

This is a risk, not yet a demonstrated root cause. A8's explicit floor map
already removes the need for the LSTM to remember visited coordinates across
thousands of turns, and most combat timing is local. Before enlarging the LSTM
or replacing it with a transformer, test sequence lengths `32`, `64`, and `128`
on fixed logged recurrent batches and a delayed-information probe. Longer
sequences should advance only if they measurably improve retained-information
behavior or held-out progression at acceptable learner cost. Research on POMDPs
also cautions that recurrence itself does not make long histories easy to learn
([Memory Traces](https://proceedings.mlr.press/v267/eberhard25a.html)).

The live evaluator now exposes a controlled recurrent-state ablation. Its normal
`carry` mode is unchanged; `reset-every-step` supplies a fresh zero LSTM state to
every decision while retaining the same observation, previous action, previous
reward, policy sampling stream, seed, and action contract. Reports and lineage
record the selected mode so the two conditions cannot be silently pooled. A
matched-seed comparison therefore answers the narrow causal question "does this
checkpoint's behavior depend on accumulated recurrent state?" It does not by
itself identify whether a difference comes from useful memory, stale memory, or
shorter-gradient optimization, so any positive dependence must be followed by a
delayed-information probe or hidden-state drift measurement before changing the
architecture.

There is a second recurrent approximation in the current collector. Actor hidden
states persist across learner updates, so policy version `v+1` initially acts
from an LSTM state produced by version `v`. During the four PPO epochs, each
chunk likewise starts from its stored behavior-version state even after the
model has changed. Collection remains internally on-policy—the action and old
log-probability were produced from the same state and frozen policy—but the new
policy likelihood is evaluated under a potentially stale recurrent
representation. The lag is one PPO update rather than a long replay-buffer lag,
so its magnitude must be measured rather than assumed catastrophic.

Before a recurrent-horizon experiment, add hidden-state drift diagnostics: for
sampled chunks, no-gradient-unroll a preceding context under the current model
and report hidden-state relative error, action-distribution KL, and value change
against the stored-state evaluation. A versioned burn-in arm should retain a
short prefix (initially 16 or 32 turns), reconstruct the target chunk's state
without loss on that prefix, and apply gradients only to the existing target
window. The same mechanism should re-anchor each actor after a policy publish;
resetting hidden state to zero at arbitrary rollout boundaries would instead
fabricate episode boundaries. This adapts the stored-state plus burn-in response
to recurrent state staleness studied by
[R2D2](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf),
while recognizing that R2D2's off-policy replay setting has much larger lag than
AutoDancer's synchronous PPO updates.

### 9. Critic diagnostics are implemented; empirical health remains unqualified

Historical training recorded value loss, but not explained variance,
predicted-value mean or spread, return-target mean or spread, or
pre-normalization advantage statistics.
Value loss alone is scale-dependent: a small MSE does not establish that the
critic explains useful variation, and a large MSE does not reveal whether the
cause is a bad predictor or a corrupted/very large target.

The existing files make this more than a theoretical concern. Across 245 V2
updates, value loss had median `0.0203`, p95 `0.1762`, and maximum `0.9173`.
Across 215 long-horizon A2 updates it had median `0.0330`, p95 `0.1847`, and
maximum `0.3487`. Across the matched 215-update A8 continuation it had median
`0.0399` and p95 `0.2412`, but two updates jumped to `183.30` and `190.00`.
Those two batches also contained implausible `player_damage = -153` shaping
totals. None of the three metric streams records explained variance or target
scale, so the current evidence cannot tell how long the critic was useful after
those shocks.

The 2026-08-27 learner now logs explained variance; value, return, raw-advantage,
and reward mean/spread/maximum magnitude; and total pre-clipping gradient norm.
The next corrected run must establish whether these remain healthy before gamma,
value coefficient, architecture, or return scaling changes. Adaptive target
normalization such as PopArt is a candidate only if these diagnostics demonstrate
a scale problem. PopArt is designed to normalize changing targets
while preserving the network's unnormalized outputs
([van Hasselt et al., 2016](https://arxiv.org/abs/1602.07714)).

### 10. Resolved: player-damage shaping used attack damage rather than health lost

The Lua bridge copies `ev.damage` from `objectDealDamage` directly
into a `player_damage` event, and the Python reward tracker multiplies that raw
amount by the configured per-point penalty. SYNCHRONY documents this field as
the amount of damage dealt; it does not say that it is clamped to the victim's
remaining health. A lethal attack can therefore report substantial overkill
even though Bard could only lose the health that remained. See the
[SYNCHRONY object event contract](https://vortexbuffer.com/synchrony/docs/events/object/#eventobjectdealdamage).

The A8 continuation contains two independent 1,024-transition batches with
exactly `-153` player-damage reward. At `-0.15` per point that means 1,020 raw
damage points, despite only five or six deaths in each batch. This is impossible
as actual health lost and proves reward contamination. It also explains the
critic-loss spikes in the same updates.

The 2026-08-27 live adapter consolidates each turn's raw player-damage events and
replaces their reward amount with authoritative before/after health loss bounded
by pre-turn health. Raw damage and event count remain diagnostic metadata. Tests
cover lethal overkill and multiple events; the qualified controller already
covers ordinary damage, death, and reset. Historical affected runs remain valid
controller evidence but are not clean reward-policy comparisons.

### 11. Resolved: `item_pickups` previously counted currency pickups

Historically, the only Lua site that emitted an `item_collected` event was the
`objectCurrency` hook. It emitted once when Bard picked up a currency entity and
stored the currency difference as the event amount. The deterministic evaluator
then incremented `item_pickups` once for every such event and called the amount
`item_value`. No equipment-pickup event fed that metric.

Reward state does independently infer newly seen inventory type IDs, so the
policy has received some equipment-related shaping. The evaluation outcome used
by several historical promotion gates, however, measures coin-pile collection,
not weapons, armor, consumables, or equipment competence. Those historical
numbers must be relabeled rather than interpreted as general item pickups.

The 2026-08-27 correction makes Lua emit `currency_collected` only from the
currency hook. The live adapter derives `item_collected` from positive inventory
deltas, and evaluation separately reports item units, unique acquired item
types, currency transactions, and currency value. Historical item metrics remain
mislabeled, but new item-based promotion gates are semantically distinct.

## Staged causal program

1. **EXP-0009 — partial execution calibration:** the completed A2-frozen trials
   reproduced Floor 2 under both sampled streams and reached Floor 3 once, but no
   Zone 2. The active continuation trial was mistakenly interrupted during
   process diagnosis, so partial results are preserved without claiming either
   a complete checkpoint comparison or a launcher failure.
2. **Environment correction — qualified and replicated:** truncation bootstrap,
   client-horizon reward semantics, actual-health damage, critic diagnostics, and
   stair/trapdoor descent attribution, and currency/inventory telemetry are
   implemented, regression-tested, and exercised by EXP-0010 with zero
   controller faults. The critic stayed finite, but progression still regressed.
3. **Action-efficiency experiment:** EXP-0010 exposed thousands of repeated
   authoritative `wall_attempt` outcomes. Test stateful known-invalid masking
   before changing reward, architecture, or task distribution.
4. **Objective experiment:** make floor/zone milestones dominant; eliminate
   unbounded renewable penalties; keep task reward separate from bounded shaping.
   Use potential-based guidance where a valid potential exists.
5. **Exploration experiment:** only if objective correction still stalls, test
   one bounded episodic novelty mechanism or curriculum at a time.
6. **Promotion:** require repeatable Zone 2 on unseen seeds, controller-valid
   reports, multiple policy/training seeds, and uncertainty intervals. Do not
   select by shaped return.

This order is intentionally falsifiable: each stage either explains the current
failure or earns the right to introduce the next mechanism.

## EXP-0011 result: invalid actions fixed, valid navigation cycles remain

All six 24-seed reports completed without controller faults. The exact-state
wall memory reduced aggregate wall attempts from `30,373 / 66,644` turns
(`45.57%`) to `1,047 / 68,503` (`1.53%`), a `96.65%` reduction. It improved
action efficiency in all three policy modes, retained `243 / 232` kills and
`29 / 30` item pickups, and proved that combat and digging remained available.

The intervention nevertheless failed its predeclared zero-shot deployment
gate. Step limits remained exactly `10 / 72`, aggregate mean floor progress
fell from `1.0694` to `1.0556`, and neither arm exceeded Zone 1 Floor 2. The
deterministic traces make the remaining defect concrete: after a wall direction
was suppressed, A2 often selected a sequence of valid moves among only a few
positions, or an ambiguous unchanged-direction action, until the same 5,000-
turn cap. Invalid-action masking therefore repairs wasted actions but does not
supply a navigation objective or persistent traversal state to a frozen policy.

EXP-0011 is rejected for promotion and `current-11` remains the frozen-policy
contract. The next isolated diagnostic targets the newly demonstrated valid-
cycle problem using the already available human-equivalent floor memory. A
map-guided navigation prior must disengage during combat, leave successful
digging and interactions available, prefer known stairs once discovered, and
otherwise choose among least-visited reachable frontiers. This is explicitly a
hybrid planning/action-prior experiment, not an invalid-action claim or a reward
change. If it cannot improve unseen-seed floor progress, proceed to the declared
finite seed distribution or floor curriculum rather than adding stronger
heuristic masking post hoc.

### Boss calibration exposes an action-contract diagnostic gap

The nine assisted Death Metal calibration episodes made `723` wall attempts in
`3,950` turns (`18.3%`). Every attempt was counted as a newly learned exact
wall-state signature. That does **not** prove 723 cache failures: a policy can
legitimately probe different walls and directions from different positions. The
aggregate unique-position counts only prove a lower bound of 15 signature
reopenings (one position has at most four physical directional probes), while
the existing report discards the per-turn keys needed to identify the rest.

Do not alter `map-navigation-prior-v1` during EXP-0016. Before declaring a v2,
record two identities for every authoritative wall attempt: the current full
invalidation signature and a diagnostic physical key containing level, player
position, direction, target terrain/object/actor identity, and relevant digging
equipment. Report first physical probes, exact repeats, reopened physical probes,
and which dynamic signature fields changed. A v2 may remove facing or beat-state
fields only when traces prove they reopen physically identical, still-invalid
actions; confusion, charge, target occupancy, terrain, and equipment changes
must continue to reopen the direction. This turns the 18.3% waste signal into a
causal mask test rather than another heuristic adjustment.

## EXP-0010 evidence and newly demonstrated design flaws

The corrected A2 replication completed 250,880 transitions and nine held-out
reports without a controller recovery. It rules out the controller, false
truncation terminals, overkill damage, and a numerically broken critic as the
immediate explanation. The strongest checkpoint was at 61,440 transitions and
reached Floor 2 on four distinct unseen seeds, but later updates lost that
competence. At 122,880 transitions the deterministic policy spent `84.97%` of
turns without changing position and timed out on `75%` of seeds. At 250,880 it
spent `85.97%` stationary and reached no Floor 2. No report reached Floor 3 or
Zone 2.

Three environment-design problems now have direct or strong supporting
evidence:

1. **Known no-op actions remain advertised.** The live protocol correctly
   reports all engine-legal directions, but a policy can repeatedly select a
   wall that it has already proved it cannot dig. This consumed thousands of
   expensive on-policy transitions. Invalid-action masking has a policy-gradient
   justification and is empirically important when invalid choices are common
   ([Huang and Ontañón, 2020](https://arxiv.org/abs/2006.14171)). EXP-0011 uses
   a conservative history-based mask: the first attempt is always allowed, and
   only the exact observed wall/position/control/inventory signature is cached.
   Any state change reopens the action; combat and successful digging are never
   cached.
2. **The training-level distribution is effectively infinite before mastery.**
   Every reset draws another procedural seed, so the critic rarely revisits the
   same long-horizon task while rewards can still propagate backward. Procgen
   separates finite training levels from unseen evaluation and finds that broad
   procedural diversity is essential for eventual generalization
   ([Cobbe et al., 2020](https://proceedings.mlr.press/v119/cobbe20a.html)).
   Prioritized Level Replay improves sample efficiency by revisiting levels with
   high estimated learning potential rather than sampling uniformly forever
   ([Jiang et al., 2021](https://proceedings.mlr.press/v139/jiang21b.html)). A
   fixed training pool with progress/TD-error prioritization is the next seed
   policy once action efficiency is repaired; evaluation remains on unseen
   normal starts.
3. **The useful credit horizon is far shorter than a floor.** With `gamma=0.99`,
   a milestone 500 turns away has a direct discount of roughly `0.0066`; with
   GAE lambda `0.95`, long direct traces decay faster still. Bootstrapping can
   propagate value over repeated visits, but unlimited fresh seeds weaken that
   mechanism. Reverse start-state curricula are a principled response to sparse
   goals, expanding starts away from a known success state as competence grows
   ([Florensa et al., 2017](https://proceedings.mlr.press/v78/florensa17a.html)).

These findings also put the 250,880-turn budget in context. NetHack, a related
partially observed roguelike benchmark, defines staircase and other subskills
instead of treating full-game completion as the first learnable target, and
published implementations validate at vastly larger interaction budgets
([NetHack Learning Environment](https://arxiv.org/abs/2006.13760)). AutoDancer
should preserve unseen-seed end-to-end evaluation while using finite replay and
assisted subskill starts during training; those starts must be tagged and must
never count as normal-start success.

The opt-in `--training-seed-pool start-end` training interface now implements
the first controlled step: uniform sampling from a declared finite pool with
independent worker streams. The exact next-draw state is stored in checkpoints
and restored across resume and periodic evaluation. The default remains the
historical unbounded distribution, and no experiment uses the finite pool until
its lineage specification declares that changed block. Uniform replay isolates
the effect of repeated exposure; TD-error prioritization remains a later,
separately versioned intervention rather than being bundled into the first test.

`map-navigation-prior-v2` now implements the narrower follow-up authorized by the
EXP-0012 failure analysis, but it is not promoted by implementation alone. It
routes over Bard's remembered revealed terrain toward known stairs, otherwise
toward the nearest reachable reveal frontier. Unlike v1, a merely visible distant
enemy does not disable the route. Control returns to the learned policy for an
enemy within two Manhattan steps, any visible dragon or boss, an active tell or
explosive state, and the entire boss floor. Each worker also remembers observed
trap coordinates for the current floor and excludes them from strategic routes;
the memory clears on a natural floor transition as well as reset. If no complete
route is known, the conservative v1 least-visited fallback remains available.

The prior constrains the action mask before inference, so PPO still records the
probability of the action under the exact effective policy rather than replacing
an already sampled action. Reports expose activation rate, masked directions, and
maximum remembered hazards. A future frozen-policy ablation must compare v2 with
v1 and the unmodified contract on identical unseen normal starts. Promotion
requires more staircase exits and Floor 2 entries in at least two policy streams,
lower step-limit/loop behavior, and no material loss of combat survival when the
local-threat handoff activates. It must not be selected merely because it moves
more often.

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

## EXP-0014 postmortem: environment defects found before the next curriculum

The completed boss experiment proved that the controller can collect stable
boss-local trajectories, but not that the learning environment presents a
sound Markov approximation or a learnable start distribution. Inspection
found the following concrete defects:

1. **False current visibility.** The memory adapter treated both historical
   `VISIBILITY=1` cells and periodic full minimap snapshots as currently
   visible. Every 32 turns this injected a floor-wide “visible now” pulse,
   creating a hidden clock feature unrelated to game state. Only local
   `VISIBILITY=2` cells may now update current sight; minimap snapshots add
   historical knowledge only.
2. **Clipped spatial memory.** A 65x65 player-centred tensor cannot retain a
   65x65 floor when Bard moves between opposite edges. The minimum lossless
   player-centred span is `2*65-1 = 129`; the policy memory now uses that span,
   while Lua's compact 65x65 snapshot is anchored to the real level bounds.
3. **Aliased boss objectives.** Lua now exposes the official `Boss.Type`, but
   Python previously overwrote A2's existing task feature with zero. A2 can now
   distinguish King Conga, Death Metal, Deep Blues, and Coral Riff without an
   architecture change. The Synchrony API defines these stable enum values and
   distinguishes objective `boss` entities from `bossAdd` entities
   ([Boss API](https://vortexbuffer.com/synchrony/docs/modules/necro.game.level.Boss/),
   [boss components](https://vortexbuffer.com/synchrony/docs/components/necro.game.data.component.character.BossComponents/)).
4. **Objective-blind diagnostics.** Damage and kills are now partitioned into
   boss, boss-add, and generic-enemy totals. A curriculum cannot pass because
   it fights renewable adds while never interacting with the objective.
5. **Untracked task horizon.** `max_turns`, PPO `gamma`, and GAE lambda are now
   explicit checkpoint/lineage parameters. A resume cannot silently change the
   episode horizon, and future discount experiments are declarable rather than
   hard-coded.
6. **No successful start states.** EXP-0014 mixed four full-strength bosses and
   observed zero successes. Reverse Curriculum Generation specifically expands
   from starts with an intermediate success probability and retains older
   starts to avoid forgetting
   ([Florensa et al., 2017](https://proceedings.mlr.press/v78/florensa17a.html)).
   The next stage therefore uses a tagged, training-only health profile on one
   boss, then reduces assistance after competence is measured. This is not
   eligible as normal-start evaluation evidence.

The discount configuration remains a later isolated variable. At `gamma=.99`,
direct credit is about `.0066` after 500 turns and `.000043` after 1,000 turns;
with GAE lambda `.95`, the one-rollout trace decays faster. Reverse starts first
move successes inside the useful credit horizon. Gamma changes, hierarchical
options/reward machines, demonstrations, and prioritized level replay remain
separate follow-ups if the calibrated local task still fails.

## Mechanisms checked and currently sound

- **Mid-chunk recurrent resets:** `episode_starts` is stored for every
  transition, and every model implementation zeroes LSTM state immediately
  before an episode-start step during PPO replay. Episodes ending inside a
  32-turn chunk therefore do not leak hidden state into the reset episode.
- **Floor map lifetime:** explicit map memory automatically clears when the
  observed zone/floor identity changes and retains only revealed terrain plus
  Bard's own traversal state.
- **Potential/PPO configuration guard:** training refuses to start when the
  reward potential's discount differs from PPO gamma. Any future gamma arm must
  supply a matching reward specification.
- **Asynchronous action randomness:** collection samples are keyed by training
  seed, worker slot, frozen policy version, and fragment turn. Worker timing
  cannot silently change which random action quantile a transition receives.
- **Episode identity and floor observations:** the schema-10 qualification
  verifies reset/run/seed identity and proves that the first action after a
  level transition consumes the new floor's observation rather than the prior
  floor state.
- **Rich-input connectivity:** A8 representation probes show that map,
  tactical grid, extended player, and inventory inputs all materially affect
  the network. A8's failure is not explained by disconnected inputs, though it
  may still reflect how those inputs are optimized or used.

## EXP-0016 acquisition audit and limits

The corrected frozen A2 policy completed assisted Death Metal once in nine
seeds, entering Zone 2 on seed 64018 in 128 turns. That establishes possibility
and gives PPO at least one real successful trajectory, but it is a noisy estimate
of the start distribution's success probability. Reverse Curriculum Generation
usually estimates a start using multiple trajectories and retains starts in an
intermediate-success band rather than treating one sample as ground truth
([Florensa et al., 2017](https://proceedings.mlr.press/v78/florensa17a.html)).
EXP-0016 therefore uses three independent optimizer/action RNG trials and a
held-out boss-seed suite. It is an acquisition test, not evidence that the
assistance level is correctly calibrated in advance.

The calibration evidence is narrower still: seed 64018 succeeded only under the
single stochastic policy stream `97001` with `map-navigation-prior-v1`; the
same nine seeds had zero completions under the current and known-wall contracts,
and no second map-prior policy stream or deterministic map-prior replication was
run. EXP-0016's three training trials therefore also test whether positive
trajectories recur under new action streams. If all three receive no success,
the correct conclusion is that this start distribution was not reliably
calibrated—not that recurrent PPO has disproved boss-skill learning. The next
start must move closer to a reproducible intermediate-success region or use a
recorded successful trace, then be calibrated with at least three rollouts per
`(seed, assistance)` identity before another acquisition experiment.

The following properties were inspected before starting EXP-0016:

1. **The navigation prior remains on-policy.** It changes the action mask before
   categorical sampling. Inference returns the selected action and its log
   probability under that exact constrained distribution, and PPO stores both.
   There is no post-sampling action replacement or stale likelihood ratio.
   Assistance removal can still create a deployment distribution shift, so each
   later curriculum stage must explicitly test reduced action assistance rather
   than silently evaluating an unconstrained policy.
2. **Nine boss seeds are a skill-acquisition pool, not a generalization pool.**
   Repetition is intentional while sparse success is being acquired, and the
   twelve held-out Death Metal seeds prevent training-set completion from passing
   the experiment. They do not establish broad procedural robustness. Procgen
   found that diverse level distributions are essential and reported that its
   environments commonly needed hundreds of training levels to generalize
   ([Cobbe et al., 2020](https://proceedings.mlr.press/v119/cobbe20a.html)).
   After local competence, the boss pool must expand substantially; later normal-
   floor training should use hundreds of seeds or an effectively unbounded stream
   while preserving a fixed unseen test bank.
3. **Uniform replay wastes samples once seed difficulty separates.** A seed the
   policy always solves and one it never solves supply less learning signal than
   a seed near the competence boundary. Prioritized Level Replay uses estimated
   learning potential to revisit informative procedural levels and improves both
   sample efficiency and generalization
   ([Jiang et al., 2021](https://proceedings.mlr.press/v139/jiang21b.html)).
   EXP-0016 deliberately keeps uniform sampling to avoid changing two mechanisms
   at once. If acquisition is real, assistance reduction should add a versioned
   competence-based sampler rather than continue uniform nine-seed replay.
4. **Terminal credit is short relative to the successful trajectory.** With
   `gamma=.99` and `lambda=.95`, the direct GAE coefficient is `.9405`; terminal
   credit 128 turns earlier is only about `.00039`. The boss strike, boss-kill
   event, and zone transition are locally adjacent, so the assisted subtask can
   still teach combat decisions, while the navigation prior supplies arena
   traversal. This configuration cannot by itself assign a Zone 2 reward to
   decisions thousands of turns earlier on Floor 1. Backward start expansion,
   retained intermediate milestones, or a separately tested longer-horizon/
   hierarchical learner remains necessary.
5. **The explicit map removes an informational disadvantage but not a planning
   objective.** Corrected 129x129 memory can losslessly retain a 65x65 floor,
   distinguishes current sight from remembered cells, and records visits. A2
   does not encode that tensor; its temporary navigation prior consumes it
   outside the learned network. A future end-to-end policy must either promote a
   map-aware architecture without losing A2 competence or treat navigation as a
   separately learned option. Merely enlarging the LSTM does not expose the map.
6. **Shaping and task success remain distinct.** Reward V4A bounds exploration,
   combat, and item credit below the floor milestone and uses potential-based
   stair guidance. The completion decision ignores shaped return. This preserves
   the purpose of potential shaping—accelerating learning without redefining the
   task—described by
   [Ng, Harada, and Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

If EXP-0016 passes, the next causal test is assistance reduction on the same boss:
player health 20 → 10 → 6 → normal, while retaining a controlled fraction of the
last mastered starts to detect forgetting. Only then should the start move to
Floor 3, Floor 2, and normal Floor 1. If it fails, the correct response is not a
new arbitrary reward weight: either create nearer real-game good starts, obtain
reproducible successful action traces for a separately declared demonstration
bootstrap, or introduce a boss-local hierarchical option. The normal-start goal
remains repeatable Zone 2 entry on multiple unseen seeds.

### Correction: assistance must be reduced along both health dimensions

The current assisted profiles all set the boss objective to one health and vary
only Bard's starting health (`boss1hp-player20`, `boss1hp-player10`, and
`boss1hp-player6`). Consequently, the previously proposed player-health-only
sequence is incomplete. It can teach arena traversal and a finishing hit, but
it never exposes the policy to Death Metal's earlier multi-hit behavior before
the final discontinuous jump to an unmodified boss. Passing EXP-0016 therefore
proves acquisition of a one-hit finishing subskill, not full boss competence.

After the current hash-qualified experiment, the bridge should support a
versioned two-dimensional assistance ladder. Each stage must record the requested
and telemetry-observed initial player health and boss-objective health. Boss
health should progress monotonically from one hit through intermediate observed
health values to the unmodified live value; player health should separately move
from 20 through lower assisted values to the unmodified live value. The exact
ladder must be calibrated from live telemetry rather than assuming an undocumented
maximum or changing both dimensions at once. Every assisted transition remains
excluded from normal-start success evidence.

Curriculum movement also needs a sampling contract rather than a single pass/fail
checkpoint. Reverse Curriculum Generation estimates success probabilities from
multiple trajectories per start and mixes previously useful starts back into
training to prevent forgetting. For AutoDancer, the atomic start identity is
`(boss type, game seed, assistance profile)`, not merely a game seed. Before a
stage is promoted, each candidate identity should receive at least three
independent policy rollouts; training should concentrate on identities with
intermediate success probability while reserving a predeclared nonzero fraction
for mastered earlier stages. The initial replay fraction should be fixed before
seeing held-out results (25 percent is a reasonable first isolated arm), logged
per rollout, and evaluated against a no-replay control if sample budget permits.
This follows the intermediate-difficulty and old-start replay mechanisms in
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html),
while later seed selection can use learning-potential and staleness signals from
[Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html).

The resulting backward path is therefore: acquire one-hit completion; reduce
player-health assistance; raise boss health in calibrated increments while
replaying mastered starts; demonstrate unassisted boss completion on multiple
unseen boss seeds; then move the start backward to Floor 3, Floor 2, and normal
Floor 1. A stage may advance only on gameplay success, never shaped return.

### Curriculum arrival-state mismatch

The current later-floor reset is not a fabricated board. Python starts a normal
seeded All Zones run and asks the game to advance one sequential level at a time
with `GameSession.nextLevel(0)`. This preserves the game's run state and produces
a legal later floor with Bard's default carried inventory. It is therefore a
useful good-start distribution, but it is only a narrow subset of the states in
which a normal policy will actually arrive. A played-through arrival may have
lost health, acquired or consumed items, and accumulated a nonzero LSTM state.
Direct curriculum episodes instead begin with restored/default run resources and
a fresh recurrent state. A policy can consequently pass a direct Floor 3 or boss
start while depending on conditions that are uncommon after Floors 1 and 2.

This is the same general issue addressed by reverse-curriculum methods when they
expand from feasible nearby states rather than arbitrary convenient states.
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
uses reachable starts near already-solvable states, while
[Jump-Start RL](https://proceedings.mlr.press/v202/uchendu23a.html) lets a guide
policy generate the exploration policy's starting-state distribution. The exact
mechanisms do not transfer directly to a live, reset-limited game, but the design
principle does: training starts should converge toward the state distribution
created by the policy that must use the learned skill.

AutoDancer must therefore qualify each backward expansion in two ways:

1. fixed later-floor starts measure whether the isolated subskill was retained;
2. starts from the immediately preceding floor must cross the boundary naturally
   and complete the same downstream target with carried health, inventory, and
   recurrent context.

The second result is the promotion evidence. Direct-start success alone only
authorizes the preceding-floor stage. Curriculum reports must record the arrival
health, inventory signature, and whether recurrent state was fresh or carried,
then compare those distributions with natural boundary crossings. Before the
first floor-stage experiment, deterministic evaluation must also compare carried
LSTM state against an explicit floor-boundary reset. If resetting does not reduce
gameplay success, resetting at every floor boundary is preferable because it
removes stale-history mismatch while current health, inventory, level identity,
and floor memory remain explicitly observed. If carried state helps, training
must include natural boundary crossings rather than assuming zero-state starts
will transfer.

This reset-distribution prerequisite is now implemented as
`mixed-curriculum-replay-v1`. The live environment validates Gymnasium reset
`options`; every episode carries an identified start level, target level, and
assistance profile; and each actor slot draws from an independent deterministic
weighted stream. Checkpoints preserve the exact sampler RNG, draw counts,
selection counts, and outcome counts. Rollout episode metadata retains the full
reset specification. An infrastructure replacement replays the failed episode's
same game seed and reset specification without counting the fault as an outcome
or silently changing the training distribution. Fixed-profile evaluation bypasses
the sampler. The distribution stream is keyed by training seed and worker slot;
it deliberately does not depend on actor timing or policy-update boundaries.
Running separate fixed-profile jobs and warm-starting between them remains
non-equivalent because it provides no within-update protection against forgetting.

EXP-0016 supplied the missing acquisition evidence: three independent assisted
Death Metal trials produced 65 Zone 2 entries in 108 held-out episodes (60.2
percent), and every one of 12 unseen boss seeds succeeded in at least one policy
trial with exact controller validity and zero restarts. Seed 68002 was selected
by the predeclared ordering (25/36, 69.4 percent). This proves only the
`boss1hp-player20` subskill. It authorizes a separately declared player-health
reduction stage with mastered-start replay; it does not promote the policy as a
normal-start Zone 2 agent.

Assistance application also needs stronger evidence and narrower targeting. The
current Lua profile runs immediately before the first assisted observation, so a
successful GOTO record proves that its assertions did not fail. It does not,
however, report the pre/post health values or number and kind of affected
entities. Moreover, `boss1hp-*` currently sets both `boss` and `bossAdd`
entities to one health. That is harmless for the recorded Death Metal calibration
(it recorded no boss-add damage or kills), but it would silently simplify
mechanics-critical adds when the curriculum expands to other boss types. A
future profile acknowledgement must include observed player health and a stable
summary of each affected objective's kind, current health, and maximum health.
Boss-objective health and boss-add health must become separate assistance
dimensions, defaulting to unmodified adds unless an experiment explicitly
declares otherwise. The learner must use the observed applied values—not merely
the requested profile label—for curriculum identity and lineage validation.

The observation audit introduces a stage-specific architecture requirement.
A2 sees local actor identity, current/max health, items, traps, and the boss type,
so the one-hit acquisition task is observable. It does not encode the already
available facing, beat delay/interval, frozen/confused timers, charge and shield
directions, tell animation, or explosive state. Those signals become important
when a boss survives long enough to expose its timing phases. A8 can read them,
but it also introduced persistent map, extended player, and expanded inventory
paths; its broad-task rejection does not isolate the value of combat timing.

If EXP-0016 acquires the finishing subskill but the first boss-health increase
does not transfer, the next architecture experiment should compare A2 against a
function-identical, zero-projected **tactical-only** residual on the same boss,
seed pool, assistance mixture, reward, and budget. Keep the A2 base frozen during
the declared adapter warm-up and require representation influence plus held-out
multi-hit completion—not activity or shaped return. Do not include persistent
map inputs in that boss-local test. Map-aware navigation should remain a separate
floor-stage or option-policy question so combat information and strategic memory
are not confounded again.

Death Metal makes this assistance reduction discontinuous rather than scalar.
The live build's boss starts with nine health, while the documented fight changes
behaviour at health bands 9–7, 6–5, 4–3, and 2–1: shield-relative attacks in the
first band, teleporting and timed summons in the middle bands, then a fast
fireball/chase phase at the end. The current one-health curriculum trains only
that final band and can bypass the shield, teleport, and summon decisions
entirely. This explains why one-hit completion cannot establish general boss
competence and why an immediate one-to-nine transfer is an unnecessarily severe
distribution shift. The phase structure is described by the community-maintained
[Death Metal mechanics reference](https://crypt-of-the-necrodancer.fandom.com/wiki/Death_Metal),
and the shield behaviour is corroborated by the game's published
[release notes](https://www.gogdb.org/product/1432297044/releasenotes).

After normal player health transfers, calibrate frozen performance at boss-health
caps 2, 4, 6, and 9 on the same held-out Death Metal seeds. Select the hardest
cap with a nonzero but sub-mastered success rate, then train it with replay from
the immediately easier cap. Promotion at each boundary requires success in the
newly introduced phase, not merely total boss damage or ordinary-enemy kills.
Profiles must leave boss adds unmodified and report observed boss health so the
experiment cannot silently skip a phase. If the first newly introduced phase
fails despite adequate successful examples at the easier cap, that is the trigger
for the tactical-only representation test described above—not for changing the
reward at the same time.

### Reachable boss states, natural prefixes, and boss coverage

The phase-aligned health sequence above is a hypothesis, not yet a valid reset
distribution. `boss1hp-*` assigns the health component directly after level load.
That operation does not prove that the rest of the entity state matches a fight
which reached the same health through legal damage. In particular, an event-driven
phase flag, summon counter, shield state, position, surrounding enemies, and the
player's recurrent history can all differ. The learner also sees the conspicuous
pair `current_health=1, max_health=9`. A policy trained on that pair can specialize
to finishing the fight without learning how to create it.

This is a stronger form of the arrival-state mismatch: the curriculum may place
the agent in a state outside the transition distribution of the target task.
[Reverse Curriculum Generation](https://proceedings.mlr.press/v78/florensa17a.html)
expands from feasible nearby states, and
[Jump-Start RL](https://proceedings.mlr.press/v202/uchendu23a.html) uses a guide
policy to produce the state from which the learning policy takes over. The next
boss-health experiment must therefore precede training with a live equivalence
probe:

1. on matched seeds, record a directly mutated health-cap state;
2. separately reach the same health by legal actions and boss damage;
3. compare boss/player positions, health, status and timing fields, boss adds,
   action masks, and several subsequent no-op-independent transition responses;
4. reject direct mutation if any policy-relevant field or response differs.

If direct mutation fails, replace health caps with a **natural-prefix handoff**.
A qualified guide executes legal actions from the ordinary boss start until the
declared health boundary, then the PPO actor takes control of that exact running
game. Record the complete guide prefix but exclude it from PPO reward and return.
Compare a fresh learner LSTM state against a state warmed by feeding the prefix's
observations through the learner without selecting its actions. Only downstream
learner transitions enter the rollout. This preserves the game's transition
dynamics and progressively shortens the guide prefix as competence improves.

Death Metal is also only one quarter of the ordinary Zone 1 boss task. Mastering
it can validate the curriculum mechanism, but it cannot establish robust normal
All Zones competence. Before normal-start promotion, repeat isolated acquisition
and assistance removal for King Conga, Deep Blues, Coral Riff, and Death Metal,
then evaluate a boss-stratified set with at least three unseen seeds per type.
Report both macro-average success across boss types and the worst type; do not
select only Death Metal seeds for the final Zone 2 claim. Boss adds must remain
unmodified unless a separately declared arm tests add assistance.

Procedural diversity is a second generalization boundary. The
[Procgen study](https://proceedings.mlr.press/v119/cobbe20a.html) found diverse
training environments essential for generalization, while the directly relevant
[CoNBot report](https://bbukaty.github.io/CoNBot/report.pdf) found that fresh
random NecroDancer levels were much harder for RL than a fixed seed. Its useful
behavior came from pretraining on 100 human Floor 1 sessions; frame history also
reduced loops, and the authors explicitly proposed recurrent memory for longer
planning. AutoDancer already has structured observations and an LSTM, but its
24-seed boss pools are calibration-sized rather than a plausible final training
distribution. Once a stage has nonzero success, expand its training seed pool and
sample by measured learning progress, retaining mastered seeds and fully disjoint
boss-stratified evaluation seeds. If reverse-curriculum transfer remains unstable,
the next evidence-backed fallback is a qualified live demonstration recorder and
a separately versioned behavior-cloning warm start—not another reward change.

The fixed `80/20` new-stage/mastered-stage mixture in EXP-0019 is consequently a
controlled first arm, not an assumed optimum. Prioritized Level Replay finds that
the learning value of a procedural level changes with the current policy and uses
that signal to revisit useful levels
([Jiang et al., 2021](https://proceedings.mlr.press/v139/jiang21b.html)); replay-based
continual RL similarly treats retained prior-task exposure as protection against
forgetting
([Caccia et al., 2023](https://proceedings.mlr.press/v232/caccia23a.html)). If the
fixed mixture fails, the next sampler should update probabilities from gameplay
completion learning progress while enforcing a nonzero mastered-stage floor. It
must publish one immutable distribution per PPO policy version, after all eight
fragments finish. Updating shared probabilities immediately when an asynchronous
actor ends would make subsequent reset choices depend on worker latency, violating
the controller's timing-independent seed/action contract. Start-state replay must
continue to generate fresh on-policy episodes; old transitions cannot be inserted
into PPO as though they came from the current policy. Shaped return must not be the
priority score because its scale differs across curriculum stages and can reward
activity without task completion.

## When to replace flat PPO with temporal abstraction

The normal-start Zone 2 objective has a natural event hierarchy that the live
bridge already observes authoritatively:

1. reveal and traverse the current floor;
2. discover and reach a descent;
3. repeat across Floors 1–3;
4. enter the boss arena;
5. defeat boss objectives and enter Zone 2.

A single primitive-action value function must currently assign credit across all
five phases despite their different horizons and state requirements. The
[options framework](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
formalizes closed-loop policies that act over multiple primitive steps, while
[Hierarchies of Reward Machines](https://proceedings.mlr.press/v202/furelos-blanco23a.html)
uses observed high-level events to decompose long-horizon sparse tasks into
independently learnable subtasks. This fits AutoDancer better than indefinitely
increasing movement shaping: floor transition, boss entry, boss damage, boss
death, and zone transition are already validated events, not inferred labels.

Temporal abstraction should not be bundled into EXP-0016. First establish that
the primitive A2 policy can acquire one assisted boss skill. If assistance can be
removed but Floor-3-to-boss training still fails, predeclare an architecture with:

- a small explicit task-state vector derived from authoritative events;
- separate learned navigation and combat option policies sharing the sensory
  encoder;
- an option selector operating at event/short fixed boundaries;
- primitive actions, rewards, and task-success evaluation unchanged;
- interruption when local threats appear, so a navigation option cannot ignore
  combat;
- normal-start evaluation with no scripted option actions.

This design also removes an informational asymmetry: A2 currently receives a
boss type through `PlayerFeature.TASK`, but receives no explicit representation
of whether the current high-level objective is exploration, a known stair route,
or a live boss objective. Humans infer and retain that task state naturally.

Hindsight Experience Replay is not a drop-in correction here. HER relabels goals
inside replay for an off-policy, goal-conditioned learner
([Andrychowicz et al., 2017](https://papers.nips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html));
AutoDancer currently uses on-policy recurrent PPO and has no goal-conditioned
critic. Adding HER would therefore change algorithm, replay semantics,
observation contract, and reward together, defeating causal diagnosis.

Demonstrations are a more compatible fallback if successful live traces can be
reproduced and recorded. Work such as
[Deep Q-learning from Demonstrations](https://ojs.aaai.org/index.php/AAAI/article/view/11757)
shows that small demonstration sets can materially reduce real-environment sample
cost, but its exact algorithm is value/replay based. For AutoDancer the controlled
analogue would be a separately versioned behavior-cloning auxiliary phase over
qualified observation/action/mask traces, followed by PPO fine-tuning and held-out
evaluation. A reported successful episode without its action sequence is not a
demonstration dataset and must not be treated as one.
