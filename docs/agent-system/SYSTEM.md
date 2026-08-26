# AutoDancer Agent — System Definition

_**This file is the living source of truth for AutoDancer's agent design.** The interactive atlas and this text twin are built from the same data._

_Question status: **0 open · 9 routed · 25 resolved**._

## One paragraph

AutoDancer is a recurrent PPO agent that learns Bard by acting in real Crypt of the NecroDancer processes. Python owns a fixed fleet of isolated game workers, exchanges actions and complete transitions with a Lua/native named-pipe bridge, constructs player-visible observations and explicit floor memory, and batches experience for a shared actor-critic. The outer experiment loop—not shaped return—decides whether a reward or architecture version is better using gameplay on unseen seeds. A2 with Reward V2 remains the measured baseline. EXP-0008 showed that extending both A2 and materially active A8 to 250,880 transitions did not produce held-out progress, so adaptation horizon is rejected. EXP-0009 now isolates whether argmax evaluation is suppressing useful behavior learned by the high-entropy sampled PPO policy.

## Decisions locked

| Axis | Decision | ADR |
|---|---|---|
| Environment | Train only against the authoritative live game; no simulator, screenshots, keyboard automation, or UI control. | [Protocol](../protocol.md) |
| Task | Bard, normal seeded All Zones, with normal episode outcomes and no Daily Challenge. | [Protocol](../protocol.md) |
| Transport | Use direct duplex named pipes for per-turn traffic; keep logs only for readiness, fatal errors, and diagnostics. | [Runtime](../runtime-efficiency.md) |
| Capacity | An explicit N means exactly N supervisor-owned workers; replace failures and never silently shrink the fleet. | [Runtime](../runtime-efficiency.md) |
| Information | Expose only player-visible state plus memory a human player could reasonably retain. | [Parity](../observation-parity.md) |
| Selection | Choose policies by held-out gameplay outcomes, never by shaped return. | [Reward history](../reward-history.md) |
| Baseline | Retain Reward V2 with Architecture A2 after V3, V4, A6, and A7 failed their declared gates. | [Reward history](../reward-history.md) |
| A7 outcome | Reject the zero-scalar-gated adapter: initialization parity passed, but zero of twelve new-input checkpoint tests reached material influence and gameplay gates failed; retain A2. | [Representation test](../representation-diagnostics.md) |
| Architecture admission | Before broad live training, require candidate input groups to show both controlled output sensitivity and encoder gradient reach; nonzero parameters alone are insufficient. | [Representation test](../representation-diagnostics.md) |
| A8 experiment | Compare unchanged A2 under legacy and repaired action contracts against an exact-parity A8 residual; freeze A2 for 10 updates and stop before broad gameplay unless representation and harm gates pass. | [A8 controls](../architecture8-controls.md) |
| A8 curve outcome | A8 passed representation and candidate gameplay criteria, but both A2 controls restarted once; stop before broad gameplay and retain A2 pending clean control retries. | [A8 result](../architecture8-controls.md#result) |
| Qualified A8 outcome | EXP-0007 passed parity, representation, controller, and early-curve gates, but A8 tied both controls on broad floor progress and missed the death gate; retain A2 at the 30,720-transition screening budget. | [Experiment contract](../../../experiments/EXP-0007/experiment.yaml) |
| A8 horizon outcome | Reject adaptation horizon as the explanation: at 250,880 transitions neither continued A2 nor A8 improved held-out progress, while frozen A2 remained the only arm to reach Floor 2. | [Decision](../../../experiments/EXP-0008/decision.json) |
| Policy execution diagnostic | EXP-0009 freezes all checkpoints and compares argmax with two reproducible stochastic sample streams; promotion requires repeatable Zone 2 progress across multiple unseen game seeds. | [Experiment contract](../../../experiments/EXP-0009/experiment.yaml) |

## Cost model

One environment turn is one acknowledged pipe command, one live engine turn, one schema validation, and one policy inference.
One default rollout is 128 transitions per worker. At eight workers, each PPO update follows 1,024 live transitions.
PPO replays 32-step recurrent chunks for four epochs. Dynamic inference waits at most 2 ms to batch whichever actors are ready.
Architecture A2 has 5,953,167 trainable parameters; A6 has 6,478,798. The added capacity is mostly perception, not a larger 512-unit LSTM.
## Deep dives

Experiment rationale and measured outcomes live in [reward-history.md](../reward-history.md). The Zone 2 environment audit lives in [rl-environment-audit.md](../rl-environment-audit.md). Action/mechanic evidence and representation gates live in [mechanic-diagnostics.md](../mechanic-diagnostics.md) and [representation-diagnostics.md](../representation-diagnostics.md). Protocol and performance details live in [protocol.md](../protocol.md) and [runtime-efficiency.md](../runtime-efficiency.md).

## Reading order (the atlas chapters)

1. **The authoritative turn** — The irreducible loop is a logical action, a real engine turn, and one exact acknowledgement. _(adds G, B, A)_
2. **Many independent games** — One Python supervisor turns the live loop into a fixed-capacity fleet. _(adds W)_
3. **What Bard can know** — Current perception and remembered space are separate, player-equivalent information sources. _(adds O, M)_
4. **Choosing with two memories** — Spatial memory tells the policy where it has been; temporal memory carries learned context. _(adds P, T)_
5. **Scoring without redefining success** — Reward shaping helps optimization; held-out gameplay remains the objective. _(adds R)_
6. **Gathering live experience** — Workers advance independently, but every accepted rollout belongs to one frozen policy version. _(adds C)_
7. **Changing the weights** — Recurrent PPO replays short sequences, then publishes one new policy version. _(adds L, K)_
8. **Keeping experiments honest** — Representation gates admit candidates; held-out gameplay decides promotion. _(adds E, D)_
9. **From A8 horizon rejection to Zone 2** — More information and longer training did not solve progression; execution semantics are the next isolated variable. _(adds F, H, E)_
10. **The whole agent system** — Explore every building block, version lineage, flow, decision, and open question.

## Structures

### The authoritative live loop

#### G · Live game task

**In one line.** The real game is the environment and the source of truth.

**What it does.** Each worker runs Bard in a normal seeded All Zones run. A transition is valid only after the engine accepts a command and reports the resulting authoritative state and events.

**How it's built.** **Version lineage.**
`Early` attached to one user-run process and depended on manual setup.
`Schema 4` introduced supervisor-created worker identities and seeded resets through a coordinator-era protocol.
`Current` launches workers directly, keeps exactly N healthy slots, and never attaches to unrelated processes. Gameplay semantics remain the real engine's.

**Steps in execution.**

1. **Reset** — Start a normal Bard All Zones run with the requested seed and a fresh run ID.
2. **Turn** — Accept exactly one logical action in a safe engine state.
3. **Resolve** — Let the engine update movement, combat, items, traps, floors, bosses, and terminal state.
4. **Report** — Return the complete acknowledged transition.

**Questions.**

- ~~**Q-G1** Which mode and character are permitted?~~ ✓ Only Bard in normal seeded All Zones; Daily Challenge is excluded (2026-08-24).
- **Q-G2** What is the practical maximum worker count? → _Run autodancer-benchmark across 1, 2, 4, 6, 8, 10, and 12 workers on the target machine._

#### B · Lua/native bridge

**In one line.** The bridge turns logical commands into acknowledged engine transitions.

**What it does.** Lua observes the live engine and accepts ASCII ACTION, RESET, and CLOSE commands. A small native module carries commands and JSON records over a worker-specific duplex Windows named pipe.

**How it's built.** **Version lineage.**
`Schema 4` used per-worker command files and JSONL telemetry discovered from logs.
`Schema 5` moved complete records to direct named pipes, retained ASCII commands, and removed per-turn log dependence.
`Schemas 6–9` kept the transport while extending observation semantics. Identity, session, run, seed, sequence, and acknowledgement checks remain mandatory. See `mods/AutoDancer/scripts/` and `src/autodancer/live/`.

**Steps in execution.**

1. **Receive** — Read a framed command from this worker’s pipe.
2. **Pend** — Keep it pending until the engine is in a safe state.
3. **Accept** — Bind command ID, requested action or seed, and run identity.
4. **Send** — Write one complete JSON transition, up to 64 KiB, back over the same pipe.

**Questions.**

- ~~**Q-B1** Should telemetry use a compact binary observation frame?~~ ✓ Not yet: direct JSON pipes exceeded the 2× throughput target, so binary framing remains contingent on future profiling (2026-08-24).
- ~~**Q-B2** Can records from another worker be accepted?~~ ✓ No; cross-worker, stale-session, duplicate, run-ID, seed, and sequence mismatches are rejected (2026-08-24).

#### A · Action contract

**In one line.** The policy chooses one of eleven engine-level Bard actions.

**What it does.** The discrete action space is up, right, down, left, wait, bomb, action-item 1, action-item 2, throw, spell 1, and spell 2. The observation carries a mask so unavailable inventory actions cannot be sampled.

**How it's built.** **Version lineage.**
`Early` control sent directions to an attached game.
`Schema 5+` standardized 11 logical actions and command acknowledgements.
`Current` always enables directions and WAIT, derives all six special-action bits from inventory/cooldown state, and requires SYNCHRONY's observed engine action to equal the injected action. Before/after state and events classify movement, wait, combat, interaction, digging, wall attempts, unchanged directions, special no-effect, and floor transitions.

**Steps in execution.**

1. **Mask** — Always enable directions and wait; match special actions exactly to inventory state.
2. **Sample** — Mask logits, then sample during training or take argmax during evaluation.
3. **Acknowledge** — Require the observed engine action to equal the injected action.
4. **Classify** — Describe the resolved mechanic from authoritative before/after state and events.

**Questions.**

- **Q-A1** Should known blocked directions be masked? → _Use outcome-classified live evidence to design a context-aware mask that preserves attacks, digging, and interactions._
- ~~**Q-A2** Is WAIT available now?~~ ✓ Yes; logical action 4 maps to engine IDLE, is always legal while Bard is running, and was live-acknowledged exactly (2026-08-24).

#### W · Worker fleet

**In one line.** Python owns exactly N isolated live-game workers.

**What it does.** The supervisor launches hidden, minimum-resolution workers with isolated profiles, unique identities and pipes, muted audio output, and only required game content. It refuses unrelated NecroDancer processes and cleans up only its own.

**How it's built.** **Version lineage.**
`Single-instance` required a running game.
`Coordinator era` used one hidden coordinator plus N native duplicate workers.
`Current` removes the unused coordinator from training: Python directly launches N workers, spreads affinity when beneficial, monitors process metrics, and replaces failed slots at fixed capacity. Implemented in `src/autodancer/live/supervisor.py`.

**Steps in execution.**

1. **Preflight** — Validate the game, native DLL, mod, requested N, and absence of unrelated processes.
2. **Isolate** — Create worker-specific LocalAppData and Roaming profiles.
3. **Launch** — Start each hidden worker with unique instance, config, log, pipe, and seed identities.
4. **Recover** — Replace a crashed or timed-out slot, reset its episode and recurrent state, and preserve N.

**Questions.**

- ~~**Q-W1** Does explicit --num-instances N permit fewer workers?~~ ✓ No; startup or recovery fails clearly rather than silently reducing capacity (2026-08-24).
- ~~**Q-W2** Should Windows affinity always override the scheduler?~~ ✓ Only when the auto benchmark finds a repeatable gain; otherwise Windows scheduling is retained (2026-08-24).

### What the agent can know

#### O · Live observation

**In one line.** A player-visible tensor contract describes the current turn.

**What it does.** The current schema exposes a 21×21 local grid, player and song state, thirteen inventory slots, a legal-action mask, map bounds, events, and identity metadata. Exact entity and item types use stable hashes; unseen facts stay unknown.

**How it's built.** **Version lineage.**
`Schema 4` established multi-worker identity.
`5` redesigned compact local grid, player, inventory, and 11-action semantics.
`6` added persistent map input.
`7` added enemy timing/status, facing, charge and shield cues, song timing, and full equipment cooldowns.
`8` added objects, interaction flags, prices, trap state, tells, and explosives.
`9` added shop-music volume and level bounds/capacity validation. See `docs/protocol.md`.

**Steps in execution.**

1. **Hook** — Read engine components after the accepted turn resolves.
2. **Assemble** — Build the 21×21 grid once and attach player, inventory, action, event, and map metadata.
3. **Validate** — Check schema, build, shapes, ranges, identities, and sequence.
4. **Convert** — Expose stable NumPy tensors to memory, reward, policy, and dashboard consumers.

**Questions.**

- **Q-O1** Can multiple objects share one cell without information loss? → _Define an ordered or set-valued representation for overlapping visible entities and objects before the next schema bump._
- **Q-O2** Are transient audio cues fully represented? → _Audit player-audible cues beyond the current shop-music signal and route justified additions through A7._

#### M · Spatial map memory

**In one line.** A 65×65 memory records revealed terrain and Bard’s own traversal.

**What it does.** The memory keeps floor-local revealed terrain, visit counts, and visit recency in absolute coordinates, then renders a player-centred 65×65 viewport. It never invents unseen terrain or retains stale off-screen entities.

**How it's built.** **Version lineage.**
`A2 / schema 5` had no explicit spatial memory; only the LSTM could remember past views.
`A3 / schema 6` added a five-channel map anchored at the spawn.
`A6 schema-9 fix` retained absolute history but changed the policy view to player-centred, eliminating clipping as Bard moved away from spawn. Levels larger than 65×65 fail clearly. Implemented in `src/autodancer/memory.py`.

**Steps in execution.**

1. **Reset floor** — Clear history when zone or floor identity changes.
2. **Merge view** — Add only terrain currently marked visible or revealed by the game.
3. **Record travel** — Increment Bard’s known visit count and recency.
4. **Render** — Centre the fixed policy viewport on Bard and encode five map channels.

**Questions.**

- ~~**Q-M1** Why not store off-screen enemies?~~ ✓ Their current positions are not player-knowable after they leave view; retaining them would create an informational advantage (2026-08-24).
- ~~**Q-M2** What happens on maps larger than 65×65?~~ ✓ Capacity validation fails instead of silently clipping; supporting larger custom levels requires a new representation (2026-08-24).

### Choosing and scoring actions

#### P · Policy and value network

**In one line.** One hybrid neural network chooses actions and estimates future return.

**What it does.** Parallel encoders process local geometry, salient entities, inventory, player/audio state, previous action/reward, and—after A2—the explicit map. They fuse to 512 features, pass through a 512-unit LSTM, and split into masked actor and critic heads.

**How it's built.** **Architecture lineage.**
`A2 · schema 5 · 5,953,167`: local residual CNN, entity transformer, player MLP, eight-slot inventory transformer, context encoder, 512 LSTM.
`A3 · schema 6 · 6,398,494`: adds 65×65 map CNN and expands fusion 896→1,024.
`A4 · schema 7 · 6,401,258`: tactical, song, and thirteen-slot equipment state.
`A5 · schema 8 · 6,478,670`: hazards, objects, interactions, prices, tells, explosives.
`A6 · schema 9 · 6,478,798`: shop audio and bounds; same 512 LSTM and heads.
`A7 · rejected · 6,401,648`: exact A2 plus a scalar-gated residual that stayed nearly closed.
`A8 · candidate`: exact A2 plus the same sensory branch and a zero-output 512×512 projection that receives a full first-step gradient.

**Steps in execution.**

1. **Embed** — Turn categorical classes, exact-type hashes, positions, and numeric state into compact features.
2. **Perceive** — Run local/map CNNs and entity/inventory attention encoders in parallel.
3. **Fuse** — Project all streams into a shared 512-value latent.
4. **Remember** — Update the LSTM hidden and cell state.
5. **Decide** — Produce 11 masked action logits and one state-value estimate.

**Questions.**

- ~~**Q-P1** Is A6 too small to learn the game?~~ ✓ The A6 pilot does not support that conclusion: it changed the latent interface and had only 51,200 new transitions versus A2’s 250,880. Capacity was not isolated (2026-08-24).
- **Q-P2** Would simply scaling parameter count solve progression? → _Establish a stable, behavior-preserving architecture and learning curve before running controlled width/depth scaling experiments._

#### T · Temporal memory

**In one line.** The LSTM carries learned context that explicit map channels do not.

**What it does.** A separate hidden and cell state summarizes action history, recent encounters, timing, and other partially observable context. It complements, rather than replaces, the explicit floor map.

**How it's built.** **Version lineage.**
`A2–A6` all use the same 512-unit LSTMCell and store exact [hidden, cell] state at every transition.
`Episode boundary` resets hidden state, previous action, and previous reward.
`Worker recovery` discards the incomplete fragment and resets recurrent state for that slot. No architecture version has yet increased temporal-memory width.

**Steps in execution.**

1. **Initialize** — Create zero hidden and cell tensors at reset.
2. **Condition** — Combine current fused perception with previous action and reward context.
3. **Update** — Produce next hidden and cell state on every live transition.
4. **Replay** — Seed each 32-step PPO chunk with the exact stored initial state.

**Questions.**

- **Q-T1** How much does the trained policy rely on its LSTM? → _Run hidden-state ablations and delayed-information probes on fixed seeds without changing rewards._
- ~~**Q-T2** Should the map be left to the LSTM instead?~~ ✓ No; explicit player-like spatial memory removes an avoidable information burden, while the LSTM handles temporal context (2026-08-24).

#### R · Reward shaping

**In one line.** Bounded learning signals point toward the sparse goal without defining success.

**What it does.** A stateful tracker deduplicates and caps exploration, combat, item, and navigation credit. Task reward, shaping reward, and gameplay outcomes are recorded separately; held-out gameplay selects policies.

**How it's built.** **Reward lineage.**
`Legacy`: direct events proved the loop but learned passivity.
`V1`: stateful exploration/combat/items improved local competence but no stairs.
`V2 · retained`: stair discovery/distance plus turn/revisit pressure produced the best local play and 24 training floor transitions.
`V3 · rejected`: bounded positive progression reduced timeouts but caused unsafe activity.
`V4A/B · rejected`: restored competence signals with 0.5/1.0 stair potential but became overwhelmingly passive.
`V5`: deliberately undefined until the action/architecture issue is isolated.

**Steps in execution.**

1. **Observe transition** — Compare authoritative events and episode-local state before and after the action.
2. **Deduplicate** — Credit each enemy, item type, position, and revealed tile only under its defined rules.
3. **Cap** — Keep renewable auxiliary positive credit below the +5 floor milestone.
4. **Split** — Return total, task/extrinsic, shaping, and named component values.

**Questions.**

- ~~**Q-R1** Should remaining stationary receive a generic penalty?~~ ✓ Not yet: A6’s unchanged-position turns were directional inputs and may include attacks or digging. Classify outcomes first (2026-08-24).
- **Q-R2** What is Reward V5? → _Defer its design until blocked-action behavior and function-preserving A7 transfer are tested._

### Collecting and changing weights

#### C · Rollout actors

**In one line.** Independent actors gather one contiguous fragment per worker and policy version.

**What it does.** Each actor owns its worker, episode, seed stream, recurrent state, pipe operations, and recovery. A central scheduler batches whichever observations are ready without forcing a per-step global barrier.

**How it's built.** **Version lineage.**
`Vector barrier` dispatched one action to every worker and waited for the slowest after every step.
`Versioned async` gives each slot an independent 128-transition state machine, dynamically batches inference with a 2 ms maximum delay, and assembles stable [time, worker] tensors only after all fragments finish.
`Recovery` discards only an incomplete fragment and recollects it under the same frozen policy version.

**Steps in execution.**

1. **Freeze** — Hold model weights constant for this rollout version.
2. **Infer** — Batch whichever actor observations arrive within the 2 ms window.
3. **Advance** — Let every worker build its own contiguous 128-transition fragment.
4. **Recover** — Replace and reset only a failed slot, then recollect its fragment.
5. **Assemble** — Stack completed fragments in stable worker-slot order.

**Questions.**

- ~~**Q-C1** Can a fast worker start using next-version weights early?~~ ✓ No; fragments never mix policy versions. Fast workers wait after completing their contribution (2026-08-24).
- ~~**Q-C2** Does asynchronous collection remove every barrier?~~ ✓ It removes the per-step barrier, but PPO still waits for one complete same-version fragment from every fixed-capacity slot (2026-08-24).

#### L · Recurrent PPO learner

**In one line.** Clipped PPO changes the shared policy from complete live-game rollouts.

**What it does.** The learner computes generalized advantage estimates, normalizes them, and replays exact recurrent chunks through clipped actor and value losses with entropy and gradient controls.

**How it's built.** **Version lineage.**
`Initial recurrent PPO` used the same core defaults but synchronous collection.
`Architecture A2` established exact 32-step LSTM replay and architecture-checked checkpoints.
`Async runtime` preserved the algorithm while feeding versioned actor fragments.
`Current diagnostics` record one pre-clipping gradient snapshot per representation group on every update, alongside CUDA inference and optimization. Rollout cadence remains 128×N transitions per update. Defaults: γ .99, GAE .95, clip .2, lr 3e-4, entropy .01, value .5, grad norm .5, four epochs.

**Steps in execution.**

1. **Bootstrap** — Estimate the value after the final rollout state.
2. **Advantage** — Compute backward GAE with terminal masking.
3. **Chunk** — Split each worker trajectory into 32-step sequences with stored initial LSTM state.
4. **Optimize** — Shuffle chunks across four epochs and update clipped policy and critic objectives.
5. **Publish** — Increment the policy version only after the whole update finishes.

**Questions.**

- **Q-L1** Is one update per 1,024 steps optimal at eight workers? → _Benchmark rollout length, epochs, minibatch chunks, and GPU utilization while holding total environment steps and evaluation seeds fixed._
- ~~**Q-L2** May PPO train on a partial healthy fleet?~~ ✓ No; fixed capacity and same-version rollout integrity take precedence over silently continuing with fewer workers (2026-08-24).

#### K · Checkpoints and metrics

**In one line.** Atomic artifacts make training recoverable and experiments auditable.

**What it does.** A checkpoint stores model, optimizer, global step, PPO configuration, architecture specification, reward specification, random states, and running metrics. Run directories keep resolved configuration and JSONL metrics.

**How it's built.** **Version lineage.**
`Early` saved working model state for the pipeline.
`A2+` requires exact architecture and PPO configuration for resume.
`Reward V4+` embeds exact reward arm weights to prevent cross-arm confusion.
`Current` supports exact resume plus explicit partial A2 warm-start paths that reset critic and optimizer when semantics change. Writes are atomic.

**Steps in execution.**

1. **Snapshot** — Collect model, optimizer, counters, configs, RNG states, and metrics.
2. **Write atomically** — Write a temporary artifact and replace the destination only when complete.
3. **Resume exactly** — Validate architecture, PPO, and reward metadata before restoring all state.
4. **Warm-start explicitly** — Transfer only declared compatible policy weights and initialize changed paths and critic afresh.

**Questions.**

- ~~**Q-K1** Can an A2 checkpoint be silently resumed as A6?~~ ✓ No; exact resume rejects it. Only the named partial warm-start path may transfer compatible weights (2026-08-24).
- ~~**Q-K2** Which artifact is the measured baseline?~~ ✓ runs/reward-v2-250k/final.pt, evaluated as Reward V2 Architecture A2 (2026-08-24).

### Evidence and operations

#### E · Evaluation and selection

**In one line.** Unseen-seed gameplay—not shaped return—decides whether a version advances.

**What it does.** Training reward is diagnostic, not the objective. Paired policies play fresh seeds with fresh recurrent state and a fixed turn cap; reports compare progression, death, timeouts, combat, items, movement, stairs, and runtime health.

**How it's built.** **Version lineage.**
`Baseline` compared checkpoint argmax play with masked random on held-out seeds.
`Reward pilots` added paired multi-checkpoint gates and separated task, shaping, and gameplay metrics.
`A6/A7/A8 gates` added transfer, representation, and long-horizon controls.
`EXP-0009` tests argmax against turn-keyed stochastic samples because PPO trained a high-entropy sampled policy and stochastic collection progressed farther than deterministic evaluation.
`Representation gate` perturbs one input group at a time and measures output sensitivity plus encoder gradients. A supported path below 1% of established-input medians is trace, not material.

**Steps in execution.**

1. **Predeclare** — Write hypotheses, seeds, budgets, metrics, and pass/fail gates before training.
2. **Probe representation** — Require material counterfactual sensitivity and gradient reach before broad training.
3. **Calibrate execution** — When policy entropy is material, compare argmax with reproducible stochastic sampling before attributing failure to learning.
4. **Evaluate** — Play ordered unseen game seeds with fresh recurrent state.
5. **Aggregate** — Preserve per-seed outcomes, repeated policy samples, and arm-level summaries.
6. **Decide** — Promote, continue, or reject using gameplay-ranked rules.
7. **Record** — Store the immutable experiment decision and supporting artifacts.

**Questions.**

- ~~**Q-E1** Why was A6 rejected even though it has more information?~~ ✓ It failed the predeclared gameplay gate and produced worse local competence and more stationary outcomes within the pilot budget (2026-08-24).
- ~~**Q-E2** Does the A6 result prove explicit map memory is harmful?~~ ✓ No; the warm start randomly reinitialized the expanded fusion interface, so representation and transfer disruption were confounded (2026-08-24).

#### D · Symbolic dashboard

**In one line.** A local page makes live training behavior and bottlenecks visible.

**What it does.** The dashboard renders every worker’s symbolic 21×21 view from telemetry and keeps health, episode, reward, action, PPO, throughput, latency, recovery, and fragment-straggler metrics on one screen.

**How it's built.** **Version lineage.**
`Initial` exposed training metrics only.
`Symbolic workers` added local grid views without game-window fidelity or screenshot capture.
`Compact layout` constrained variable-length sections so reward components could not push the whole page below the viewport.
`Async fix` publishes each actor transition to the dashboard, restoring visible movement during direct-pipe collection. Implemented in `src/autodancer/training/dashboard.py`.

**Steps in execution.**

1. **Publish** — Collector sends the latest observation, action, reward, and worker info.
2. **Aggregate** — Training loop attaches PPO, reward-component, throughput, and recovery metrics.
3. **Render** — Local HTTP clients draw symbolic cells and compact metric panels.

**Questions.**

- ~~**Q-D1** Does rendering symbolic workers affect the game workers?~~ ✓ Only modestly: it reuses observation tensors already built for training and avoids image capture or full game rendering (2026-08-24).
- ~~**Q-D2** Is the dashboard part of policy input?~~ ✓ No; it is an observer only and cannot control or alter gameplay (2026-08-24).

### Experimental candidates

#### F · Architecture A7 adapter

**In one line.** Exact A2 initialization succeeded, but every new-input path remained below material influence.

**What it does.** A7 is an implemented experimental model. The complete A2 actor-critic remains intact while a separate branch reads only map, tactical, hazard, interaction, audio, and expanded inventory fields that A2 could not see.

**How it's built.** **Measured lineage.**
`Parity`: every A2 tensor—including critic—loaded under `base.*`, with exactly zero output and recurrent-state error at gate zero.
`Training`: all three 51,200-step pilots were finite, but final gates were only −0.001276, +0.000265, and −0.000030.
`Representation`: tactical grid, map, extended player, and extended inventory paths were numerically active, but zero of twelve checkpoint/group pairs reached both 1% sensitivity and gradient materiality; none changed an argmax action.
`Gameplay`: two pilots reached floor 2, but items, timeouts, stationary behavior, and runtime health failed the declared gate.

**Steps in execution.**

1. **Load** — Copy every A2 checkpoint tensor into the preserved base and reset only the optimizer.
2. **Prove parity** — Measure exact output and deterministic-action parity before live training.
3. **Adapt** — Let PPO learn the scalar gate and residual while continuing to update the A2 base.
4. **Measure influence** — Perturb each new input group and backpropagate to its encoder.
5. **Reject** — Retain A2 after representation and held-out gameplay gates fail.

**Questions.**

- ~~**Q-F1** Was A7 behavior-preserving at initialization?~~ ✓ Yes; unit tests and the real V2 checkpoint measured exact tensor and output parity, including reset-containing sequences (2026-08-24).
- ~~**Q-F2** Did the richer observations receive meaningful influence?~~ ✓ No: all twelve trained checkpoint/group checks were only trace-active, with zero material groups and zero argmax action changes (2026-08-24).
- **Q-F3** What replaces the scalar-gate design? → _Use a representation-learning preflight to test an exact-zero output projection or staged frozen-base adapter before live gameplay training._

#### H · Architecture A8 controls

**In one line.** A8 uses the rich inputs, but longer training still did not improve held-out progression.

**What it does.** A8 preserves every A2 tensor and adds schema-9 perception through a zero-output 512×512 matrix projection. EXP-0007 proved exact initialization parity and material influence from all four new groups. EXP-0008 then continued A2 and A8 exactly to 250,880 transitions; neither improved final held-out progress, and A8 lost local combat and item competence relative to frozen A2.

**How it's built.** **Qualified EXP-0007.**
`A2 frozen`: broad progress 1.00, death 0.30, 44 kills, 37 items, unchanged position 0.913.
`A8`: progress 1.00, death 0.20, 43 kills, 38 items, unchanged position 0.587.
**EXP-0008 final.**
`A2 frozen`: progress 1.125 and Floor 2.
`A2 continuation`: progress 1.00, 73 kills, 55 items.
`A8 continuation`: progress 1.00, 38 kills, 28 items.
`Interpretation`: added information and 250,880 transitions did not solve strategic navigation.

**Steps in execution.**

1. **Prove parity** — Load the real V2 checkpoint and require zero logits, value, recurrent-state, and action error.
2. **Open projection** — Freeze A2 for ten updates while the full residual matrix receives the first gradients.
3. **Qualify broadly** — Use the schema-10 controller and fresh held-out seeds to separate activity from gameplay progress.
4. **Continue exactly** — Resume A2 and A8 model, critic, optimizer, counters, and RNG state to 250,880 transitions.
5. **Test emergence** — Require A8 to beat continued A2 at two consecutive checkpoints including the final checkpoint.

**Questions.**

- ~~**Q-H1** Does A8 replace A2?~~ ✓ No. EXP-0008 found no final held-out progress advantage at 250,880 transitions and worse local competence than frozen A2 (2026-08-26).
- ~~**Q-H2** Did A8 solve A7’s gradient starvation?~~ ✓ Yes: all four new observation groups were material at warmup and final, with exact A2 behavior at initialization (2026-08-26).
- ~~**Q-H3** Did A8 merely need more of the same training?~~ ✓ No evidence supports that explanation: the exact continuation to 250,880 transitions did not improve held-out floor progress (2026-08-26).

## Flows (representative packets)

Payload shapes are what the design implies, not measured traffic.

### One live Bard turn

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | P → A | masked action | `{"action":"RIGHT","policy_version":17}` |
| 2 | A → B | ACTION | `{"command_id":884,"action":1}` |
| 3 | B → G | accepted input | `{"worker":"worker-0003"}` |
| 4 | G → B | resolved turn | `{"events":["move"],"terminal":false}` |
| 5 | B → O | schema-9 record | `{"sequence":885,"acknowledged":884}` |
| 6 | O → M | visible terrain | `{"grid":"21×21×29"}` |
| 7 | M → P | floor memory | `{"map":"65×65×5"}` |
| 8 | O → P | current state | `{"player":21,"inventory":"13×8"}` |
| 9 | T → P | prior [h,c] | `{"width":512}` |
| 10 | P → T | next [h,c] | `{"reset":false}` |

### One policy update

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | W → G | N isolated games | `{"capacity":8}` |
| 2 | O → C | ready observations | `{"max_batch_delay_ms":2}` |
| 3 | R → C | reward components | `{"task":0,"shaping":0.005}` |
| 4 | P → C | frozen-version inference | `{"policy_version":17}` |
| 5 | C → L | rollout batch | `{"shape":"[128,8]","version":17}` |
| 6 | L → P | updated weights | `{"policy_version":18,"epochs":4}` |
| 7 | L → K | atomic snapshot | `{"global_step":18432}` |

### Choose a version

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | K → E | candidate checkpoints | `{"candidates":["A2","A6-1","A6-2","A6-3"]}` |
| 2 | E → W | ordered unseen seeds | `{"seeds":"57001–57024","execution":"argmax + seeded samples"}` |
| 3 | W → E | gameplay outcomes | `{"progress":1.011,"deaths":0.322}` |
| 4 | E → D | report and diagnostics | `{"decision":"retain A2"}` |

### Recover one worker

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | B → C | timeout/disconnect | `{"worker":"worker-0005"}` |
| 2 | C → W | replace slot | `{"discard_incomplete_fragment":true}` |
| 3 | W → G | fresh owned process | `{"same_slot":5}` |
| 4 | W → B | fresh pipe/session | `{"healthy_capacity":8}` |
| 5 | C → T | clear recurrent state | `{"h":0,"c":0}` |

### Rejected A7 compatibility experiment

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | P → F | exact A2 path | `{"transferred":"all compatible policy behavior"}` |
| 2 | O → F | A6-only inputs | `{"branch":"residual"}` |
| 3 | F → A | gated logits | `{"initial_gate":0}` |
| 4 | F → E | paired candidate | `{"reward":"V2 unchanged"}` |

### Controlled A8 learning experiment

| # | From → To | Packet | Representative payload |
|---|---|---|---|
| 1 | K → H | exact V2 A2 | `{"parity_error":0}` |
| 2 | O → H | schema-9-only inputs | `{"projection_initially_zero":true}` |
| 3 | H → L | frozen-base warmup | `{"updates":10}` |
| 4 | L → E | four-point curves | `{"controls":["legacy-no-wait","current"]}` |
| 5 | E → H | admit or stop | `{"broad_only_after_pass":true}` |

## Questions — index

Reference by ID. ✓ resolved (with date) · → routed to a named next step · otherwise open.

- ~~**Q-G1**~~ (G) ✓ Only Bard in normal seeded All Zones; Daily Challenge is excluded (2026-08-24).
- **Q-G2** (G) What is the practical maximum worker count? → _Run autodancer-benchmark across 1, 2, 4, 6, 8, 10, and 12 workers on the target machine._
- ~~**Q-B1**~~ (B) ✓ Not yet: direct JSON pipes exceeded the 2× throughput target, so binary framing remains contingent on future profiling (2026-08-24).
- ~~**Q-B2**~~ (B) ✓ No; cross-worker, stale-session, duplicate, run-ID, seed, and sequence mismatches are rejected (2026-08-24).
- **Q-A1** (A) Should known blocked directions be masked? → _Use outcome-classified live evidence to design a context-aware mask that preserves attacks, digging, and interactions._
- ~~**Q-A2**~~ (A) ✓ Yes; logical action 4 maps to engine IDLE, is always legal while Bard is running, and was live-acknowledged exactly (2026-08-24).
- ~~**Q-W1**~~ (W) ✓ No; startup or recovery fails clearly rather than silently reducing capacity (2026-08-24).
- ~~**Q-W2**~~ (W) ✓ Only when the auto benchmark finds a repeatable gain; otherwise Windows scheduling is retained (2026-08-24).
- **Q-O1** (O) Can multiple objects share one cell without information loss? → _Define an ordered or set-valued representation for overlapping visible entities and objects before the next schema bump._
- **Q-O2** (O) Are transient audio cues fully represented? → _Audit player-audible cues beyond the current shop-music signal and route justified additions through A7._
- ~~**Q-M1**~~ (M) ✓ Their current positions are not player-knowable after they leave view; retaining them would create an informational advantage (2026-08-24).
- ~~**Q-M2**~~ (M) ✓ Capacity validation fails instead of silently clipping; supporting larger custom levels requires a new representation (2026-08-24).
- ~~**Q-P1**~~ (P) ✓ The A6 pilot does not support that conclusion: it changed the latent interface and had only 51,200 new transitions versus A2’s 250,880. Capacity was not isolated (2026-08-24).
- **Q-P2** (P) Would simply scaling parameter count solve progression? → _Establish a stable, behavior-preserving architecture and learning curve before running controlled width/depth scaling experiments._
- **Q-T1** (T) How much does the trained policy rely on its LSTM? → _Run hidden-state ablations and delayed-information probes on fixed seeds without changing rewards._
- ~~**Q-T2**~~ (T) ✓ No; explicit player-like spatial memory removes an avoidable information burden, while the LSTM handles temporal context (2026-08-24).
- ~~**Q-R1**~~ (R) ✓ Not yet: A6’s unchanged-position turns were directional inputs and may include attacks or digging. Classify outcomes first (2026-08-24).
- **Q-R2** (R) What is Reward V5? → _Defer its design until blocked-action behavior and function-preserving A7 transfer are tested._
- ~~**Q-C1**~~ (C) ✓ No; fragments never mix policy versions. Fast workers wait after completing their contribution (2026-08-24).
- ~~**Q-C2**~~ (C) ✓ It removes the per-step barrier, but PPO still waits for one complete same-version fragment from every fixed-capacity slot (2026-08-24).
- **Q-L1** (L) Is one update per 1,024 steps optimal at eight workers? → _Benchmark rollout length, epochs, minibatch chunks, and GPU utilization while holding total environment steps and evaluation seeds fixed._
- ~~**Q-L2**~~ (L) ✓ No; fixed capacity and same-version rollout integrity take precedence over silently continuing with fewer workers (2026-08-24).
- ~~**Q-K1**~~ (K) ✓ No; exact resume rejects it. Only the named partial warm-start path may transfer compatible weights (2026-08-24).
- ~~**Q-K2**~~ (K) ✓ runs/reward-v2-250k/final.pt, evaluated as Reward V2 Architecture A2 (2026-08-24).
- ~~**Q-E1**~~ (E) ✓ It failed the predeclared gameplay gate and produced worse local competence and more stationary outcomes within the pilot budget (2026-08-24).
- ~~**Q-E2**~~ (E) ✓ No; the warm start randomly reinitialized the expanded fusion interface, so representation and transfer disruption were confounded (2026-08-24).
- ~~**Q-D1**~~ (D) ✓ Only modestly: it reuses observation tensors already built for training and avoids image capture or full game rendering (2026-08-24).
- ~~**Q-D2**~~ (D) ✓ No; it is an observer only and cannot control or alter gameplay (2026-08-24).
- ~~**Q-F1**~~ (F) ✓ Yes; unit tests and the real V2 checkpoint measured exact tensor and output parity, including reset-containing sequences (2026-08-24).
- ~~**Q-F2**~~ (F) ✓ No: all twelve trained checkpoint/group checks were only trace-active, with zero material groups and zero argmax action changes (2026-08-24).
- **Q-F3** (F) What replaces the scalar-gate design? → _Use a representation-learning preflight to test an exact-zero output projection or staged frozen-base adapter before live gameplay training._
- ~~**Q-H1**~~ (H) ✓ No. EXP-0008 found no final held-out progress advantage at 250,880 transitions and worse local competence than frozen A2 (2026-08-26).
- ~~**Q-H2**~~ (H) ✓ Yes: all four new observation groups were material at warmup and final, with exact A2 behavior at initialization (2026-08-26).
- ~~**Q-H3**~~ (H) ✓ No evidence supports that explanation: the exact continuation to 250,880 transitions did not improve held-out floor progress (2026-08-26).

## What the platform gives vs what we own

**Platform gives:** NecroDancer supplies authoritative mechanics, level generation, entities, events, Bard timing, and SYNCHRONY duplicate-instance support. The mod runtime supplies Lua hooks; the native library supplies duplex named pipes.

**We own:** Worker lifecycle, identity and sequence validation, observations, player-visible map memory, legal-action masking, reward state, policy/value networks, recurrent rollout collection, PPO updates, checkpoints, evaluation gates, and the symbolic dashboard.

## Planned filesystem

```
mods/AutoDancer/scripts/
  AutoDancer.lua        engine hooks and telemetry
  Bridge.lua            command acceptance
src/autodancer/
  live/                 protocol, pipe, supervisor
  envs/                 live and vector interfaces
  observation.py        schema-to-tensor contract
  memory.py             persistent floor map
  rewards.py            versioned reward tracker
  training/
    model.py            recurrent actor-critic
    action_contract.py  versioned policy-side action masks
    representation.py   sensitivity and gradient gates
    architecture8_compare.py  initial A8 decision gates
    architecture8_horizon_compare.py  long-horizon A8 gates
    async_collector.py  versioned actor scheduler
    ppo.py              recurrent PPO and checkpoints
    baseline.py         deterministic/stochastic evaluation
    stochastic_policy_compare.py  Zone 2 execution-mode gate
    dashboard.py        local symbolic telemetry
docs/agent-system/
  atlas/data.mjs        authored system definition
  atlas.html            generated interactive atlas
  SYSTEM.md             generated text twin
```

## How this file is maintained

Generated from `docs/agent-system/atlas/data.mjs` by `node docs/agent-system/atlas/build.mjs`, which also builds the interactive atlas (`atlas.html`, published at http://localhost:8780/atlas.html). Edit the data file, rebuild, republish — never edit this file by hand.
