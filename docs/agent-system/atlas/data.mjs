export const META = {
  title: 'AutoDancer Agent',
  artifactUrl: 'http://localhost:8780/atlas.html',
  sourcePath: 'docs/agent-system/atlas/data.mjs',
  buildCmd: 'node docs/agent-system/atlas/build.mjs',
  stats: [
    { k: 'Current default', v: 'schema 9 · Architecture A6' },
    { k: 'Measured baseline', v: 'Reward V2 · Architecture A2' },
  ],
  intro: `_**This file is the living source of truth for AutoDancer's agent design.** The interactive atlas and this text twin are built from the same data._`,
  onePara: `AutoDancer is a recurrent PPO agent that learns Bard by acting in real Crypt of the NecroDancer processes. Python owns a fixed fleet of isolated game workers, exchanges actions and complete transitions with a Lua/native named-pipe bridge, constructs player-visible observations and explicit floor memory, and batches experience for a shared actor-critic. The outer experiment loop—not shaped return—decides whether a reward or architecture version is better using gameplay on unseen seeds. A2 with Reward V2 remains the measured baseline. EXP-0008 showed that extending both A2 and materially active A8 to 250,880 transitions did not produce held-out progress, so adaptation horizon is rejected. EXP-0009 now isolates whether argmax evaluation is suppressing useful behavior learned by the high-entropy sampled PPO policy.`,
  costModel: [
    'One environment turn is one acknowledged pipe command, one live engine turn, one schema validation, and one policy inference.',
    'One default rollout is 128 transitions per worker. At eight workers, each PPO update follows 1,024 live transitions.',
    'PPO replays 32-step recurrent chunks for four epochs. Dynamic inference waits at most 2 ms to batch whichever actors are ready.',
    'Architecture A2 has 5,953,167 trainable parameters; A6 has 6,478,798. The added capacity is mostly perception, not a larger 512-unit LSTM.',
  ],
  deepDive: 'Experiment rationale and measured outcomes live in [reward-history.md](../reward-history.md). The Zone 2 environment audit lives in [rl-environment-audit.md](../rl-environment-audit.md). Action/mechanic evidence and representation gates live in [mechanic-diagnostics.md](../mechanic-diagnostics.md) and [representation-diagnostics.md](../representation-diagnostics.md). Protocol and performance details live in [protocol.md](../protocol.md) and [runtime-efficiency.md](../runtime-efficiency.md).',
  platformGives: 'NecroDancer supplies authoritative mechanics, level generation, entities, events, Bard timing, and SYNCHRONY duplicate-instance support. The mod runtime supplies Lua hooks; the native library supplies duplex named pipes.',
  weOwn: 'Worker lifecycle, identity and sequence validation, observations, player-visible map memory, legal-action masking, reward state, policy/value networks, recurrent rollout collection, PPO updates, checkpoints, evaluation gates, and the symbolic dashboard.',
  filesystem: `mods/AutoDancer/scripts/
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
  SYSTEM.md             generated text twin`,
};

export const DECISIONS = [
  { axis: 'Environment', decision: 'Train only against the authoritative live game; no simulator, screenshots, keyboard automation, or UI control.', adr: '[Protocol](../protocol.md)' },
  { axis: 'Task', decision: 'Bard, normal seeded All Zones, with normal episode outcomes and no Daily Challenge.', adr: '[Protocol](../protocol.md)' },
  { axis: 'Transport', decision: 'Use direct duplex named pipes for per-turn traffic; keep logs only for readiness, fatal errors, and diagnostics.', adr: '[Runtime](../runtime-efficiency.md)' },
  { axis: 'Capacity', decision: 'An explicit N means exactly N supervisor-owned workers; replace failures and never silently shrink the fleet.', adr: '[Runtime](../runtime-efficiency.md)' },
  { axis: 'Information', decision: 'Expose only player-visible state plus memory a human player could reasonably retain.', adr: '[Parity](../observation-parity.md)' },
  { axis: 'Selection', decision: 'Choose policies by held-out gameplay outcomes, never by shaped return.', adr: '[Reward history](../reward-history.md)' },
  { axis: 'Baseline', decision: 'Retain Reward V2 with Architecture A2 after V3, V4, A6, and A7 failed their declared gates.', adr: '[Reward history](../reward-history.md)' },
  { axis: 'A7 outcome', decision: 'Reject the zero-scalar-gated adapter: initialization parity passed, but zero of twelve new-input checkpoint tests reached material influence and gameplay gates failed; retain A2.', adr: '[Representation test](../representation-diagnostics.md)' },
  { axis: 'Architecture admission', decision: 'Before broad live training, require candidate input groups to show both controlled output sensitivity and encoder gradient reach; nonzero parameters alone are insufficient.', adr: '[Representation test](../representation-diagnostics.md)' },
  { axis: 'A8 experiment', decision: 'Compare unchanged A2 under legacy and repaired action contracts against an exact-parity A8 residual; freeze A2 for 10 updates and stop before broad gameplay unless representation and harm gates pass.', adr: '[A8 controls](../architecture8-controls.md)' },
  { axis: 'A8 curve outcome', decision: 'A8 passed representation and candidate gameplay criteria, but both A2 controls restarted once; stop before broad gameplay and retain A2 pending clean control retries.', adr: '[A8 result](../architecture8-controls.md#result)' },
  { axis: 'Qualified A8 outcome', decision: 'EXP-0007 passed parity, representation, controller, and early-curve gates, but A8 tied both controls on broad floor progress and missed the death gate; retain A2 at the 30,720-transition screening budget.', adr: '[Experiment contract](../../../experiments/EXP-0007/experiment.yaml)' },
  { axis: 'A8 horizon outcome', decision: 'Reject adaptation horizon as the explanation: at 250,880 transitions neither continued A2 nor A8 improved held-out progress, while frozen A2 remained the only arm to reach Floor 2.', adr: '[Decision](../../../experiments/EXP-0008/decision.json)' },
  { axis: 'Policy execution diagnostic', decision: 'EXP-0009 freezes all checkpoints and compares argmax with two reproducible stochastic sample streams; promotion requires repeatable Zone 2 progress across multiple unseen game seeds.', adr: '[Experiment contract](../../../experiments/EXP-0009/experiment.yaml)' },
];

export const GROUPS = [
  { id: 'live', title: 'The authoritative live loop' },
  { id: 'sense', title: 'What the agent can know' },
  { id: 'decide', title: 'Choosing and scoring actions' },
  { id: 'learn', title: 'Collecting and changing weights' },
  { id: 'evidence', title: 'Evidence and operations' },
  { id: 'future', title: 'Experimental candidates' },
];

export const NODES = [
  {
    id: 'G', code: 'G', name: 'Live game task', short: 'LIVE GAME', group: 'live', gx: 1, gy: 1, w: 3, d: 2, h: 42, kind: 'slab',
    one: 'The real game is the environment and the source of truth.',
    what: 'Each worker runs Bard in a normal seeded All Zones run. A transition is valid only after the engine accepts a command and reports the resulting authoritative state and events.',
    how: `<strong>Version lineage.</strong><br><code>Early</code> attached to one user-run process and depended on manual setup.<br><code>Schema 4</code> introduced supervisor-created worker identities and seeded resets through a coordinator-era protocol.<br><code>Current</code> launches workers directly, keeps exactly N healthy slots, and never attaches to unrelated processes. Gameplay semantics remain the real engine's.`,
    steps: [['Reset', 'Start a normal Bard All Zones run with the requested seed and a fresh run ID.'], ['Turn', 'Accept exactly one logical action in a safe engine state.'], ['Resolve', 'Let the engine update movement, combat, items, traps, floors, bosses, and terminal state.'], ['Report', 'Return the complete acknowledged transition.']],
    cond: [{ q: 'Which mode and character are permitted?', r: 'Only Bard in normal seeded All Zones; Daily Challenge is excluded (2026-08-24).' }, { q: 'What is the practical maximum worker count?', to: 'Run autodancer-benchmark across 1, 2, 4, 6, 8, 10, and 12 workers on the target machine.' }],
  },
  {
    id: 'B', code: 'B', name: 'Lua/native bridge', short: 'PIPE BRIDGE', group: 'live', gx: 5.5, gy: 1, w: 3, d: 2, h: 56, kind: 'gate',
    one: 'The bridge turns logical commands into acknowledged engine transitions.',
    what: 'Lua observes the live engine and accepts ASCII ACTION, RESET, and CLOSE commands. A small native module carries commands and JSON records over a worker-specific duplex Windows named pipe.',
    how: `<strong>Version lineage.</strong><br><code>Schema 4</code> used per-worker command files and JSONL telemetry discovered from logs.<br><code>Schema 5</code> moved complete records to direct named pipes, retained ASCII commands, and removed per-turn log dependence.<br><code>Schemas 6–9</code> kept the transport while extending observation semantics. Identity, session, run, seed, sequence, and acknowledgement checks remain mandatory. See <code>mods/AutoDancer/scripts/</code> and <code>src/autodancer/live/</code>.`,
    steps: [['Receive', 'Read a framed command from this worker’s pipe.'], ['Pend', 'Keep it pending until the engine is in a safe state.'], ['Accept', 'Bind command ID, requested action or seed, and run identity.'], ['Send', 'Write one complete JSON transition, up to 64 KiB, back over the same pipe.']],
    cond: [{ q: 'Should telemetry use a compact binary observation frame?', r: 'Not yet: direct JSON pipes exceeded the 2× throughput target, so binary framing remains contingent on future profiling (2026-08-24).' }, { q: 'Can records from another worker be accepted?', r: 'No; cross-worker, stale-session, duplicate, run-ID, seed, and sequence mismatches are rejected (2026-08-24).' }],
  },
  {
    id: 'A', code: 'A', name: 'Action contract', short: '11 ACTIONS', group: 'live', gx: 10, gy: 1, w: 3, d: 2, h: 44, kind: 'gate',
    one: 'The policy chooses one of eleven engine-level Bard actions.',
    what: 'The discrete action space is up, right, down, left, wait, bomb, action-item 1, action-item 2, throw, spell 1, and spell 2. The observation carries a mask so unavailable inventory actions cannot be sampled.',
    how: `<strong>Version lineage.</strong><br><code>Early</code> control sent directions to an attached game.<br><code>Schema 5+</code> standardized 11 logical actions and command acknowledgements.<br><code>Current</code> always enables directions and WAIT, derives all six special-action bits from inventory/cooldown state, and requires SYNCHRONY's observed engine action to equal the injected action. Before/after state and events classify movement, wait, combat, interaction, digging, wall attempts, unchanged directions, special no-effect, and floor transitions.`,
    steps: [['Mask', 'Always enable directions and wait; match special actions exactly to inventory state.'], ['Sample', 'Mask logits, then sample during training or take argmax during evaluation.'], ['Acknowledge', 'Require the observed engine action to equal the injected action.'], ['Classify', 'Describe the resolved mechanic from authoritative before/after state and events.']],
    cond: [{ q: 'Should known blocked directions be masked?', to: 'Use outcome-classified live evidence to design a context-aware mask that preserves attacks, digging, and interactions.' }, { q: 'Is WAIT available now?', r: 'Yes; logical action 4 maps to engine IDLE, is always legal while Bard is running, and was live-acknowledged exactly (2026-08-24).' }],
  },
  {
    id: 'W', code: 'W', name: 'Worker fleet', short: 'WORKER FLEET', group: 'live', gx: 1, gy: 5.5, w: 3, d: 3, h: 58, kind: 'cards',
    one: 'Python owns exactly N isolated live-game workers.',
    what: 'The supervisor launches hidden, minimum-resolution workers with isolated profiles, unique identities and pipes, muted audio output, and only required game content. It refuses unrelated NecroDancer processes and cleans up only its own.',
    how: `<strong>Version lineage.</strong><br><code>Single-instance</code> required a running game.<br><code>Coordinator era</code> used one hidden coordinator plus N native duplicate workers.<br><code>Current</code> removes the unused coordinator from training: Python directly launches N workers, spreads affinity when beneficial, monitors process metrics, and replaces failed slots at fixed capacity. Implemented in <code>src/autodancer/live/supervisor.py</code>.`,
    steps: [['Preflight', 'Validate the game, native DLL, mod, requested N, and absence of unrelated processes.'], ['Isolate', 'Create worker-specific LocalAppData and Roaming profiles.'], ['Launch', 'Start each hidden worker with unique instance, config, log, pipe, and seed identities.'], ['Recover', 'Replace a crashed or timed-out slot, reset its episode and recurrent state, and preserve N.']],
    cond: [{ q: 'Does explicit --num-instances N permit fewer workers?', r: 'No; startup or recovery fails clearly rather than silently reducing capacity (2026-08-24).' }, { q: 'Should Windows affinity always override the scheduler?', r: 'Only when the auto benchmark finds a repeatable gain; otherwise Windows scheduling is retained (2026-08-24).' }],
  },
  {
    id: 'O', code: 'O', name: 'Live observation', short: 'OBSERVATION', group: 'sense', gx: 5.5, gy: 5.5, w: 3, d: 3, h: 66, kind: 'screen',
    one: 'A player-visible tensor contract describes the current turn.',
    what: 'The current schema exposes a 21×21 local grid, player and song state, thirteen inventory slots, a legal-action mask, map bounds, events, and identity metadata. Exact entity and item types use stable hashes; unseen facts stay unknown.',
    how: `<strong>Version lineage.</strong><br><code>Schema 4</code> established multi-worker identity.<br><code>5</code> redesigned compact local grid, player, inventory, and 11-action semantics.<br><code>6</code> added persistent map input.<br><code>7</code> added enemy timing/status, facing, charge and shield cues, song timing, and full equipment cooldowns.<br><code>8</code> added objects, interaction flags, prices, trap state, tells, and explosives.<br><code>9</code> added shop-music volume and level bounds/capacity validation. See <code>docs/protocol.md</code>.`,
    steps: [['Hook', 'Read engine components after the accepted turn resolves.'], ['Assemble', 'Build the 21×21 grid once and attach player, inventory, action, event, and map metadata.'], ['Validate', 'Check schema, build, shapes, ranges, identities, and sequence.'], ['Convert', 'Expose stable NumPy tensors to memory, reward, policy, and dashboard consumers.']],
    cond: [{ q: 'Can multiple objects share one cell without information loss?', to: 'Define an ordered or set-valued representation for overlapping visible entities and objects before the next schema bump.' }, { q: 'Are transient audio cues fully represented?', to: 'Audit player-audible cues beyond the current shop-music signal and route justified additions through A7.' }],
  },
  {
    id: 'M', code: 'M', name: 'Spatial map memory', short: 'FLOOR MEMORY', group: 'sense', gx: 10, gy: 5.5, w: 3, d: 3, h: 34, kind: 'store',
    one: 'A 65×65 memory records revealed terrain and Bard’s own traversal.',
    what: 'The memory keeps floor-local revealed terrain, visit counts, and visit recency in absolute coordinates, then renders a player-centred 65×65 viewport. It never invents unseen terrain or retains stale off-screen entities.',
    how: `<strong>Version lineage.</strong><br><code>A2 / schema 5</code> had no explicit spatial memory; only the LSTM could remember past views.<br><code>A3 / schema 6</code> added a five-channel map anchored at the spawn.<br><code>A6 schema-9 fix</code> retained absolute history but changed the policy view to player-centred, eliminating clipping as Bard moved away from spawn. Levels larger than 65×65 fail clearly. Implemented in <code>src/autodancer/memory.py</code>.`,
    steps: [['Reset floor', 'Clear history when zone or floor identity changes.'], ['Merge view', 'Add only terrain currently marked visible or revealed by the game.'], ['Record travel', 'Increment Bard’s known visit count and recency.'], ['Render', 'Centre the fixed policy viewport on Bard and encode five map channels.']],
    cond: [{ q: 'Why not store off-screen enemies?', r: 'Their current positions are not player-knowable after they leave view; retaining them would create an informational advantage (2026-08-24).' }, { q: 'What happens on maps larger than 65×65?', r: 'Capacity validation fails instead of silently clipping; supporting larger custom levels requires a new representation (2026-08-24).' }],
  },
  {
    id: 'P', code: 'P', name: 'Policy and value network', short: 'ACTOR CRITIC', group: 'decide', gx: 5.5, gy: 10.5, w: 4, d: 3, h: 82, kind: 'tall',
    one: 'One hybrid neural network chooses actions and estimates future return.',
    what: 'Parallel encoders process local geometry, salient entities, inventory, player/audio state, previous action/reward, and—after A2—the explicit map. They fuse to 512 features, pass through a 512-unit LSTM, and split into masked actor and critic heads.',
    how: `<strong>Architecture lineage.</strong><br><code>A2 · schema 5 · 5,953,167</code>: local residual CNN, entity transformer, player MLP, eight-slot inventory transformer, context encoder, 512 LSTM.<br><code>A3 · schema 6 · 6,398,494</code>: adds 65×65 map CNN and expands fusion 896→1,024.<br><code>A4 · schema 7 · 6,401,258</code>: tactical, song, and thirteen-slot equipment state.<br><code>A5 · schema 8 · 6,478,670</code>: hazards, objects, interactions, prices, tells, explosives.<br><code>A6 · schema 9 · 6,478,798</code>: shop audio and bounds; same 512 LSTM and heads.<br><code>A7 · rejected · 6,401,648</code>: exact A2 plus a scalar-gated residual that stayed nearly closed.<br><code>A8 · candidate</code>: exact A2 plus the same sensory branch and a zero-output 512×512 projection that receives a full first-step gradient.`,
    steps: [['Embed', 'Turn categorical classes, exact-type hashes, positions, and numeric state into compact features.'], ['Perceive', 'Run local/map CNNs and entity/inventory attention encoders in parallel.'], ['Fuse', 'Project all streams into a shared 512-value latent.'], ['Remember', 'Update the LSTM hidden and cell state.'], ['Decide', 'Produce 11 masked action logits and one state-value estimate.']],
    cond: [{ q: 'Is A6 too small to learn the game?', r: 'The A6 pilot does not support that conclusion: it changed the latent interface and had only 51,200 new transitions versus A2’s 250,880. Capacity was not isolated (2026-08-24).' }, { q: 'Would simply scaling parameter count solve progression?', to: 'Establish a stable, behavior-preserving architecture and learning curve before running controlled width/depth scaling experiments.' }],
  },
  {
    id: 'T', code: 'T', name: 'Temporal memory', short: 'LSTM STATE', group: 'decide', gx: 10.5, gy: 10.5, w: 3, d: 3, h: 38, kind: 'store',
    one: 'The LSTM carries learned context that explicit map channels do not.',
    what: 'A separate hidden and cell state summarizes action history, recent encounters, timing, and other partially observable context. It complements, rather than replaces, the explicit floor map.',
    how: `<strong>Version lineage.</strong><br><code>A2–A6</code> all use the same 512-unit LSTMCell and store exact [hidden, cell] state at every transition.<br><code>Episode boundary</code> resets hidden state, previous action, and previous reward.<br><code>Worker recovery</code> discards the incomplete fragment and resets recurrent state for that slot. No architecture version has yet increased temporal-memory width.`,
    steps: [['Initialize', 'Create zero hidden and cell tensors at reset.'], ['Condition', 'Combine current fused perception with previous action and reward context.'], ['Update', 'Produce next hidden and cell state on every live transition.'], ['Replay', 'Seed each 32-step PPO chunk with the exact stored initial state.']],
    cond: [{ q: 'How much does the trained policy rely on its LSTM?', to: 'Run hidden-state ablations and delayed-information probes on fixed seeds without changing rewards.' }, { q: 'Should the map be left to the LSTM instead?', r: 'No; explicit player-like spatial memory removes an avoidable information burden, while the LSTM handles temporal context (2026-08-24).' }],
  },
  {
    id: 'R', code: 'R', name: 'Reward shaping', short: 'REWARD', group: 'decide', gx: 1, gy: 10.5, w: 3, d: 3, h: 54, kind: 'box',
    one: 'Bounded learning signals point toward the sparse goal without defining success.',
    what: 'A stateful tracker deduplicates and caps exploration, combat, item, and navigation credit. Task reward, shaping reward, and gameplay outcomes are recorded separately; held-out gameplay selects policies.',
    how: `<strong>Reward lineage.</strong><br><code>Legacy</code>: direct events proved the loop but learned passivity.<br><code>V1</code>: stateful exploration/combat/items improved local competence but no stairs.<br><code>V2 · retained</code>: stair discovery/distance plus turn/revisit pressure produced the best local play and 24 training floor transitions.<br><code>V3 · rejected</code>: bounded positive progression reduced timeouts but caused unsafe activity.<br><code>V4A/B · rejected</code>: restored competence signals with 0.5/1.0 stair potential but became overwhelmingly passive.<br><code>V5</code>: deliberately undefined until the action/architecture issue is isolated.`,
    steps: [['Observe transition', 'Compare authoritative events and episode-local state before and after the action.'], ['Deduplicate', 'Credit each enemy, item type, position, and revealed tile only under its defined rules.'], ['Cap', 'Keep renewable auxiliary positive credit below the +5 floor milestone.'], ['Split', 'Return total, task/extrinsic, shaping, and named component values.']],
    cond: [{ q: 'Should remaining stationary receive a generic penalty?', r: 'Not yet: A6’s unchanged-position turns were directional inputs and may include attacks or digging. Classify outcomes first (2026-08-24).' }, { q: 'What is Reward V5?', to: 'Defer its design until blocked-action behavior and function-preserving A7 transfer are tested.' }],
  },
  {
    id: 'C', code: 'C', name: 'Rollout actors', short: 'ACTOR FLEET', group: 'learn', gx: 1, gy: 15.5, w: 3, d: 3, h: 58, kind: 'cards',
    one: 'Independent actors gather one contiguous fragment per worker and policy version.',
    what: 'Each actor owns its worker, episode, seed stream, recurrent state, pipe operations, and recovery. A central scheduler batches whichever observations are ready without forcing a per-step global barrier.',
    how: `<strong>Version lineage.</strong><br><code>Vector barrier</code> dispatched one action to every worker and waited for the slowest after every step.<br><code>Versioned async</code> gives each slot an independent 128-transition state machine, dynamically batches inference with a 2 ms maximum delay, and assembles stable [time, worker] tensors only after all fragments finish.<br><code>Recovery</code> discards only an incomplete fragment and recollects it under the same frozen policy version.`,
    steps: [['Freeze', 'Hold model weights constant for this rollout version.'], ['Infer', 'Batch whichever actor observations arrive within the 2 ms window.'], ['Advance', 'Let every worker build its own contiguous 128-transition fragment.'], ['Recover', 'Replace and reset only a failed slot, then recollect its fragment.'], ['Assemble', 'Stack completed fragments in stable worker-slot order.']],
    cond: [{ q: 'Can a fast worker start using next-version weights early?', r: 'No; fragments never mix policy versions. Fast workers wait after completing their contribution (2026-08-24).' }, { q: 'Does asynchronous collection remove every barrier?', r: 'It removes the per-step barrier, but PPO still waits for one complete same-version fragment from every fixed-capacity slot (2026-08-24).' }],
  },
  {
    id: 'L', code: 'L', name: 'Recurrent PPO learner', short: 'PPO LEARNER', group: 'learn', gx: 5.5, gy: 15.5, w: 3.5, d: 3, h: 78, kind: 'job',
    one: 'Clipped PPO changes the shared policy from complete live-game rollouts.',
    what: 'The learner computes generalized advantage estimates, normalizes them, and replays exact recurrent chunks through clipped actor and value losses with entropy and gradient controls.',
    how: `<strong>Version lineage.</strong><br><code>Initial recurrent PPO</code> used the same core defaults but synchronous collection.<br><code>Architecture A2</code> established exact 32-step LSTM replay and architecture-checked checkpoints.<br><code>Async runtime</code> preserved the algorithm while feeding versioned actor fragments.<br><code>Current diagnostics</code> record one pre-clipping gradient snapshot per representation group on every update, alongside CUDA inference and optimization. Rollout cadence remains 128×N transitions per update. Defaults: γ .99, GAE .95, clip .2, lr 3e-4, entropy .01, value .5, grad norm .5, four epochs.`,
    steps: [['Bootstrap', 'Estimate the value after the final rollout state.'], ['Advantage', 'Compute backward GAE; true termination must stop value bootstrap, while time-limit truncation must bootstrap its final observation but stop the trace across reset.'], ['Chunk', 'Split each worker trajectory into 32-step sequences with stored initial LSTM state.'], ['Optimize', 'Shuffle chunks across four epochs and update clipped policy and critic objectives.'], ['Publish', 'Increment the policy version only after the whole update finishes.']],
    cond: [{ q: 'Is one update per 1,024 steps optimal at eight workers?', to: 'Benchmark rollout length, epochs, minibatch chunks, and GPU utilization while holding total environment steps and evaluation seeds fixed.' }, { q: 'Is γ=.99 aligned with multi-floor play?', to: 'No evidence establishes that. Its effective horizon is about 100 turns and γ^1000 is roughly 4.3e-5, while failed episodes last 3,000–5,000 turns. Test horizon only as a versioned optimization/objective change with critic stability controls and matching potential-shaping discount.' }, { q: 'May PPO train on a partial healthy fleet?', r: 'No; fixed capacity and same-version rollout integrity take precedence over silently continuing with fewer workers (2026-08-24).' }],
  },
  {
    id: 'K', code: 'K', name: 'Checkpoints and metrics', short: 'CHECKPOINTS', group: 'learn', gx: 10.5, gy: 15.5, w: 3, d: 3, h: 32, kind: 'store',
    one: 'Atomic artifacts make training recoverable and experiments auditable.',
    what: 'A checkpoint stores model, optimizer, global step, PPO configuration, architecture specification, reward specification, random states, and running metrics. Run directories keep resolved configuration and JSONL metrics.',
    how: `<strong>Version lineage.</strong><br><code>Early</code> saved working model state for the pipeline.<br><code>A2+</code> requires exact architecture and PPO configuration for resume.<br><code>Reward V4+</code> embeds exact reward arm weights to prevent cross-arm confusion.<br><code>Current</code> supports exact resume plus explicit partial A2 warm-start paths that reset critic and optimizer when semantics change. Writes are atomic.`,
    steps: [['Snapshot', 'Collect model, optimizer, counters, configs, RNG states, and metrics.'], ['Write atomically', 'Write a temporary artifact and replace the destination only when complete.'], ['Resume exactly', 'Validate architecture, PPO, and reward metadata before restoring all state.'], ['Warm-start explicitly', 'Transfer only declared compatible policy weights and initialize changed paths and critic afresh.']],
    cond: [{ q: 'Can an A2 checkpoint be silently resumed as A6?', r: 'No; exact resume rejects it. Only the named partial warm-start path may transfer compatible weights (2026-08-24).' }, { q: 'Which artifact is the measured baseline?', r: 'runs/reward-v2-250k/final.pt, evaluated as Reward V2 Architecture A2 (2026-08-24).' }],
  },
  {
    id: 'E', code: 'E', name: 'Evaluation and selection', short: 'EVALUATION', group: 'evidence', gx: 1, gy: 20.5, w: 3.5, d: 3, h: 48, kind: 'gate',
    one: 'Unseen-seed gameplay—not shaped return—decides whether a version advances.',
    what: 'Training reward is diagnostic, not the objective. Paired policies play fresh seeds with fresh recurrent state and a fixed turn cap; reports compare progression, death, timeouts, combat, items, movement, stairs, and runtime health.',
    how: `<strong>Version lineage.</strong><br><code>Baseline</code> compared checkpoint argmax play with masked random on held-out seeds.<br><code>Reward pilots</code> added paired multi-checkpoint gates and separated task, shaping, and gameplay metrics.<br><code>A6/A7/A8 gates</code> added transfer, representation, and long-horizon controls.<br><code>EXP-0009</code> tests argmax against turn-keyed stochastic samples because PPO trained a high-entropy sampled policy and stochastic collection progressed farther than deterministic evaluation.<br><code>Representation gate</code> perturbs one input group at a time and measures output sensitivity plus encoder gradients. A supported path below 1% of established-input medians is trace, not material.`,
    steps: [['Predeclare', 'Write hypotheses, seeds, budgets, metrics, and pass/fail gates before training.'], ['Probe representation', 'Require material counterfactual sensitivity and gradient reach before broad training.'], ['Calibrate execution', 'When policy entropy is material, compare argmax with reproducible stochastic sampling before attributing failure to learning.'], ['Evaluate', 'Play ordered unseen game seeds with fresh recurrent state.'], ['Aggregate', 'Preserve per-seed outcomes, repeated policy samples, and arm-level summaries.'], ['Decide', 'Promote, continue, or reject using gameplay-ranked rules.'], ['Record', 'Store the immutable experiment decision and supporting artifacts.']],
    cond: [{ q: 'Why was A6 rejected even though it has more information?', r: 'It failed the predeclared gameplay gate and produced worse local competence and more stationary outcomes within the pilot budget (2026-08-24).' }, { q: 'Does the A6 result prove explicit map memory is harmful?', r: 'No; the warm start randomly reinitialized the expanded fusion interface, so representation and transfer disruption were confounded (2026-08-24).' }],
  },
  {
    id: 'D', code: 'D', name: 'Symbolic dashboard', short: 'DASHBOARD', group: 'evidence', gx: 6, gy: 20.5, w: 3.5, d: 3, h: 44, kind: 'screen',
    one: 'A local page makes live training behavior and bottlenecks visible.',
    what: 'The dashboard renders every worker’s symbolic 21×21 view from telemetry and keeps health, episode, reward, action, PPO, throughput, latency, recovery, and fragment-straggler metrics on one screen.',
    how: `<strong>Version lineage.</strong><br><code>Initial</code> exposed training metrics only.<br><code>Symbolic workers</code> added local grid views without game-window fidelity or screenshot capture.<br><code>Compact layout</code> constrained variable-length sections so reward components could not push the whole page below the viewport.<br><code>Async fix</code> publishes each actor transition to the dashboard, restoring visible movement during direct-pipe collection. Implemented in <code>src/autodancer/training/dashboard.py</code>.`,
    steps: [['Publish', 'Collector sends the latest observation, action, reward, and worker info.'], ['Aggregate', 'Training loop attaches PPO, reward-component, throughput, and recovery metrics.'], ['Render', 'Local HTTP clients draw symbolic cells and compact metric panels.']],
    cond: [{ q: 'Does rendering symbolic workers affect the game workers?', r: 'Only modestly: it reuses observation tensors already built for training and avoids image capture or full game rendering (2026-08-24).' }, { q: 'Is the dashboard part of policy input?', r: 'No; it is an observer only and cannot control or alter gameplay (2026-08-24).' }],
  },
  {
    id: 'F', code: 'F', name: 'Architecture A7 adapter', short: 'A7 REJECTED', group: 'future', gx: 10.5, gy: 20.5, w: 3.5, d: 3, h: 62, kind: 'tall',
    one: 'Exact A2 initialization succeeded, but every new-input path remained below material influence.',
    what: 'A7 is an implemented experimental model. The complete A2 actor-critic remains intact while a separate branch reads only map, tactical, hazard, interaction, audio, and expanded inventory fields that A2 could not see.',
    how: `<strong>Measured lineage.</strong><br><code>Parity</code>: every A2 tensor—including critic—loaded under <code>base.*</code>, with exactly zero output and recurrent-state error at gate zero.<br><code>Training</code>: all three 51,200-step pilots were finite, but final gates were only −0.001276, +0.000265, and −0.000030.<br><code>Representation</code>: tactical grid, map, extended player, and extended inventory paths were numerically active, but zero of twelve checkpoint/group pairs reached both 1% sensitivity and gradient materiality; none changed an argmax action.<br><code>Gameplay</code>: two pilots reached floor 2, but items, timeouts, stationary behavior, and runtime health failed the declared gate.`,
    steps: [['Load', 'Copy every A2 checkpoint tensor into the preserved base and reset only the optimizer.'], ['Prove parity', 'Measure exact output and deterministic-action parity before live training.'], ['Adapt', 'Let PPO learn the scalar gate and residual while continuing to update the A2 base.'], ['Measure influence', 'Perturb each new input group and backpropagate to its encoder.'], ['Reject', 'Retain A2 after representation and held-out gameplay gates fail.']],
    cond: [{ q: 'Was A7 behavior-preserving at initialization?', r: 'Yes; unit tests and the real V2 checkpoint measured exact tensor and output parity, including reset-containing sequences (2026-08-24).' }, { q: 'Did the richer observations receive meaningful influence?', r: 'No: all twelve trained checkpoint/group checks were only trace-active, with zero material groups and zero argmax action changes (2026-08-24).' }, { q: 'What replaces the scalar-gate design?', to: 'Use a representation-learning preflight to test an exact-zero output projection or staged frozen-base adapter before live gameplay training.' }],
  },
  {
    id: 'H', code: 'H', name: 'Architecture A8 controls', short: 'A8 CONTROLS', group: 'future', gx: 14.5, gy: 20.5, w: 3.5, d: 3, h: 68, kind: 'tall',
    one: 'A8 uses the rich inputs, but longer training still did not improve held-out progression.',
    what: 'A8 preserves every A2 tensor and adds schema-9 perception through a zero-output 512×512 matrix projection. EXP-0007 proved exact initialization parity and material influence from all four new groups. EXP-0008 then continued A2 and A8 exactly to 250,880 transitions; neither improved final held-out progress, and A8 lost local combat and item competence relative to frozen A2.',
    how: `<strong>Qualified EXP-0007.</strong><br><code>A2 frozen</code>: broad progress 1.00, death 0.30, 44 kills, 37 items, unchanged position 0.913.<br><code>A8</code>: progress 1.00, death 0.20, 43 kills, 38 items, unchanged position 0.587.<br><strong>EXP-0008 final.</strong><br><code>A2 frozen</code>: progress 1.125 and Floor 2.<br><code>A2 continuation</code>: progress 1.00, 73 kills, 55 items.<br><code>A8 continuation</code>: progress 1.00, 38 kills, 28 items.<br><code>Interpretation</code>: added information and 250,880 transitions did not solve strategic navigation.`,
    steps: [['Prove parity', 'Load the real V2 checkpoint and require zero logits, value, recurrent-state, and action error.'], ['Open projection', 'Freeze A2 for ten updates while the full residual matrix receives the first gradients.'], ['Qualify broadly', 'Use the schema-10 controller and fresh held-out seeds to separate activity from gameplay progress.'], ['Continue exactly', 'Resume A2 and A8 model, critic, optimizer, counters, and RNG state to 250,880 transitions.'], ['Test emergence', 'Require A8 to beat continued A2 at two consecutive checkpoints including the final checkpoint.']],
    cond: [{ q: 'Does A8 replace A2?', r: 'No. EXP-0008 found no final held-out progress advantage at 250,880 transitions and worse local competence than frozen A2 (2026-08-26).' }, { q: 'Did A8 solve A7’s gradient starvation?', r: 'Yes: all four new observation groups were material at warmup and final, with exact A2 behavior at initialization (2026-08-26).' }, { q: 'Did A8 merely need more of the same training?', r: 'No evidence supports that explanation: the exact continuation to 250,880 transitions did not improve held-out floor progress (2026-08-26).' }],
  },
];

export const FLOWS = [
  { id: 'turn', name: 'One live Bard turn', hops: [
    ['P', 'A', 'masked action', { action: 'RIGHT', policy_version: 17 }, 'xy'],
    ['A', 'B', 'ACTION', { command_id: 884, action: 1 }, 'yx'],
    ['B', 'G', 'accepted input', { worker: 'worker-0003' }, 'xy'],
    ['G', 'B', 'resolved turn', { events: ['move'], terminal: false }, 'yx'],
    ['B', 'O', 'schema-9 record', { sequence: 885, acknowledged: 884 }, 'xy'],
    ['O', 'M', 'visible terrain', { grid: '21×21×29' }, 'yx'],
    ['M', 'P', 'floor memory', { map: '65×65×5' }, 'xy'],
    ['O', 'P', 'current state', { player: 21, inventory: '13×8' }, 'yx'],
    ['T', 'P', 'prior [h,c]', { width: 512 }, 'xy'],
    ['P', 'T', 'next [h,c]', { reset: false }, 'yx'],
  ] },
  { id: 'train', name: 'One policy update', hops: [
    ['W', 'G', 'N isolated games', { capacity: 8 }, 'xy'],
    ['O', 'C', 'ready observations', { max_batch_delay_ms: 2 }, 'yx'],
    ['R', 'C', 'reward components', { task: 0, shaping: 0.005 }, 'xy'],
    ['P', 'C', 'frozen-version inference', { policy_version: 17 }, 'yx'],
    ['C', 'L', 'rollout batch', { shape: '[128,8]', version: 17 }, 'xy'],
    ['L', 'P', 'updated weights', { policy_version: 18, epochs: 4 }, 'yx'],
    ['L', 'K', 'atomic snapshot', { global_step: 18432 }, 'xy'],
  ] },
  { id: 'evaluate', name: 'Choose a version', hops: [
    ['K', 'E', 'candidate checkpoints', { candidates: ['A2', 'A6-1', 'A6-2', 'A6-3'] }, 'xy'],
    ['E', 'W', 'ordered unseen seeds', { seeds: '57001–57024', execution: 'argmax + seeded samples' }, 'yx'],
    ['W', 'E', 'gameplay outcomes', { progress: 1.011, deaths: 0.322 }, 'xy'],
    ['E', 'D', 'report and diagnostics', { decision: 'retain A2' }, 'yx'],
  ] },
  { id: 'recover', name: 'Recover one worker', hops: [
    ['B', 'C', 'timeout/disconnect', { worker: 'worker-0005' }, 'xy'],
    ['C', 'W', 'replace slot', { discard_incomplete_fragment: true }, 'yx'],
    ['W', 'G', 'fresh owned process', { same_slot: 5 }, 'xy'],
    ['W', 'B', 'fresh pipe/session', { healthy_capacity: 8 }, 'yx'],
    ['C', 'T', 'clear recurrent state', { h: 0, c: 0 }, 'xy'],
  ] },
  { id: 'a7', name: 'Rejected A7 compatibility experiment', hops: [
    ['P', 'F', 'exact A2 path', { transferred: 'all compatible policy behavior' }, 'xy'],
    ['O', 'F', 'A6-only inputs', { branch: 'residual' }, 'yx'],
    ['F', 'A', 'gated logits', { initial_gate: 0 }, 'xy'],
    ['F', 'E', 'paired candidate', { reward: 'V2 unchanged' }, 'yx'],
  ] },
  { id: 'a8', name: 'Controlled A8 learning experiment', hops: [
    ['K', 'H', 'exact V2 A2', { parity_error: 0 }, 'xy'],
    ['O', 'H', 'schema-9-only inputs', { projection_initially_zero: true }, 'yx'],
    ['H', 'L', 'frozen-base warmup', { updates: 10 }, 'xy'],
    ['L', 'E', 'four-point curves', { controls: ['legacy-no-wait', 'current'] }, 'yx'],
    ['E', 'H', 'admit or stop', { broad_only_after_pass: true }, 'xy'],
  ] },
];

export const CH = [
  {
    id: 'turn', title: 'The authoritative turn', reveal: ['G', 'B', 'A'],
    lede: 'The irreducible loop is a logical action, a real engine turn, and one exact acknowledgement.',
    story: `<p>AutoDancer does not imitate the game. <mark>NecroDancer itself resolves every action.</mark> The bridge exists to make that live turn look like a strict RL transition without driving menus, keys, or screenshots.</p>`,
    flow: [['A', 'B', 'ACTION', { command_id: 884, action: 'RIGHT' }], ['B', 'G', 'accepted input', { safe_state: true }], ['G', 'B', 'resolved turn', { sequence: 885 }]],
  },
  {
    id: 'fleet', title: 'Many independent games', reveal: ['W'],
    lede: 'One Python supervisor turns the live loop into a fixed-capacity fleet.',
    story: `<p>Every slot has its own process, profile, pipe, identity, seed stream, and episode. A failure pauses only that slot’s fragment, but <mark>capacity never silently shrinks</mark>.</p>`,
    flow: [['W', 'G', 'launch exactly N', { workers: 8 }], ['G', 'B', 'worker readiness', { instance_id: 'worker-0003' }], ['B', 'W', 'identity validated', { healthy: true }]],
  },
  {
    id: 'sense', title: 'What Bard can know', reveal: ['O', 'M'],
    lede: 'Current perception and remembered space are separate, player-equivalent information sources.',
    story: `<p>The 21×21 observation describes now; the 65×65 map remembers revealed terrain and travel. The design excludes unseen terrain and stale enemies, so richer inputs do not become privileged engine state.</p>`,
    flow: [['G', 'B', 'engine components', { visible_only: true }], ['B', 'O', 'schema-9 record', { grid: '21×21×29' }], ['O', 'M', 'revealed cells', { world_coordinates: true }]],
  },
  {
    id: 'choose', title: 'Choosing with two memories', reveal: ['P', 'T'],
    lede: 'Spatial memory tells the policy where it has been; temporal memory carries learned context.',
    story: `<p>A hybrid actor-critic merges geometry, entities, inventory, player state, explicit map, and history. The map and LSTM are complementary: one is inspectable floor knowledge, the other is learned recurrent state.</p>`,
    flow: [['O', 'P', 'current state', { schema: 9 }], ['M', 'P', 'map memory', { channels: 5 }], ['T', 'P', 'prior [h,c]', { width: 512 }], ['P', 'A', 'masked action', { choices: 11 }], ['P', 'T', 'next [h,c]', { stored: true }]],
  },
  {
    id: 'reward', title: 'Scoring without redefining success', reveal: ['R'],
    lede: 'Reward shaping helps optimization; held-out gameplay remains the objective.',
    story: `<p>The tracker makes useful intermediate events learnable while bounding renewable credit. The history from Legacy through V4 is evidence that a higher or cleaner training return can still produce worse play.</p>`,
    flow: [['O', 'R', 'before/after state', { floor: 1, position_changed: true }], ['G', 'R', 'events', { enemy_kill: false }], ['R', 'P', 'previous reward context', { task: 0, shaping: 0.005 }]],
  },
  {
    id: 'collect', title: 'Gathering live experience', reveal: ['C'],
    lede: 'Workers advance independently, but every accepted rollout belongs to one frozen policy version.',
    story: `<p>The asynchronous collector removes the slowest-worker barrier from each turn. It still waits at the fragment boundary so PPO receives exactly 128 contiguous transitions from every slot with no version mixing.</p>`,
    flow: [['O', 'C', 'ready observation', { worker: 3 }], ['P', 'C', 'batched inference', { delay_ms: 2 }], ['R', 'C', 'transition reward', { named_components: true }], ['C', 'T', 'store exact state', { every_transition: true }]],
  },
  {
    id: 'learn', title: 'Changing the weights', reveal: ['L', 'K'],
    lede: 'Recurrent PPO replays short sequences, then publishes one new policy version.',
    story: `<p>At eight workers, the default learner updates after 1,024 fresh transitions. Exact recurrent states make 32-step replay faithful; atomic checkpoints make interruption and experiment comparison recoverable.</p>`,
    flow: [['C', 'L', 'same-version rollout', { transitions: 1024 }], ['L', 'P', 'four PPO epochs', { clip: 0.2 }], ['L', 'K', 'atomic checkpoint', { exact_resume: true }]],
  },
  {
    id: 'evidence', title: 'Keeping experiments honest', reveal: ['E', 'D'],
    lede: 'Representation gates admit candidates; held-out gameplay decides promotion.',
    story: `<p>A candidate must first prove that its new inputs can influence outputs and receive gradients materially. Evaluation uses unseen ordered game seeds and predeclared gameplay gates. EXP-0009 additionally calibrates deterministic argmax against reproducible stochastic execution because PPO trained a sampled policy. The dashboard is operational visibility—not an input.</p>`,
    flow: [['K', 'E', 'candidate', { architecture: 6 }], ['E', 'W', 'unseen seeds', { count: 30 }], ['W', 'E', 'outcomes', { mean_floor_progress: 1.011 }], ['E', 'D', 'decision', { retain: 'A2' }]],
  },
  {
    id: 'next', title: 'From A8 horizon rejection to Zone 2', reveal: ['F', 'H', 'E'],
    lede: 'More information and longer training did not solve progression; execution semantics are the next isolated variable.',
    story: `<p>EXP-0008 rejected the claim that A8 only needed a longer adaptation horizon: neither continued arm improved final held-out progress at 250,880 transitions. The sampled training policy nevertheless reached Floor 3 while argmax evaluation reached only Floor 2. <mark>EXP-0009 freezes every checkpoint and tests whether reproducible stochastic execution reaches Zone 2 across multiple unseen seeds.</mark> If it does not, the next intervention corrects truncation bootstrapping and reward-objective scale before introducing another architecture.</p>`,
    flow: [['K', 'E', 'frozen A2/A8 checkpoints', { changed_blocks: ['evaluation'] }], ['E', 'W', 'same unseen seeds', { count: 24 }], ['W', 'E', 'argmax + two sample streams', { policy_seeds: [0, 91001, 91002] }], ['E', 'D', 'Zone 2 gate', { distinct_seeds: 3, repeated_seeds: 2 }]],
  },
  {
    id: 'all', title: 'The whole agent system', reveal: [],
    lede: 'Explore every building block, version lineage, flow, decision, and open question.',
    story: `<p>Choose a flow at bottom left. Hover any structure for its summary, click to pin it, and use the arrow to go inside. The <mark>Open questions</mark> tab is the forward work queue; resolved questions preserve why the current design exists.</p>`,
    flow: null,
  },
];

export const HOW_HTML = `<div class="eyebrow">AutoDancer · live-game RL</div><h1 class="t">How it is built</h1><div class="sub">one real engine loop, many isolated actors, one shared learner</div>
<h3 class="sec">Current contract</h3><p><strong>Windows · NecroDancer v4.2.1-b5713 · Bard · normal seeded All Zones · schema 9.</strong> Python owns exactly N hidden workers and talks to each through a dedicated duplex named pipe.</p>
<h3 class="sec">Default learning unit</h3><pre>8 workers × 128 transitions
→ stable [time, worker] rollout
→ 32-step recurrent chunks
→ 4 clipped PPO epochs
→ one new policy version</pre>
<h3 class="sec">Version boundaries</h3><p>Observation schema, model architecture, PPO configuration, and reward specification are stored with checkpoints. Exact resume rejects mismatches; cross-version transfer must use an explicit warm-start path.</p>
<h3 class="sec">Source map</h3><pre>${META.filesystem}</pre>
<h3 class="sec">Evidence rule</h3><p>Shaped return is never a promotion metric. The system compares gameplay on held-out seeds under predeclared execution semantics and records accepted and rejected hypotheses with immutable experiment decisions.</p>`;
