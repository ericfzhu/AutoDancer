# Bounded performance project — 2026-09-06

## Decision

Keep the real engine for now. Use the optional PPO sequence encoder batch of 16
for explicitly declared performance trials. The subsequent
[guide warm-up pilot](prefix-warmup-performance.md) measured a further 18.1%
loop-time reduction with batched learner warm-up over actual guide observations.
Both optimizations remain opt-in. A whole-game port is not justified by the
measured bottleneck yet.

The default training path is unchanged. Enable the measured candidate with:

```powershell
--sequence-encoding-batch-size 16
```

This is a runtime optimization, not an architecture or reward change. It applies
to PPO updates only; action collection and imitation retain their existing paths.
Parameters and checkpoint shapes are unchanged. Batch-dependent floating-point
differences mean this is not bitwise continuation; declare the setting in a new
trial and pass it again on resume. It is recorded in config.json, metrics, and
lineage parameters. Other batch sizes require their own numerical/memory checks.

## Measurements

Machine: NVIDIA RTX 3070, 8 GB VRAM; PyTorch 2.11.0+cu128.

The three historical EXP-0034 trials each collected 92,160 learner transitions in
2.62–2.75 hours of measured training-loop time. Collection consumed 78.0–78.2%.
The mean fastest-to-slowest fragment gap was 20.4–23.5 seconds. Historical logs
did not separate PPO updates from other work outside collection; no such split
is inferred retroactively.

Two fresh eight-worker trials then ran exactly one 1,024-transition rollout and
one PPO update each. Both initialized from EXP-0034 seed-103001's final checkpoint
using function-preserving fine-tuning with a fresh optimizer, seed 103001,
DeathMetalPotentialV5, player20 assistance, and the qualified tail-32-through-60
window. Both used four PPO epochs. No gameplay promotion is implied.

| Training-loop measurement | Historical encoder loop | Encoder batch 16 |
| --- | ---: | ---: |
| Collection | 70.515 s | 67.375 s |
| PPO update | 22.110 s | 12.109 s |
| Total measured loop | 92.844 s | 79.625 s |
| Learner actions | 1,024 | 1,024 |
| Guide actions | 2,759 | 2,759 |
| Completed curriculum episodes | 56 | 56 |
| Controller restarts | 0 | 0 |

PPO was 1.83x faster. The measured loop was 1.17x faster (14.2% less time), but
collection also varied by 3.14 seconds even though its implementation was unchanged.
Holding baseline collection constant gives an approximately 10.8% loop-time
reduction attributable to the measured update saving. This is one ordered pair,
not a replicated throughput estimate or evidence of long-run learning equivalence.
Initial worker launch, final checkpoint writing, and shutdown are excluded from
this table. Periodic evaluation and the dashboard were disabled in both trials.

The baseline's slowest worker gives a useful breakdown of the collection path:

| Stage | Wall time on that worker |
| --- | ---: |
| Whole worker fragment | 70.485 s |
| Whole guide-prefix processing | 44.641 s |
| ↳ Learner inference during prefix | 28.875 s |
| ↳ Game step during prefix | 13.545 s |
| Learner-action inference | 6.972 s |
| Learner-action game step | 10.233 s |
| Resets during collection | 7.814 s |

Prefix inference includes scheduler queueing and transfers, not just GPU kernels.
Game-step time includes transport and Python environment processing, not just
engine compute. Prefix totals contain their sub-stages: do not add them twice.
Worker times overlap across threads and must not be summed as elapsed time.
The remaining fragment time covers accounting, observation copies, and validation.

## Implementation and validation

`evaluate_sequence_batched` flattens batch and time for independent observation
encoders, then unrolls the LSTM in temporal order. It retains stored initial
states, resets and warm episode boundaries, action masks, and previous-action/
reward inputs. Actor and critic heads process the resulting sequence together.
Encoder chunking bounds temporary workspaces, not all saved backward activations.

A game-free, production-shape A8 forward/backward benchmark measured:

| Encoder batch | Median minibatch time | Peak allocated GPU memory | Numerical gate |
| --- | ---: | ---: | --- |
| Historical loop | 1.029 s | 1.729 GB | Reference |
| 16 | 0.498 s | 1.722 GB | Passed |
| 32 | 0.270 s | 1.722 GB | Failed gradient tolerance |

These are three measured repetitions per variant after warm-up. Memory is PyTorch
allocated memory, not total GPU usage. Inputs are synthetic with production
dimensions, so these numbers alone do not establish a live speedup. Batch 32 is
not recommended: its speed does not override the numerical gate.

CPU tests compare every returned value and every participating parameter gradient
across A2, A6, A7, and A8, with nonzero adapters, masks, and cold/warm boundaries.
They use rtol=2e-4 and atol=2e-5; the GPU benchmark applies the same tolerances.
The selected performance, collector, training, and imitation suites passed
86 tests. Ruff checks pass for the changed Python code.

## Timing instrumentation

Every collected fragment now reports stage seconds and call counts in
`worker_stage_timings`, alongside `performance_timing_version=1`:

- `learner_environment`, `learner_inference`, and `learner_reset`;
- `prefix_environment`, `prefix_inference`, `guide_inference`, and `prefix_reset`;
- `prefix_total`, which includes prefix sub-stages;
- `telemetry`, when a callback is active.

Training also records `ppo_update_seconds`, optional `imitation_update_seconds`,
and `evaluation_and_reset_seconds` when periodic evaluation runs. Initial resets
are outside fragment measurements. Recovery discards the failed fragment's stage
timings, so qualify performance results with zero recovery counts. Timers do not
insert per-stage GPU synchronization into collection; synchronous inference
responses already materialize their action/value outputs. The isolated GPU
benchmark explicitly synchronizes at measurement boundaries.

## Engine and snapshot feasibility

The supervisor's symbolic profile makes small hidden windows. That alone does
not establish that rendering or audio computation is disabled. The installed
game configuration declares a 60 Hz framerate, and bridge commands are polled
on `event.tick`; this warrants investigation but does not prove a frame-rate cap
is the dominant cost. Worker startup is sequential and took roughly a minute to
launch eight workers in the pilot, which also matters for short experiments.

Read-only inspection of the installed WSP archive found compiled Lua modules for
Snapshot, GameState, Rollback, SnapshotRequest, and ReplaySeek. This is evidence
of internal snapshot/replay machinery, not a verified external reset API. No game
archive, game configuration, Lua bridge, or native binary was changed by this work.
The live launches produced their normal logs and isolated worker profiles.

Before using snapshots for training, prove restoration of simulation RNG, entities,
timers, pending events, and level state. Restore Python map/action/reward bookkeeping
too, and recompute the learner state under the current policy.

## Guide warm-up follow-up

The subsequent [batched guide warm-up pilot](prefix-warmup-performance.md)
implements and validates the experiment proposed below. See that report for
the current measurements and runtime flags.

Batch guide warm-up before attempting snapshots. Qualified trace actions are
already predetermined, so learner inference could be deferred until the handoff:
retain each actual observation and feedback value while replaying the game, then
encode and unroll the sequence in batches. This can remove thousands of sequential
inference round trips while preserving the live trajectory.

Do not blindly cache warm hidden states by game seed. The trace digest explicitly
excludes wall-clock-derived music fields, which the policy still observes; matching
trace digests do not establish identical policy inputs. Cached hidden states would
also become stale after a PPO update. A deferred warm-up prototype must compare
the full final hidden/cell state and next-action distribution against online warm-up
on actual observations before a live timing trial.

If that succeeds, reassess remaining engine time. Only then prototype a minimal
host/snapshot interface or a small Death Metal simulator, with transition parity
and live transfer as explicit gates. The current timings do not support a claimed
10x or 100x end-to-end gain from removing the engine alone.

## Reproduce

For the subsequent frozen-policy worker-count and inference-transfer sweep,
see [Live worker scaling](worker-scaling.md).

```powershell
.venv/Scripts/python.exe -m autodancer.training.performance `
  --output runs/performance-project/sequence-repeat.json sequence `
  runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt `
  --encoder-batches 16 32 --repeats 3

.venv/Scripts/python.exe -m autodancer.training.performance `
  --output runs/performance-project/live-summary-repeat.json summarize `
  runs/performance-project/live/encoder-0/metrics.jsonl `
  runs/performance-project/live/encoder-16/metrics.jsonl

powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-performance-pilot.ps1 `
  -RunDir runs/performance-project/live-repeat
```

The live launcher refuses an existing output directory and stops after any failure.
It does not register or promote a learning experiment. The summary command requires
one uninterrupted process log per file; it cannot infer arbitrary resume boundaries
from old cumulative-throughput logs.

Local evidence: `runs/performance-project/historical.json`, `sequence-a8.json`,
`live-summary.json`, and both `live/encoder-*/` directories. The historical
experiments and source checkpoints were not modified.
