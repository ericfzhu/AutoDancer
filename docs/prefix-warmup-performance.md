# Batched learner warm-up over qualified guide traces

## Pilot result (2026-09-07)

The eight-worker pilot reduced measured collection time by **21.2%** and total
training-loop time by **18.1%**. Retain this as an opt-in performance setting for
qualified warm trace curricula; one pair does not establish long-run learning
equivalence or justify changing every training default.

| Measurement | Online warm-up | Batched warm-up (16) |
| --- | ---: | ---: |
| Collection | 66.766 s | 52.625 s |
| PPO update | 10.547 s | 10.687 s |
| Total measured training loop | 77.391 s | 63.390 s |
| Learner actions | 1,024 | 1,024 |
| Guide actions | 2,759 | 2,759 |
| Completed curriculum episodes | 56 | 56 |
| Worker restarts | 0 | 0 |

The preceding live validation trial checked 62 nonempty prefix handoffs covering
2,759 guide actions. All final hidden/cell states, handoff action probabilities,
and handoff values passed the numerical gate. Empty prefixes require no warm-up.
The validation trial also completed its PPO update without collector recovery.

In the separate timing runs, 55 of 56 completed episodes had identical action
sequences when matched by worker and episode order. Worker 0006's eighth episode
(seed 92008) differed in its first four actions but still completed in 13 turns.
Both runs had the same completion and guide-action counts. Actual wall-clock
observation fields and batching-dependent floating-point differences can vary
between launches; this experiment does not identify the cause of that divergence.
Do not present these measurements as bitwise continuation or identical training.

On the slowest worker, online prefix inference took 26.416 s; batched warm-up,
including queueing and packing, took 4.048 s. Total prefix time fell from 41.797 s
to 25.188 s. Some savings were offset by increased guide game-step time (13.173
to 16.838 s) and learner inference wait (6.683 to 8.435 s). This makes scheduler
fairness between long warm-up jobs and learner actions a useful follow-up, rather
than assuming every removed inference call becomes an equal wall-time saving.

These are one ordered baseline/candidate pair on the same machine, without
confidence intervals. Total loop time excludes initial worker launch, final
checkpoint writing, and shutdown; it includes the measured update overhead.
Initial process startup remains a substantial separate cost for short trials.

The final targeted test suite passed **91 tests** across performance, collector,
training, inference transport, and trace-prefix tests. Tests cover A2/A6/A7/A8
recurrent parity, partial encoder chunks, retained outputs, rejected incorrect
warm states, and trace divergence checks under deferred replay. Ruff passed.

Evidence: `runs/performance-project/prefix-warmup/summary.json`, and the `verify`,
`online`, and `batched` directories containing configs, metrics, episode journals,
and resulting checkpoints. No checkpoint was promoted.

## Implementation

The qualified-trace collector can replay its predetermined guide actions without
running learner inference at every game step. It copies each actual policy input,
including action-contract observations, previous action, and policy-feedback
reward. At handoff, the inference scheduler encodes those observations in chunks
and advances the LSTM in order, returning the final hidden and cell state.

Enable this experimental path with `--trace-prefix-warmup-batch-size 16`.
The default is zero, retaining online warm-up. This option requires a qualified
trace prefix in warm recurrent-state mode. It does not apply to natural guide
policies, whose actions still require online inference.

## Correctness boundary

Every guide action still runs in the real game and undergoes the existing action
mask, trace digest, termination, and truncation checks. Reward trackers, map
memory, telemetry, and previous-action/reward bookkeeping advance as before.
Recorded observations are copied before a game step can mutate their arrays.
The handoff observation is not consumed by warm-up; it remains the learner's
first action input. PPO receives the warm hidden state at its episode boundary.

There is no cache across episodes or policy updates. CPU storage scales with
prefix length; GPU encoder workspaces are bounded by the selected batch size.
Warm-up requests share the existing inference thread with learner action requests,
so a long warm-up can delay other workers. This is measured as part of collection.

`--trace-prefix-warmup-verify` additionally runs the original online warm-up on
the same live inputs. It asserts final hidden/cell-state, handoff action-probability,
and handoff value parity at rtol=2e-4 and atol=2e-5. A mismatch aborts training.
This verification adds work and must be excluded from speed comparisons.
Floating-point batching differences remain possible even when this gate passes;
it does not establish identical long-run stochastic learning trajectories.

The flags are recorded in run configuration and lineage parameters. Pass them
again on resume; model architecture and checkpoint parameter shapes are unchanged.
Scheduler metrics record warm-up request, step, and verification counts. Worker
`prefix_batched_warmup_seconds` includes queueing, packing, and recurrent work;
it is nested inside `prefix_total_seconds`. Worker times overlap.

## Reproduce the bounded pilot

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-prefix-warmup-pilot.ps1 `
  -RunDir runs/performance-project/prefix-warmup-repeat
```

This launches eight workers for each of three fresh trials: verification, online
warm-up, then batched warm-up. Every trial initializes from the same A8 checkpoint
and seed, uses the qualified tail-32-through-60 Death Metal window, and collects
1,024 learner actions followed by one four-epoch PPO update. PPO encoder batch 16
and legacy inference transfers are fixed across trials. Evaluation is disabled.
The launcher refuses existing output and stops if training or verification fails.
It does not promote a policy or change training defaults.
