# Fixed-action engine pacing test — 2026-09-07

## Result

Disabling the frame cap and skipping game drawing improved one-worker fixed-action
replay throughput by about **1.3x**, not 10x. Worker CPU consumption increased
substantially, and resident memory stayed around 1.25 GiB. Do not enable these
settings globally on the assumption that they improve multi-worker training.

| Measurement | Default | Uncapped, VSync off | Uncapped, VSync off, game drawing skipped |
| --- | ---: | ---: | ---: |
| Verified frame limit | 60 | 0 (unlimited) | 0 (unlimited) |
| Observed ticks/s, whole sampled run | 58 | 527 | 2,264 |
| All action calls/s, including exit transitions | 41.30 | 50.64 | 55.31 |
| Speedup of action calls | 1.00x | 1.23x | 1.34x |
| Ordinary action calls/s, excluding terminal transitions | 59.72 | 79.60 | 90.49 |
| Median ordinary action call | 16.38 ms | 11.88 ms | 10.27 ms |
| Mean terminal/level-transition action | 701 ms | 671 ms | 655 ms |
| Mean episode reset | 758 ms | 692 ms | 619 ms |
| Replay plus reset total, nine episodes | 27.06 s | 22.81 s | 20.76 s |
| Replay plus reset speedup | 1.00x | 1.19x | 1.30x |
| Worker CPU seconds during replay, all threads | 8.41 | 39.13 | 41.50 |
| Maximum sampled worker RSS | 1.25 GiB | 1.26 GiB | 1.27 GiB |

CPU time is summed across process threads and can exceed wall time. The uncapped
modes spent about 4.7–4.9 times as many CPU seconds for the same actions. This can
harm worker scaling even though one worker finishes sooner. FPS is not simulation
throughput. These results do not establish that the remaining action latency is
pure game-rule computation; input buffering, clocks, transport, and observation
processing are still present.

## Workload and correctness

Each mode used one worker, seeds 92008/92096/92116 repeated three times, and the
qualified full action traces with player20 assistance from level 4 to level 5.
Every mode completed 825 actions and nine curriculum episodes, with zero restarts.
There was no policy inference, PPO, or checkpoint promotion. Reset and startup were
measured separately from action-call throughput. The baseline and candidates ran
sequentially in fixed order; three repeats are not independent fresh-process trials.

All 2,475 actions and 27 resets in the main comparison matched their qualified
normalized observation digests. Per-action rewards matched the baseline, and
terminations/completions matched. Full observations were not identical: normalized
qualification deliberately excludes elapsed/remaining music-time fields. This is
observational replay parity over these seeds, not a proof of identical hidden RNG
state, simulation clocks, or arbitrary gameplay.

## Rendering verification

The skip hook uses the existing render-UI event to set `renderGame=false`. It skips
game drawing, not the whole graphics host, UI, asset loading, audio, or every visual
update. Consequently, this is not a fully headless engine test.

The first render counter was registered in the `catchErrors` stage and
reported zero even in the baseline; those counter values are not useful evidence.
A separate three-seed check moved the counter to the `overlay` stage and observed
**498 game-render callbacks in the default mode versus zero in skip mode**, with
15,247 skip-hook invocations. All replay digests still matched. That check measured
41.41 versus 52.09 action calls/s (1.26x), consistent with a modest, variable gain.

The engine's tick/render/frame counters are retained as raw diagnostic values.
Their scopes can overlap and were not calibrated into mutually exclusive CPU
fractions. In particular, the default render counter is near the frame interval;
do not claim that entire interval is drawing computation. No exact sleep-time,
serialization-time, or pure simulation-time fraction is established by this test.

## Isolation and a timing assumption uncovered

Initial smoke runs replayed successfully but did not load the probe from copied
worker profiles. A command-line array override also failed to establish the desired
mod paths. Those attempts were rejected as instrumented measurements. The successful
experiment used private copies of the game host, a private JSON configuration with
an explicit mod path, and a copied AutoDancer mod. Original game assets were read
from their existing directory. The shared installed mod was not synchronized or
modified. Installed executable/archive/library/config hashes matched the preceding
inspection after the main comparison.

An initial render-skipping run hit the bridge's 600-tick action watchdog during a
normal exit transition: at high FPS, its nominal 10-second allowance shrank below
one second. The successful comparison changed **only the private bridge copies**
to use elapsed-time limits equivalent to the original 60-Hz watchdogs (10 seconds
for actions and 45 seconds for queued reset acceptance). Every measured mode used
the same correction. The normal Python timeouts remained active. Other tick-based
assumptions, including startup/grace/heartbeat behavior, still need examination
before treating arbitrary frame rates as production-safe.

The production bridge, training settings, and installed game files remain unchanged.

## Implication for a training port

Removing display pacing and game drawing alone is not the large gain we hoped for.
A training-oriented host would need to advance simulation directly with an explicit
clock, bypass interactive input scheduling, and avoid expensive episode/level
transitions through validated state restoration. It could also avoid loading the
assets and subsystems still present here. The current test neither proves nor
rules out 10x faster environment execution through those deeper changes.

## Reproduce and evidence

```powershell
.venv/Scripts/python.exe tools/run_engine_pacing_probe.py `
  --output runs/performance-project/engine-pacing-repeat --repeats 3
```

The harness refuses existing output, validates runtime frame limits and render
counters, saves all replay signatures and timings, and closes workers on exit.
It uses a private game-host copy, so allow disk space for copied libraries/packages.

Main evidence: `runs/performance-project/engine-pacing-wallclock2/results.json`
and `summary.json`. Render verification: `engine-render-counter-check/`.
Earlier smoke/failed attempts remain separate. The current harness uses the
corrected overlay counter and rejects an unverified render mode. Ruff passed;
validation for this diagnostic tool is the bounded live replay described above.
