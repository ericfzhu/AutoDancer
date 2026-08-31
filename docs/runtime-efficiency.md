# Live-worker runtime efficiency

Validated on 2026-08-22 using NecroDancer `v4.2.1-b5713`, a Ryzen 7 3700X,
32 GB RAM, and eight hidden Bard workers.

## Result

- Previous eight-worker live-training baseline: 5.32 transitions/second.
- Direct-pipe fixed-action benchmark: 38.64 transitions/second (7.27×).
- Pipe latency: 16 ms median, 63 ms p95, 106 ms p99.
- Versioned async collector: 79–81 transitions/second in the two-update smoke test.
- End-to-end CPU-only PPO: 16.97 transitions/second across two finite updates.
- Restarts in the normal acceptance run: zero.

The fixed-action report is stored under
`.runtime/efficiency-8w-acceptance/benchmark.json`; runtime artifacts are ignored
by Git.

## Forced recovery

One owned worker was force-terminated during a four-update run. The supervisor
replaced exactly that slot, discarded its incomplete fragment, reset its LSTM
state, and recollected under the unchanged policy version. Training completed
at 4096 transitions with finite losses, exactly eight healthy workers, and one
recorded restart. The rollout immediately following recovery reached 88.68
collector transitions/second; final end-to-end throughput was 16.36.

## Architecture

Commands and schema-10 JSON/status messages share one per-worker 256 KiB duplex
named pipe. Pipe `HELLO` messages establish readiness; logs are only diagnostic
evidence. Each worker
uses an isolated ephemeral profile, minimized hidden rendering, disabled asset
reload and online integrations, and a dedicated recurrent actor state. A single
inference scheduler dynamically micro-batches ready actors against a frozen
policy. PPO updates only after every slot supplies one contiguous 128-step
fragment from that policy version.

## Optional Steam playtime presence

The diagnostic `--steam-presence-worker` option nominates one zero-based slot;
only that worker initializes Steamworks and the exact slot is recorded in run
metadata. It accrues ordinary wall-clock playtime, not the sum of parallel
worker-hours.

Do not use this option for controlled training or evaluation. Live replay tests
on the supported build showed that Steam initialization changes generated grid
state and gameplay outcomes for otherwise identical seeds and action sequences.
It is retained for explicit diagnostics and benchmarking while a separate,
non-actor presence mechanism is investigated.
