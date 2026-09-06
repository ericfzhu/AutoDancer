# Live worker scaling

This bounded experiment compares the existing inference transfer path (`legacy`)
with reusable pinned host/device input buffers and one packed CPU result transfer
per batch (`packed`). Training exposes this as `--inference-transfer-mode` and
keeps `legacy` as the default. Neither path changes the policy architecture,
reward function, or checkpoint weights.

## Results after reboot (2026-09-07)

The machine reports 31.92 GiB physical RAM. Available RAM after reboot was
19.4 GiB, compared with roughly 12–13 GiB before reboot. No applications or
services were disabled by this experiment.

| Workers | Legacy steps/s | Packed steps/s | Packed change |
| --- | ---: | ---: | ---: |
| 1 | 23.99 | 25.32 | +5.6% |
| 2 | 39.72 | 35.54 | -10.5% |
| 4 | 48.51 | 48.73 | +0.5% |
| 8 | 65.41 | 67.56 | +3.3% |
| 16 | Skipped | Skipped | Insufficient estimated RAM headroom |

All eight executed variants completed with zero worker restarts. Eight workers
were the fastest tested capacity: about 2.7 times the single-worker throughput
within either mode. At eight workers, packed reduced explicit result transfers
from 6,144 to 520, but improved observed throughput by only 3.3%. The mixed
results and short trials do not justify changing the default from legacy.

The eight-worker legacy trial had mean inference batch size 3.84, mean queue
wait 33.3 ms, system CPU utilization 45.8%, and GPU utilization 26.9%. Available
RAM reached a sampled minimum of 8.44 GiB. Worker RSS summed to 9.72 GiB.
The packed trial had mean batch size 3.94 and queue wait 31.9 ms. Low aggregate
CPU/GPU utilization does not rule out a serial CPU bottleneck or latency limits.

Launching eight workers took 66–67 seconds per variant, separate from the
30–31 seconds of measured collection. Reusing a healthy worker pool is therefore
worth evaluating for short experiments, with explicit resets between trials.

The guard skipped both 16-worker variants: it estimated 22.79 GiB required
against 18.86 GiB available immediately before launch. This is not a measured
16-worker failure or proof that 16 workers cannot fit. Eight is the best tested
capacity, not an established optimum among every possible worker count.

Use eight workers as the next training pilot configuration, while retaining
headroom for PPO buffers and the guide model absent from this benchmark. Further
work should measure inference scheduling and game-step latency, and the separate
guide warm-up batching opportunity described in the performance report. These
results do not establish an order-of-magnitude gain from transfer packing.

Local evidence: `runs/performance-project/scaling-post-reboot/scaling.json`
and its per-variant directories. This benchmark did not change training defaults.

## Method

The scaling harness uses the frozen A8 checkpoint from
`runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt`, CUDA,
seed 103001, normal Bard starts, and `map-navigation-prior-v1`. Each variant
launches fresh game workers, warms up 32 actions per worker, and measures two
128-action fragments per worker. The scheduler has a 2 ms batching delay.
Each capacity runs legacy followed by packed. Startup and initial reset are
reported separately from collection throughput.

These are short collection measurements, not end-to-end PPO training or the
Death Metal guide-prefix curriculum. There are no learning updates or checkpoint
promotions. Variant order is fixed and timings have no repeated-trial confidence
intervals; small differences should not be treated as established gains.

Before each launch, a conservative memory guard estimates worker RSS plus a
1 GiB reserve. It starts at 1.25 GiB per worker and increases to 110% of the
largest observed worker RSS. A skipped capacity is untested, not proven
impossible. Summed worker RSS can include shared pages. CPU, GPU, and available
RAM samples are system-wide. Queue and stage timings are host wall times rather
than isolated CUDA kernel timings.

## Reproduce

Choose a fresh output directory:

```powershell
.venv/Scripts/python.exe -m autodancer.training.scaling `
  --game-dir 'X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64' `
  --mod-dir mods/AutoDancer `
  --checkpoint runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt `
  --output runs/performance-project/scaling-repeat `
  --capacities 1 2 4 8 16
```

The output contains `scaling.json` and per-variant `result.json` files, including
batch histograms, worker stage times, resource samples, startup costs, and restart
counts. A controller recovery invalidates the run. An error stops the sweep.

## Validation

`tests/test_inference_transport.py` compares real model outputs on CPU and CUDA,
checks action-mask handling, and verifies retained results survive staging-buffer
reuse. The existing collector and training tests cover the default path.
