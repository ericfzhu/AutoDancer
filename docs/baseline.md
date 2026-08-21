# Live Bard PPO baseline

This is the first measured AutoDancer baseline against the live game. It is a
reference point for future experiments, not a claim of competent play.

## Run

- Date: 2026-08-21
- Game: Crypt of the NecroDancer `v4.2.1-b5713`, Steam build `22938426`
- Character/mode: Bard, normal seeded All Zones
- Workers: 8 live workers plus one hidden coordinator
- Training: 8,192 live transitions, 8 recurrent PPO updates
- Rollout/chunk: 128 transitions per worker, 32-step recurrent chunks
- Training seed: 19001
- Training throughput at completion: 6.88 transitions/second
- Worker restarts: 0
- Checkpoint SHA-256:
  `8786df439cb6e9cda8c3a7f3715f3619ff53998a564772af853e1906d236631f`

Training observed 157 completed episodes, 9 enemy kills, and 4 item pickups.
All 25 model tensors changed from initialization and remained finite.

## Held-out evaluation

Both policies ran on the same 16 unseen seeds, 20001 through 20016, with a
128-turn episode cap. The reference selected uniformly from each observation's
legal action mask. The checkpoint policy selected deterministic argmax actions.

| Metric | Masked random | PPO checkpoint | Delta |
| --- | ---: | ---: | ---: |
| Death rate | 93.75% | 6.25% | -87.50 pp |
| Mean turns | 19.19 | 121.44 | +102.25 |
| Mean shaped return | -1.3504 | -0.2277 | +1.1228 |
| Player damage | 63 | 7 | -56 |
| Completion rate | 0% | 0% | 0 pp |
| Enemy kills | 0 | 0 | 0 |
| Item pickups | 0 | 0 | 0 |
| Furthest floor | 1-1 | 1-1 | none |

The checkpoint has learned a strong survival/passivity policy. It is clearly
different from random play, but it has not learned useful exploration, combat,
collection, or floor completion on held-out seeds. This is the behavior future
runs must beat; survival alone is not sufficient.

## Artifacts

The complete per-seed results, configuration, metrics, and resumable checkpoint
are under `.runtime/baseline-bard-live-8w-20260821/`:

- `baseline.json`: full fixed-seed reference/checkpoint comparison
- `final.pt`: model, optimizer, RNG state, metrics, and counters
- `latest.pt`: latest atomic checkpoint
- `metrics.jsonl`: one training record per PPO update
- `config.json`: PPO and supervisor configuration
