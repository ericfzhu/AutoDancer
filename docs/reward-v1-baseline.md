# Schema-5 reward-v1 live baseline

This experiment measures whether the progress-first reward profile changes the
passive behavior observed in the original live baseline.

## Run

- Date: 2026-08-22
- Game: Crypt of the NecroDancer `v4.2.1-b5713`, Steam build `22938426`
- Character/mode: Bard, normal seeded All Zones
- Workers: 8 live workers plus one hidden coordinator
- Training: 8,192 live transitions, 8 recurrent PPO updates
- Architecture: schema-5 hybrid CNN/entity-attention/inventory-attention/LSTM
- Reward: profile version 1
- Rollout/chunk: 128 transitions per worker, 32-step recurrent chunks
- Training seed: 220826
- Final throughput: 5.32 transitions/second
- Training worker restarts: 1, automatically restored to fixed capacity
- Checkpoint SHA-256:
  `b4615b1f9c26fca4706d7c438ae309033f29d9f625a1130a950888bfa2fd6d85`

Training observed 110 completed episodes, 104 deaths, 10 enemy kills, and 2
currency-pickup events. Losses remained finite. The final rollout had no deaths
but also very little exploration, so fixed-seed evaluation was required to
separate useful survival from passivity.

## Held-out evaluation

Both policies used seeds 20001 through 20016 with a 128-turn cap. The reference
sampled uniformly from legal actions; the checkpoint used deterministic argmax.

| Metric | Masked random | Reward-v1 PPO | Delta |
| --- | ---: | ---: | ---: |
| Completion rate | 0% | 0% | none |
| Death rate | 81.25% | 43.75% | -37.50 pp |
| Step-limit rate | 18.75% | 56.25% | +37.50 pp |
| Mean turns | 33.50 | 82.75 | +49.25 |
| Mean shaped return | -2.1704 | -1.3422 | +0.8281 |
| Furthest floor | 1-1 | 1-1 | none |
| Enemy kills | 0 | 6 | +6 |
| Item/currency pickups | 0 | 6 | +6 |
| Enemy damage | 1 | 7 | +6 |
| Player damage | 55 | 32 | -23 |
| Mean maximum gold | 0.00 | 0.75 | +0.75 |

## Conclusion

Reward-v1 fixed the strongest symptom of the old baseline: the deterministic
policy now fights, collects currency, and explores while still surviving much
longer than masked random. It is not competent yet. Neither policy progressed
beyond floor 1-1, making navigation to a discovered staircase the clearest next
bottleneck.

The next reward iteration should instrument stair discovery and use bounded,
potential-difference shaping for distance to a known staircase. That supplies
a learnable navigation signal while giving zero net reward for walking loops.
A longer run is justified only after that component is visible in rollout
metrics and verified against reward cycling tests.

## Artifacts

The ignored runtime directory `.runtime/reward-v1-bard-live-8w-20260822/`
contains `final.pt`, `latest.pt`, `metrics.jsonl`, `config.json`, and the full
per-seed `baseline.json` report.
