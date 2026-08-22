# Reward-v2 live validation

Validated on 2026-08-22 with eight Bard workers, schema 5, and the versioned
asynchronous PPO collector.

## Equal-budget precursor run

An 8,192-transition run using staircase shaping but before the final revisit
penalty completed eight finite PPO updates with zero worker restarts. On held-out
seeds 20001–20016, its deterministic policy was compared with masked random:

| Metric | Masked random | Trained policy |
| --- | ---: | ---: |
| Death rate | 87.50% | 18.75% |
| Step-limit rate | 12.50% | 81.25% |
| Mean turns | 37.69 | 109.38 |
| Player damage | 56 | 12 |
| Item pickups | 0 | 1 |
| Furthest floor | 1-1 | 1-1 |
| Enemy kills | 0 | 0 |

This was a strong survival improvement but still exhibited safe movement loops
and never activated staircase shaping. That evidence motivated the final
`revisit = -0.01` component.

## Final profile smoke

The final reward-v2 profile completed two live PPO updates over 2,048
transitions with finite losses and zero restarts. Collector throughput was
83.70 and 90.89 transitions/second. Revisit penalties were present in both
rollouts (`-7.02` and `-7.80`), proving repeated-coordinate behavior is no
longer neutral. Combat, novelty, damage, terminal, and container components
continued to appear normally.

The smoke did not discover stairs, so it does not establish improved floor
completion. Stair discovery and signed distance shaping are covered by
deterministic cycling and cross-floor tests. The next competence experiment
should train the final profile for substantially longer than 8,192 transitions
and compare staircase discoveries and furthest-floor distributions, not only
mean return.

## Artifacts

Ignored runtime artifacts are under `.runtime/reward-v2-bard-live-8w-20260822/`
and `.runtime/reward-v2-revisit-live-smoke/`.
