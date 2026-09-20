# Configuration index

These small JSON files define behavior, so they belong in Git. Versioned files
are retained for checkpoint metadata, tests and historical experiment replay.
Their presence does not mean they are recommended training defaults.

| Files | Status / purpose |
| --- | --- |
| `task-first-floor-v1.json`, `task-zone-one-v1.json`, `task-boss-floor-v1.json` | Current proposed fixed trial-and-error tasks; no teacher prefix |
| `reward-trial-sparse-v5.json`, `reward-trial-exploration-v5.json` | Paired trial-and-error reward definitions; smoke-tested, not proven learning improvements |
| `reward-death-metal-potential-v5.json` | Existing bounded boss-progress potential profile and historical experiment reference |
| `reward-death-metal-guide-v1.json` through `v3.json` | Historical guide reward profiles; not part of the current no-demonstrations setup |
| `reward-v2.json`, `reward-v4a.json`, `reward-v4b.json` | Historical reward baselines and comparison arms |
| `curriculum-exp0017-player10-replay.json`, `curriculum-exp0018-player6-replay.json`, `curriculum-exp0019-player8-replay.json` | Historical experiment-specific curriculum files; retained at original paths |
| `adaptive-curriculum-zone2-v1.json` | Historical adaptive curriculum configuration |

Usage and limitations: [trial-and-error contracts](../docs/trial-and-error-contracts.md).
New one-off experiment settings should live in their experiment specification
unless the runtime needs a separate file. Do not silently edit old versions to
match new code or duplicate the same settings in another report.
