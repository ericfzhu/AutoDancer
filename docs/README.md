# Documentation and repository policy

Start here instead of treating every historical report as the current design.

## Maintained references

- [Harness design](harness-design.md): proposed contracts and implementation gaps;
  it explicitly separates implemented, offline-tested and live-qualified behavior.
- [Trial-and-error tasks](trial-and-error-contracts.md): reusable task/reward
  profiles and their limits. See the [config index](../configs/README.md).
- [Protocol](protocol.md), [rewards](rewards.md) and
  [equipment catalog](game-control-catalog.md): interface and source evidence.
  Catalog availability is not proof of ordinary Bard loot eligibility.
- [Experiment workflow](experiment-lineage.md) and
  [experiment ledger](../experiments/README.md): declarations, decisions and results.

## Historical references

Architecture comparisons, reward history, performance reports, environment audits,
mechanics probes and baselines describe their recorded versions and workloads.
They are evidence, not automatically current defaults. In particular,
`architecture.md`, `reward-history.md`, `rl-environment-audit.md` and
`environment-design-audit.md` span older implementations. Check source contracts
and the current harness design before using their settings.

The older reports are retained where they contain distinct technical evidence.
New experimental results belong in the experiment's existing README and decision,
not another top-level document. Extend a maintained reference only when the
implementation or lasting design decision changes.

## What belongs in Git

| Content | Location / treatment |
| --- | --- |
| Production code, tests, reproducible runners and analyzers | Track in existing source/test/tool directories |
| Reusable task/reward configs | Track in `configs/`; document status and preserve versioned history |
| Experiment-specific settings | Prefer the immutable `experiments/EXP-NNNN/experiment.yaml`; avoid a duplicate standalone config |
| Hypothesis, scope, evaluation and decision | Track specification, concise README and `decision.json` together |
| Registry and baseline promotions | Track in `experiments/` |
| Logs, checkpoints, trajectories, generated metrics, plots and long diagnostic dumps | Keep in ignored `runs/`, `artifacts/` or `.runtime/` |
| Maintained explanations | Update the references above rather than add a progress document |
| Scratch notes and temporary sweeps | Keep under ignored `runs/` |

Do not remove a versioned config or reproducible analysis script merely because
its experiment finished. Check references and frozen manifests first. Likewise,
do not ignore all JSON/YAML files: some define the scientific contract.

Ignoring files is not a backup strategy. Shareable experiments require a durable
artifact copy plus location and hashes in their record. Current `runs/` locations
are local paths; this cleanup does not claim they have been uploaded elsewhere.

## September 2026 consolidation

Nine overlapping standalone reports were consolidated into the experiment ledger
and concise experiment READMEs. Byte-exact originals are preserved under ignored
`runs/documentation-archive-20260920/`. The tracked
[report relocation index](../experiments/report-locations.json) records original
paths, SHA-256 hashes, current summaries and local archive paths.

Frozen specifications and decisions were not rewritten. When an old decision
names a retired documentation path, resolve it through that index. A summary is
not a byte-identical replacement for the original evidence. Raw run data and
checkpoints were not moved or deleted.
