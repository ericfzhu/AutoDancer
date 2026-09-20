# AutoDancer experiment lineage

This directory is the Git-tracked scientific record. `registry.yaml` identifies every
experiment and pins the SHA-256 of its immutable `experiment.yaml`. `baselines.yaml`
records deliberate promotions. Each experiment directory contains its predeclared
question, hypothesis, controlled change, invariants, arms, evaluation, and decision
rule; completed experiments add `decision.json`.

`components.yaml` is the append-only catalog of versioned agent blocks. Every
experiment names the exact component versions it changes and the important versions
it holds fixed.

Runtime metrics, machine/controller provenance, and artifacts are tracked in the local
MLflow store under `.runtime/mlflow/`. A run directory also receives `lineage.json`, so
the evidence remains understandable without MLflow.

See `docs/experiment-lineage.md` for the workflow.

Keep one concise results summary in each experiment README. Full logs, metrics,
checkpoints and generated analysis belong in ignored run/artifact storage. See
[repository policy](../docs/README.md) and [config status](../configs/README.md).
Historical decisions are append-only: retired documentation paths resolve through
[report-locations.json](report-locations.json), which records original report hashes
and their local archives. Summaries do not replace original evidence bytes.

## Bounded learning experiment series

This ledger distinguishes completed measurements from conditional work that was
not authorized by its preparation gate. A completed experiment need not establish
an improvement, and an inconclusive result does not promote a policy.

| Experiment | Question | Evidence and disposition |
| --- | --- | --- |
| EXP-0035 | Fixed versus varied direct encounters | Preparation failed the predeclared competence gate. The specification requires stopping the six conditional training runs; frozen direct-versus-handoff diagnosis completed. |
| EXP-0036 | Corrected controls, short direct pilot | Training and paired evaluation completed; no boss contact in final evaluation, navigation dominated failures. |
| EXP-0037 | Restore navigation assistance | Frozen paired comparison completed; contact increased to 8/48. |
| EXP-0038 | Learn with navigation assistance | Training and final evaluation completed; larger PPO steps lost all final contact. |
| EXP-0039 | Audit the first update and reward signals | Live batch capture completed; post-capture checkpoint bookkeeping failed. Saved-batch offline reproduction validated the measured update. Failure is preserved, not relabeled as successful full training. |
| EXP-0040 | Bound update size and inspect critic fitting | Three saved-batch step sizes and separate critic-head diagnostic completed; 3e-6 passed the local movement screen. |
| EXP-0041 | Live stability and retention | 16,384 transitions and 48 final episodes completed; contact 8 to 20, damage 17 to 41, zero wins. |
| EXP-0042 | Current-control demonstrations | All three old traces replayed; two valid, third rejected at repeated THROW. No invalid trace is used for learning. |
| EXP-0044 | Historical missing-boundary supplement | Recorded commit restored in isolation, but mod identity failed preflight. No dependent gameplay; historical gap remains unproven. |
| EXP-0043 | Curriculum learning and direct transfer | All stages completed. Assisted success stayed 8/8; direct wins 0/48, contact 6/48 and damage13 versus control20/48 and41. Training cost3.13x. Inconclusive, no promotion. |

Each experiment's immutable specification and decision are under its numbered
experiments directory. Concise results are in each experiment README; detailed evidence remains in
hash-bound run artifacts. Original standalone reports are indexed in `report-locations.json`. The current sequence includes EXP-0043's final
conditional and direct evaluations, validation, written decision and registry
audit, plus the EXP-0044 historical identity preflight. It does not imply open-ended training until a win appears.

### Historical registry cleanup

Two older completed pipelines still had planned registry entries. The retrospective
audit in runs/experiment-completion-audit-20260918/historical-artifact-audit.json
checked all three training trials and underlying final reports for each.

EXP-0024's original comparator was recomputed exactly: zero successes in 216
trained held-out episodes. Its registry decision now records rejection.

EXP-0034 had 189/189 successful final episodes at its seven listed evaluation
boundaries. It also trained on tail56, which has no final evaluation. Its old
comparison is preserved, but the broader complete-window retention claim remains
unproven; the retrospective registry decision is inconclusive. The current code
has different observation/control identity, so a new run cannot be passed off as
missing historical evidence. This limitation is not a new performance finding.

EXP-0044 tested whether the recorded historical commit could reproduce the old runtime. It could not reproduce the recorded AutoDancer.lua hash even after LF normalization. The worktree is retained for inspection; no qualification report was rewritten and no dependent game run was launched. This failed preparation closes the bounded supplement without claiming the historical gap is fixed.

### Final audit (2026-09-18)

All 44 registry entries now have terminal decisions. EXP-0043 completed every required live stage and passed its artifact validator; no game workers remain. Conditional stops are explicitly distinguished from executed runs. The full requirement/evidence ledger is runs/experiment-completion-audit-20260918/current-series-completion.json. Current-series final conclusions are recorded without claiming a solved full fight or resolved historical tail56 gap.
