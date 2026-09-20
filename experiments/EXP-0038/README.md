# EXP-0038

The navigation-assisted matched PPO pilot completed 16384 transitions and 16 updates with zero recovery and finite metrics. Training took 529.297 seconds; total new execution 1025.500 seconds. Frozen final evaluation regressed from 17 boss damage in 8/48 episodes to zero damage in 0/48, with 0/48 completions before and after. Wall-attempt rate fell from 30.54% to 17.77%, deaths from 40 to 2, and 46/48 final episodes reached the turn limit. Training produced 41 boss damage in 17 of 56 completed episodes but no wins. Both policy streams and both pools lost baseline boss damage. The first update had approximate KL 0.9262 and clip fraction 0.4197; this is a diagnostic clue, not causal proof. No promotion or final-test gameplay. Audit initial reward/advantage signals and policy movement before another full training run.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/bounded-navigation-learning-exp0038/validated-analysis.json`
- `runs/bounded-navigation-learning-exp0038/manifest.json`
- `runs/bounded-navigation-learning-exp0038/optimization-diagnostics.json`
- `runs/bounded-navigation-learning-exp0038/behavior-summary.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
