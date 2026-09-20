# EXP-0036

The qualified short pilot completed 16384 transitions and 16 PPO updates with zero recoveries and finite core metrics. Frozen and final policies both scored 0/32 training and 0/16 development completions, with zero evaluation boss damage. Wall attempts increased from 85.35% to 89.18%. During training, 13 of 55 completed episodes dealt 34 total boss damage, but none completed the task. Training took 498.344 seconds and measured execution 1586.983 seconds. No held-out final-test gameplay or promotion. This single optimizer trial does not establish inability to learn; isolate navigation assistance next, since it changed alongside weapon-state semantics.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs\bounded-learning-exp0036-complete\validated-analysis.json`
- `runs\bounded-learning-exp0036-complete\manifest.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
