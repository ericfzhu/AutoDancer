# EXP-0041

Completed 16384 transitions and 48 matched final evaluation episodes in 970.907 seconds. Boss-contact episodes increased from 8 to 20 and total damage from 17 to 41; development damage increased from 2 to 11. Wins remained zero and deaths remained 40/48. Retention screen passed, but completion-based replication trigger did not. All 16 updates had full-batch sampled-action approximate KL below 0.02 (maximum 0.015205), all 256 optimizer steps executed, no KL stops or recovery. Single optimizer seed and reused development maps limit inference. Keep small-step settings as experimental control; investigate current-contract curriculum qualification before more training. No production reward changes or policy promotion.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/stable-navigation-learning-exp0041/validated-analysis.json`
- `runs/stable-navigation-learning-exp0041/saved-signal-analysis.json`
- `runs/stable-navigation-learning-exp0041/manifest.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
