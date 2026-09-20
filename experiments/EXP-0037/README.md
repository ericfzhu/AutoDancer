# EXP-0037

Frozen navigation assistance reduced wall-attempt rate from 85.35% to 30.54% and produced 17 boss damage in 8/48 episodes, compared with zero in reused paired controls. Both arms completed 0/48; deaths increased from 3 to 40. Training-seed damage was 15 in 7/32 episodes; development damage was 2 in 1/16. All 48 new episodes were valid with zero recoveries and no scope violations. New evaluation took 385.078 seconds with no optimizer or final-test gameplay. The result supports navigation dependence, not learned competence or policy promotion. Next isolate learning with navigation retained, holding architecture and reward fixed.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/bounded-navigation-exp0037/validated-analysis.json`
- `runs/bounded-navigation-exp0037/manifest.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
