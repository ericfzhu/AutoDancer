# EXP-0043

All stages completed and validated in 2848.610 seconds. Fresh current-contract two-trace qualification passed. Frozen and final conditional evaluation both won8/8; mean learner turns15 to13.25. Training completed16384 transitions,16 updates,256 optimizer steps and1209/1209 successful endings in1751.562 seconds. Maximum full-batch sampled-action approximate KL0.001680; no recovery or early stops. Final direct starts had0/48 wins,6 contact episodes and13 damage, versus source8 contacts/17 damage and direct-trained EXP-0041 control20 contacts/41 damage. Curriculum training took3.13 times as long; prefix acquisition accounted for64.47 percent of summed worker fragment time. This already-mastered two-map curriculum did not show useful full-fight transfer. Single optimizer seed, success-selected demonstration maps and changed training seed distribution limit inference. Journal boss damage can omit finishing hits; conclusions use authoritative completions and frozen evaluation. No reward/default changes or promotion.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/bounded-curriculum-transfer-exp0043/validated-analysis.json`
- `runs/bounded-curriculum-transfer-exp0043/signal-comparison.json`
- `runs/bounded-curriculum-transfer-exp0043/manifest.json`
- `runs/bounded-curriculum-transfer-exp0043/qualification.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
