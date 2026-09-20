# EXP-0039

A fresh 1024-transition batch reproduced zero observed rewards and nonzero critic-derived advantages. Pre-update recurrent replay max absolute log-probability difference was 9.32e-5. Original PPO made 16 minibatch steps with final whole-batch sampled-action approximate KL 0.606318; target_kl=0.02 stopped after one step but final whole-batch KL was already 0.293809. All sampled BOMB/THROW actions had negative normalized advantages; this is state-confounded, not causal action utility. Offline replay verified both effects on identical saved data within 1e-4 KL. Zeroing stored and bootstrap values analytically made every advantage zero. The live wrapper failed after saving the diagnostic due to checkpoint interval zero; failed status preserved, wrapper corrected, independent offline recovery used no new gameplay and stayed within the total cap. No gameplay evaluation, promotion or proof of the cause of EXP-0038 failure. Reward arithmetic and research support separating objective, shaping, critic calibration, curriculum and optimizer step control.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/first-update-audit-exp0039/audit.json`
- `runs/first-update-audit-exp0039/offline-validation.json`
- `runs/first-update-audit-exp0039/signal-validation.json`
- `runs/first-update-audit-exp0039/reward-arithmetic.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
