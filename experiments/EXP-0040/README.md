# EXP-0040

All offline diagnostics completed in 26.609 seconds with zero new gameplay. With target_kl=0.02, joint rates 3e-4, 3e-5 and 3e-6 produced final saved-batch sampled-action approximate KL 0.293809, 0.041222 and 0.005414 after 1, 4 and 16 optimizer steps. Only 3e-6 passed the predeclared average-change screen; it is a live-follow-up candidate, not an optimal or proven gameplay setting. A separate 32-step critic-head fit reduced held-out-worker fixed-target MSE from 4.5676 to 0.7011 and the one observed terminal-prefix MSE from 12.7108 to 4.0225, while actor parameters and sampled-action probabilities stayed unchanged. Fixed targets are bootstrapped, and the terminal subset is one correlated failed trajectory. No calibrated-critic PPO arm was run because the saved batch lacks final bootstrap observations. No reward or production default changes and no policy promotion.

Specification: [experiment.yaml](experiment.yaml). Decision: [decision.json](decision.json).

Evidence (repository-relative artifact paths, not stored in Git):

- `runs/saved-batch-stability-exp0040/validated-summary.json`
- `runs/saved-batch-stability-exp0040/results.json`
- `runs/saved-batch-stability-exp0040/manifest.json`

The original detailed report is archived locally; its former path and SHA-256
are recorded in [report-locations.json](../report-locations.json).
