# EXP-0009: Stochastic policy execution calibration

PPO trains AutoDancer by sampling a categorical policy with an entropy bonus, but
all promotion evaluations have converted that policy to per-state argmax. The
trained policies still have high entropy, stochastic training has reached Floor
3, and deterministic evaluation has reached only Floor 2. This experiment tests
that execution mismatch without changing any checkpoint or learning component.

Each checkpoint plays the same 24 unseen game seeds once under argmax and twice
under reproducible stochastic sampling. Passing requires Zone 2 on multiple game
seeds and agreement across both policy-sample streams; a single lucky trajectory
does not pass.

Runtime evidence is written to `runs/stochastic-policy-calibration` and MLflow.
