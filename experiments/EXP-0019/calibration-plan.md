# EXP-0019 frozen player8 calibration

After the fresh player8 controller qualification passes, evaluate the unchanged
EXP-0017 seed-68002 checkpoint on the ascending first 24 telemetry-confirmed
Death Metal seeds in `75001-75256`. Use deterministic argmax and reproducible
stochastic policy seeds `106001` and `106002`, eight workers, 500 turns, Reward
V4A, A2, and `map-navigation-prior-v1`.

Frozen transfer passes only with at least 60% sampled completion, at most 40%
sampled death, at least 16 distinct sampled successful seeds, at least 33%
deterministic completion, and exact controller validity with zero restarts. If it
passes, do not optimize. If it fails, select the 24-seed training pool from
`75301-75556` before the first training trial and run the immutable 80/20
player8/player10 mixture. Calibration seeds are never reused for training or
final evaluation.
