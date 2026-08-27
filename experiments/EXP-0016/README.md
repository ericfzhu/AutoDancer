# EXP-0016 rationale

This is the executable correction of EXP-0015. The earlier immutable contract was
rejected before training because its training-level distribution label disagreed
with the lineage guard: a finite seed pool combined with a later real-game GOTO start
is a reverse curriculum, not an ordinary finite-pool normal-start distribution.

All scientific details are otherwise unchanged. EXP-0016 tests whether PPO can
amplify a calibrated 1-of-9 assisted Death Metal completion signal into repeatable
held-out assisted-boss completion across three independent training seeds. Passing
advances to less assistance; it does not count as normal-start Zone 2 progress.
