# EXP-0018 normal-health transfer calibration

This design calibration tests the unchanged EXP-0017-selected policy with
`boss1hp-player6`, which removes player-health assistance while retaining the
one-hit boss objective. It reuses EXP-0017's 24 acceptance seeds only after that
experiment's immutable decision; those seeds are development history for
EXP-0018 and cannot supply its acceptance evidence.

The parent is evaluated with argmax and two timing-independent sampled streams
(`102001` and `102002`), 500 turns per episode. The two sampled streams select
the next reset mixture; argmax remains a deployment diagnostic.

Before reading the results, the mastered-start replay rule is:

- sampled player6 completion at least 40%: 80% player6 / 20% mastered player10;
- at least 15% but below 40%: 60% player6 / 40% mastered player10;
- above 0% but below 15%: 50% player6 / 50% mastered player10;
- zero sampled completions: do not train across this jump; introduce and
  requalify an intermediate player-health profile.

If sampled completion is at least 50%, at least 12 of 24 seeds succeed, sampled
death is at most 50%, deterministic completion is at least 20%, and controller
validity is exact, the frozen parent directly passes player-health removal and
no optimization is run. Otherwise the calibrated mixture is used in a separate
registered experiment with fresh final seeds. Shaped return never participates.
