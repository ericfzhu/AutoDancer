# EXP-0017 assistance-reduction calibration

This calibration is experiment design, not EXP-0017 evaluation. It freezes the
selected EXP-0016 seed-68002 policy and measures `boss1hp-player10` on the 12
Death Metal seeds already consumed by EXP-0016. Those seeds are development
history from this point onward and cannot provide EXP-0017 acceptance evidence.

The parent is evaluated once with argmax and with two timing-independent sampled
streams (`99001` and `99002`), 500 turns per episode. The two sampled streams are
the calibration statistic because EXP-0016 demonstrated a material argmax versus
categorical-policy gap. Argmax remains a separately reported deployment
diagnostic.

Before reading the results, the mastered-start replay rule is:

- sampled player10 completion at least 40%: 80% player10 / 20% player20;
- at least 15% but below 40%: 60% player10 / 40% player20;
- above 0% but below 15%: 50% player10 / 50% player20 and do not reduce
  assistance again without a new calibrated intermediate profile;
- zero sampled completions: do not run EXP-0017; add and requalify an intermediate
  player-health profile instead.

The ratio is fixed for every worker and training trial through the deterministic
`mixed-curriculum-replay-v1` sampler. Infrastructure recovery repeats the same
seed and start specification and cannot change these counts. Final selection
will use fresh, telemetry-selected Death Metal seeds, gameplay completion and
death—not shaped return or these calibration seeds.
