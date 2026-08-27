# EXP-0015 rationale

EXP-0014 failed because its mixed, full-strength four-boss distribution produced no
successful training or evaluation trajectories. That is not evidence that A2 or A8
cannot learn bosses; it means the proposed reverse curriculum did not contain the
intermediate-success starts required by its own hypothesis.

Postmortem work fixed visibility, full-floor memory retention, boss identity, boss
objective telemetry, and episode-horizon lineage. A frozen A2 checkpoint using the
map-navigation prior then completed an assisted Death Metal start on seed 64018 in
128 turns, entering Zone 2. The same fixed setup completed 1 of 9 Death Metal seeds,
which supplies a calibrated 11.1% starting success rate without making the task
trivial.

EXP-0015 asks only whether PPO can amplify that signal into repeatable held-out
assisted-boss completion. It does not compare architectures and cannot promote a
normal-start policy. Passing advances to less assistance; failing rejects this start
distribution without inventing a favorable reward or success rule after seeing the
results.
