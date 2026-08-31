# EXP-0026: live-calibrated one-action Death Metal trace tails

This experiment supersedes the unrun EXP-0025 design without mutating its
registered contract. It converts rare legal EXP-0024 successes into the first
measured reverse-curriculum boundary.

The boundary is exactly one learner action. On the qualified recurrent
observations, the frozen source checkpoint assigns the final successful action
probabilities 8.8%, 50.0%, and 18.4% for seeds 92008, 92096, and 92116. By
contrast, the products of its recorded-action probabilities over the final 16
actions are approximately `3.7e-5`, `6.9e-9`, and `1.15e-2`. Exact trace
probability is stricter than task success because alternative successful actions
may exist, so the launcher first measures actual live completion and trains only
if it lies inside the declared 10--90% competence band.

Frozen final checkpoints must reproduce across at least three game seeds and all
three optimizer trials. Passing authorizes a separately registered longer-tail
experiment. It does not establish normal-start Zone 2.
