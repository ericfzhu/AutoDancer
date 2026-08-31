# EXP-0025: qualified live Death Metal trace tails

This experiment converts rare legal EXP-0024 successes into a controlled reverse
curriculum. Complete successful action sequences are found only on disclosed
training seeds and replay-qualified in fresh game processes. PPO begins near the
end of each sequence; the replayed prefix supplies live observations and recurrent
context but contributes no learner transition or reward.

The first boundary is exactly one learner action. This is performance-calibrated,
not a guessed tail length: on the qualified recurrent observations, the frozen
source checkpoint assigns the final successful action probabilities 8.8%, 50.0%,
and 18.4% for seeds 92008, 92096, and 92116. By contrast, the products of its
recorded-action probabilities over the final 16 actions are approximately
`3.7e-5`, `6.9e-9`, and `1.15e-2`. Exact trace probability is not identical to
task success probability because alternative successful actions may exist, but
it is strong evidence that 16 actions was an unjustified first curriculum jump.

Each frozen final checkpoint is tested under deterministic and two predeclared
stochastic modes and must reproduce across at least three game seeds and all three
optimizer trials. Training-curve success cannot pass the gate. Passing authorizes
longer tails, then unassisted full-boss evaluation. It does not establish
normal-start Zone 2.
