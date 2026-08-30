# EXP-0023: legal phase-3 Death Metal successor

EXP-0022 proved that its seed-86002 policy can legally reach phase 3 and once
clear the untouched boss, but full-fight completion remained too sparse for PPO.
This experiment freezes that policy as an ordinary-action prefix guide and lets
the learner take over the exact running game only after the live state proves
phase 3 has begun.

The experiment tests one causal question: whether dense access to the final
Death Metal segment can produce repeatable held-out Zone 2 entries. It does not
claim normal-start competence. A pass advances the reverse curriculum backward;
a failure tells us whether the bottleneck is guide acquisition or learning the
final segment, without conflating either result with reward return.
