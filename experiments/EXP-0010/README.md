# EXP-0010: corrected learning-integrity A2 replication

This experiment isolates the environment and return-target corrections made
after EXP-0009. It does not propose a new reward or architecture.

The promoted Reward-V2/A2 checkpoint supplies the policy, representation, and
recurrent weights. The critic and optimizer are reset because historical reward
targets contained client-horizon terminal errors and impossible lethal-overkill
damage. A full 250,880-transition run is used so the result is not interpreted
from a short pilot.

Passing requires repeatable Zone 2 progress across unseen game seeds and both
reproducible stochastic policy streams. If corrected normal-start learning does
not pass, the next intervention is a separately declared floor curriculum.
