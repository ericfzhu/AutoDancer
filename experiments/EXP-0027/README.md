# EXP-0027: corrected recurrent trace-tail acquisition

This experiment supersedes unrun EXP-0026 after a pre-training audit found that
PPO replay restored only the first LSTM state in each 32-step chunk. Warm
one-action episodes inside that chunk were therefore recomputed from the previous
episode's hidden state even though collection used the exact replay-warmed state.

EXP-0027 keeps the one-action live calibration, source checkpoint, qualified
training traces, seeds, budgets, and gameplay gates unchanged. Its sole learning
change is to mark every handoff as an episode boundary and restore the stored
recurrent state at each such boundary. Passing authorizes a separately registered
handoff-window experiment; it does not establish normal-start Zone 2.
