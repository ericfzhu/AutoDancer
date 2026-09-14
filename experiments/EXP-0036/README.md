# EXP-0036: bounded direct-start learning pilot

The immutable protocol is in `experiment.yaml`. This is one short exploratory
PPO fine-tune under the qualified bounded Bard contract, with paired frozen and
final evaluations. It does not reopen EXP-0035 or use its final-test seeds.

The corrected execution directory is
`runs/bounded-learning-exp0036-complete`. The runner and analysis tools are
`tools/run_bounded_learning_pilot.py` and
`tools/analyze_bounded_learning_pilot.py`.

Two orchestration issues were corrected before training:

1. `runs/bounded-learning-exp0036` stopped before gameplay because the tracked
   command omitted its required reward lineage label.
2. `runs/bounded-learning-exp0036-final` completed frozen stream 215001, then
   rejected stream 215002 before gameplay because different evaluation trials
   shared one lineage directory. Evaluations now have separate directories.

The completed first reference report was reused with unchanged experiment and
model/controller/reward source hashes, rather than replayed or selected by its
outcome. Its original invocation and report hash are retained. The corrected run
keeps that attempt's original budget start time, so the completed evaluation and
repair interval count against the 45-minute execution cap. Neither correction
changes gameplay settings, seed allocation, training budget or decision rule.

All historical attempts remain preserved. Only a fully validated final analysis
can support the experiment decision; no checkpoint is promoted by this pilot.

Result: the complete audit passed, but frozen and final policies both scored
0/32 on training and 0/16 on development, with zero boss damage in evaluation.
Training completed 16,384 transitions in 498.344 seconds. See
`docs/bounded-learning-pilot.md` and `validated-analysis.json` in the corrected
execution directory for the evidence and limitations. Decision: inconclusive.
