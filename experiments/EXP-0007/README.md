# EXP-0007: Qualified Architecture 8 replication

This experiment resolves the inconclusive A8 result without changing rewards. It
compares the frozen A2/V2 baseline, ordinary A2 fine-tuning, and the A8 staged
residual under the now-qualified schema-10 controller.

The immutable protocol and thresholds are in `experiment.yaml`. Runtime evidence is
written to `runs/architecture8-qualified-replication` and MLflow. The recoverable
launcher is `tools/run-architecture8-qualified-replication.ps1`.

Passing this pilot means “advance to multiseed confirmation,” not “promote A8.”
