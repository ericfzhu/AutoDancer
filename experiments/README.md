# AutoDancer experiment lineage

This directory is the Git-tracked scientific record. `registry.yaml` identifies every
experiment and pins the SHA-256 of its immutable `experiment.yaml`. `baselines.yaml`
records deliberate promotions. Each experiment directory contains its predeclared
question, hypothesis, controlled change, invariants, arms, evaluation, and decision
rule; completed experiments add `decision.json`.

`components.yaml` is the append-only catalog of versioned agent blocks. Every
experiment names the exact component versions it changes and the important versions
it holds fixed.

Runtime metrics, machine/controller provenance, and artifacts are tracked in the local
MLflow store under `.runtime/mlflow/`. A run directory also receives `lineage.json`, so
the evidence remains understandable without MLflow.

See `docs/experiment-lineage.md` for the workflow.
