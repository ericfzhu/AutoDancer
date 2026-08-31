# EXP-0028: adaptive live trace-tail acquisition

EXP-0027 stopped before optimization because its one-action handoff was already
completed in all nine predeclared source-policy episodes. That is a valid
calibration rejection: PPO cannot demonstrate acquisition on a task the source
policy has already saturated.

EXP-0028 preserves the exact source checkpoint, qualified live traces, A8 policy,
reward, action contract, and recurrent-boundary correction. It changes only how
the learner boundary is selected. The launcher tests tail lengths 1 through 16
in ascending order, using deterministic execution and four fixed stochastic
policy streams across all three qualified trace seeds. It selects the shortest
tail with 10--90% live completion and writes the complete attempted boundary
curve before any training begins.

The experiment is a reverse-curriculum acquisition test. Passing authorizes a
later retained window of nearby handoffs. It does not demonstrate normal All
Zones competence and cannot satisfy the Zone 2 promotion gate by itself.

Run after the current-hash controller qualification passes:

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\adaptive-death-metal-trace-tail" `
  -CandidateTailActions (1..16) `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -ExperimentId "EXP-0028" `
  -ExperimentArm "a8-adaptive-trace-tail" `
  -TrainingDistributionVersion "qualified-trace-tail-v6" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```
