# EXP-0034: tail-52-retained Death Metal handoff expansion

EXP-0033 produced three independently trained checkpoints that each cleared all
54 frozen episodes spanning tails 32, 36, 40, 44, 48, and 52. This successor
retains and independently evaluates that complete window while calibrating the
next harder boundary.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\retained-death-metal-trace-window-5" `
  -SourceCheckpointOverride "runs\retained-death-metal-trace-window-4\training\seed-102001\final.pt" `
  -RetainedTailWindow 32,36,40,44,48,52 `
  -MinimumRetentionEligibleTrials 2 `
  -CandidateTailActions 56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -TrainingSeeds 103001,103002,103003 `
  -TotalSteps 92160 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0034" `
  -ExperimentArm "a8-complete-trace-window-5" `
  -TrainingDistributionVersion "qualified-trace-window-v13" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```

This remains assisted training-seed evidence. Passing at tail 81 authorizes a
separate prefix-free full-boss evaluation, not normal-start promotion.
