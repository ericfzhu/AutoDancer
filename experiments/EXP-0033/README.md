# EXP-0033: complete-window Death Metal handoff expansion

EXP-0032 acquired tail 48, with all three final checkpoints retaining tails 32,
36, and 40. Tail 44 was also in the training window and was saturated during
source calibration, but the previous comparison did not name it as a retained
boundary. This successor protects and independently evaluates the complete
tail-32-through-tail-48 window.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\retained-death-metal-trace-window-4" `
  -SourceCheckpointOverride "runs\retained-death-metal-trace-window-3\training\seed-101002\final.pt" `
  -RetainedTailWindow 32,36,40,44,48 `
  -MinimumRetentionEligibleTrials 2 `
  -CandidateTailActions 52,56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -TrainingSeeds 102001,102002,102003 `
  -TotalSteps 92160 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0033" `
  -ExperimentArm "a8-complete-trace-window-4" `
  -TrainingDistributionVersion "qualified-trace-window-v11" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```

This remains assisted training-seed evidence. Passing at tail 81 would authorize
a separate prefix-free full-boss evaluation, not normal-start promotion.
