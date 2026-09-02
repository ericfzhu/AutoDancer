# EXP-0032: reproducible retained Death Metal handoff-window expansion

EXP-0031 acquired tail 40, but one optimizer trial forgot tail 32. This successor
retains tails 32, 36, and 40 and requires at least two complete checkpoints to
preserve every boundary before advancing.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\retained-death-metal-trace-window-3" `
  -SourceCheckpointOverride "runs\retained-death-metal-trace-window-2\training\seed-100002\final.pt" `
  -RetainedTailWindow 32,36,40 `
  -MinimumRetentionEligibleTrials 2 `
  -CandidateTailActions 44,48,52,56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -TrainingSeeds 101001,101002,101003 `
  -TotalSteps 92160 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0032" `
  -ExperimentArm "a8-retained-trace-window-3" `
  -TrainingDistributionVersion "qualified-trace-window-v10" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```

This remains assisted training-seed evidence. Prefix-free unseen-seed play is
required before any normal-start Zone 2 promotion claim.
