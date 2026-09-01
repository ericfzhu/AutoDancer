# EXP-0031: second retained live Death Metal handoff-window expansion

EXP-0030 acquired tail 36 without losing tail 32. This successor retains both
boundaries and searches from tail 40 toward the maximum legal tail 81.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\retained-death-metal-trace-window-2" `
  -SourceCheckpointOverride "runs\retained-death-metal-trace-window\training\seed-99001\final.pt" `
  -RetainedTailWindow 32,36 `
  -CandidateTailActions 40,44,48,52,56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -TrainingSeeds 100001,100002,100003 `
  -TotalSteps 92160 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0031" `
  -ExperimentArm "a8-retained-trace-window-2" `
  -TrainingDistributionVersion "qualified-trace-window-v9" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```

The result remains conditional training-seed evidence. Only a later prefix-free,
unseen-seed evaluation can advance toward normal-start Zone 2 promotion.
