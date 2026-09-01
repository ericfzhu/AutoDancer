# EXP-0030: retained live Death Metal handoff window

EXP-0029 accepted tail 32 as a reproducibly learnable live boss boundary. This
successor follows Backplay's window principle: retain that mastered state while
adding every four-action handoff through the next measured competence frontier.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\retained-death-metal-trace-window" `
  -SourceCheckpointOverride "runs\expanded-death-metal-trace-tail\training\seed-97001\final.pt" `
  -RetainedTailActions 32 `
  -CandidateTailActions 36,40,44,48,52,56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -TrainingSeeds 99001,99002,99003 `
  -TotalSteps 92160 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0030" `
  -ExperimentArm "a8-retained-trace-window" `
  -TrainingDistributionVersion "qualified-trace-window-v8" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```

The experiment is conditional curriculum evidence only. It cannot satisfy the
normal-start Zone 2 objective.
