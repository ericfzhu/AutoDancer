# EXP-0029: expanded live competence boundary

EXP-0028 found that all final-1 through final-16 live handoffs were saturated at
15/15 source-policy completions. EXP-0029 moves the boundary substantially
earlier, testing tails 20 through 80 in four-action increments and then the
maximum legal tail 81. It uses a 128-turn cap so longer valid continuations are
not converted into artificial timeouts.

Worker 0 initializes Steam presence while the remaining seven workers retain the
lightweight profile. Gameplay parity for this split was established by exact
three-trace replay and the eight-worker benchmark before registration.

```powershell
.\tools\run-qualified-trace-tail-pilot.ps1 `
  -RunDir "runs\expanded-death-metal-trace-tail" `
  -CandidateTailActions 20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,81 `
  -CalibrationPolicySeeds 0,98001,98002,98003,98004 `
  -LearnerTurnCap 128 `
  -SteamPresenceWorker 0 `
  -ExperimentId "EXP-0029" `
  -ExperimentArm "a8-expanded-trace-tail" `
  -TrainingDistributionVersion "qualified-trace-tail-v7" `
  -ControllerQualification "runs\controller-qualification-steam-parity\qualification.json"
```
