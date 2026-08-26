# Architecture 8 controlled-learning experiment

## Question

Architecture 7 preserved the complete Architecture 2 policy at initialization,
but its single scalar gate starved the new observation branch: none of the four
new input groups acquired material output influence. Architecture 8 tests a
narrower claim: **can schema-9 information learn through a behavior-preserving
residual path when the A2 base is held fixed long enough to prevent drift?**

This is not a reward experiment. Every arm uses Reward V2, the same live Bard
task, the same training seed, and the same PPO settings.

## Arms

| Arm | Initial model | Action contract | Trainable path |
|---|---|---|---|
| A2 legacy | exact V2 A2 checkpoint | WAIT forced illegal | complete A2 |
| A2 fixed | exact V2 A2 checkpoint | current 11-action mask | complete A2 |
| A8 candidate | exact V2 A2 checkpoint | current 11-action mask | schema-9 adapter first; complete model after update 10 |

The legacy control reconstructs the policy-side action contract under which the
V2 checkpoint was trained. The fixed control isolates the effect of making the
live WAIT action available. A8 contains the complete A2 actor, critic, and LSTM
under `base`, plus the A7 sensory encoder and a 512-to-512 residual projection.
The projection is initialized to zero, so all logits, values, recurrent states,
and actions match A2 exactly. Unlike A7's scalar gate, the projection receives a
full matrix gradient on the first update. After it opens, gradients reach the
sensory encoder.

## Predeclared protocol

- Source checkpoint: `runs/reward-v2-250k/final.pt`.
- Reward: `configs/reward-v2.json`, unchanged.
- Training seed: `36001` for all arms.
- Budget: 30,720 transitions per arm with checkpoints at 0, 10,240, 20,480,
  and 30,720.
- A8 freezes the complete A2 base for the first ten PPO updates (10,240 live
  transitions), then unfreezes it.
- Learning-curve evaluation: deterministic seeds `45001–45016`, 1,500-turn
  cap, at every checkpoint.
- Representation gates: at A8 warmup and final, all four new groups must reach
  both at least 1% of median established-input sensitivity and nonzero gradient
  reach.
- The pipeline is sequential and recoverable. Completed checkpoints and reports
  are reused after interruption.

The curve gate fails if training is non-finite, any worker restarts, either
representation gate fails, A8 retains less than 90% of fixed-A2 progress, death
or step-limit rate exceeds fixed A2 by more than ten percentage points, or A8
retains less than 60% of fixed-A2 kills or items per episode.

## Broad gameplay integration

Broad gameplay is item 6, not an automatic continuation of a weak learning
curve. It runs only after every curve gate passes. The three final policies then
play deterministic seeds `46001–46030` with a 3,000-turn cap.

A8 passes this integration gate only if it:

- beats both controls on mean floor progress;
- remains within five death-rate percentage points of fixed A2;
- retains at least 80% of fixed-A2 kills and item pickups per episode;
- does not worsen step-limit or unchanged-position rates;
- completes with no worker restart; and
- still has all four new observation groups materially active.

Passing means **ready for multi-seed confirmation**, not promotion. Failure
retains A2 and identifies whether the cause was action-contract drift,
representation learning, local-gameplay harm, or broad integration.

## Running

From PowerShell at the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-architecture8-controls.ps1
```

The symbolic dashboard is served at <http://127.0.0.1:8765>. Results are written
under `runs/architecture8-controls/`; `pipeline-complete.json` records the
terminal gated decision.

## Result

The learning-curve stage completed. A8 made all four new input groups material
at warmup and final, reached final mean progress `1.125` versus fixed A2's
`1.0625`, lowered death rate from `0.375` to `0.3125`, and retained more than
the required combat and item competence. A8 itself had no worker restart.

The overall gate stopped before broad gameplay because both A2 controls each
recorded one training worker restart. This is an invalid-control health result,
not a broad-gameplay rejection of A8. Under the predeclared rules A2 remains the
baseline until the controls are repeated cleanly and the broad gate is actually
run.

The isolated retry preserves the original artifacts and reuses the completed
A8 checkpoints and reports:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-architecture8-control-retry.ps1
```

Retry controls live under `training/control-retry-1/` and their reports under
matching evaluation subdirectories. If either control restarts, the retry stops
before evaluation. If both remain healthy, the corrected comparator recomputes
the curve gate and automatically runs broad gameplay when admitted.

## Qualified replication and horizon follow-up

After the schema-10 controller qualification, EXP-0007 repeated the comparison
cleanly with frozen A2, equally fine-tuned A2, and A8. All training and
evaluation health checks passed and the broad evaluation ran on 30 fresh seeds.

At 30,720 transitions A8 retained frozen-A2 competence (43 versus 44 kills and
38 versus 37 item pickups), reduced death rate from `0.30` to `0.20`, and reduced
unchanged-position rate from `0.913` to `0.587`. However, all three policies had
mean progress `1.00`, remained on floor 1, and recorded zero staircase
discoveries. A8 also missed the predeclared death bound relative to the highly
passive fine-tuned A2 control. EXP-0007 therefore retained A2 as an
**early-budget rejection**; it did not establish that A8 cannot improve with
more adaptation.

EXP-0008 tests that remaining horizon hypothesis without changing the reward,
architecture, observation, controller, PPO configuration, action contract, or
training seed. It exactly resumes both 30,720-transition trainable checkpoints
and continues them to 250,880 transitions, with common held-out evaluation at
30,720, 61,440, 122,880, and 250,880. A8 advances only if its progression
advantage repeats at the last two checkpoints and satisfies the declared safety
and competence bounds. A final-only advantage is inconclusive and must be
replicated rather than promoted.
