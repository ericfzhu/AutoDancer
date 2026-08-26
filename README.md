# AutoDancer

AutoDancer trains a recurrent PPO agent directly against independent
**Crypt of the NecroDancer** Bard games. There is no simulator, screenshot
capture, keyboard automation, or UI control.

Exactly the requested fixed number of workers are launched by the Python
supervisor. Each worker has its own identity, duplex named pipe, engine log,
seed, run ID, and LSTM state. Python refuses to start around an
unowned NecroDancer process and never silently trains with fewer workers than
requested.

## Install

Install the project and training dependencies:

```powershell
uv sync --extra train --extra test
```

Build `native/autodancer_native.dll`, copy it beside `Necrodancer.exe`, then copy
[`mods/AutoDancer`](mods/AutoDancer) to the unpackaged SYNCHRONY mod directory
and enable it under **Customize → Mods**. The supported initial game build is
`v4.2.1-b5713`. Lua changes require a mod reload or a game restart.

## Train

Close every existing NecroDancer process first. The supervisor launches exactly
`--num-instances` hidden, symbolic-only native workers:

```powershell
uv run autodancer-train `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --mod-dir "$env:LOCALAPPDATA\NecroDancer\mods\AutoDancer" `
  --num-instances 4 `
  --total-steps 1000000 `
  --run-dir ".\runs\bard-ppo" `
  --device auto
```

Add `--dashboard` to serve the live symbolic worker view at
`http://127.0.0.1:8765/`, or use `--dashboard PORT` to choose another port:

```powershell
uv run autodancer-train `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --num-instances 8 `
  --total-steps 1000000 `
  --run-dir ".\runs\bard-ppo" `
  --dashboard
```

The page renders the exact 21 × 21 symbolic grid seen by each policy, plus
health, inventory, action, reward, events, seed, floor, latency, worker health,
and PPO progress. It does not capture or transmit game frames.

Rollouts contain 128 valid transitions per worker. Workers advance independently
through a centrally micro-batched frozen policy, then PPO updates after one
policy-versioned fragment arrives from every slot. PPO uses 32-step recurrent
chunks, four update epochs, action masking, periodic deterministic evaluation,
atomic checkpoints, and JSONL metrics. Continue optimizer/model/RNG state with
`--resume .\runs\bard-ppo\latest.pt`.

Benchmark direct live-worker throughput without PPO updates:

```powershell
uv run autodancer-benchmark `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --sweep "1,2,4,6,8,10,12" `
  --steps 128 `
  --run-dir ".\runs\worker-benchmark"
```

The benchmark reports throughput, latency percentiles, process memory/CPU use,
restarts, and the recommended tested worker count.

Run the fixed-seed live mechanic diagnostic before changing rewards or policy
architecture:

```powershell
uv run autodancer-diagnose `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --seeds "46001,46002,46003" `
  --max-steps 1000 `
  --output ".\runs\mechanic-diagnostics\report.json"
```

The diagnostic verifies the logical-to-engine mapping and acknowledgement path,
checks the legal mask against the actual inventory, and classifies real-engine
outcomes as movement, waiting, combat, interaction, digging, wall attempts,
unchanged directions, and floor transitions. It then searches the fixed seeded
runs for reproducible evidence of combat, items, stairs, and returning to a
visited position. Missing scenarios are reported as unobserved rather than
silently treated as passes.

Measure whether policy outputs and learning gradients materially use each
observation group:

```powershell
uv run autodancer-representation `
  ".\runs\reward-v2-250k\final.pt" `
  ".\runs\architecture7-v2-pilot\training\seed-35001\final.pt" `
  --output ".\runs\architecture7-v2-pilot\representation.json"
```

The report separates unsupported, inactive, merely trace, and material input
paths. See `docs/representation-diagnostics.md` for the controlled
counterfactual, gradient test, and the measured A7 results.

Run the recoverable Architecture 8 controlled-learning experiment with:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-architecture8-controls.ps1
```

The launcher compares unchanged A2 fine-tuning under legacy and current action
contracts with the exact-parity A8 candidate. It evaluates fixed learning-curve
points first and runs broad held-out gameplay only if A8 passes its predeclared
representation, health, and local-gameplay gates. See
[`docs/architecture8-controls.md`](docs/architecture8-controls.md).

If the initial A8 experiment stops only because its A2 controls restarted,
repeat those controls without overwriting the original evidence:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-architecture8-control-retry.ps1
```

Establish a reproducible gameplay baseline by comparing a checkpoint's
deterministic policy with a masked-random policy on the same explicit seeds:

```powershell
uv run autodancer-baseline `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --checkpoint ".\runs\bard-ppo\final.pt" `
  --num-instances 4 `
  --seeds "17001,17002,17003,17004,17005,17006,17007,17008" `
  --max-steps 256 `
  --output ".\runs\bard-ppo\baseline.json"
```

The report records per-seed returns, survival, progress, gold, damage, kills,
pickups, deaths, and completions for both policies, plus their aggregate deltas.

The default action order is `up`, `right`, `down`, `left`, `wait`, `bomb`,
`item 1`, `item 2`, `throw`, `spell 1`, and `spell 2`. Observations contain a
`21 × 21 × 29` symbolic grid, a 21-value player vector, a `13 × 8` inventory,
a persistent `65 × 65 × 5` current-floor map memory, and an 11-value
legal-action mask. See [`docs/observation-parity.md`](docs/observation-parity.md)
for the human-information parity audit and remaining gaps.

The default versioned reward profile prioritizes floor progress over bounded
exploration and combat shaping. Pass `--reward-config weights.json` to override
individual weights; the exact profile is stored in checkpoints and must match
on resume. See [`docs/rewards.md`](docs/rewards.md) for the current defaults and
[`docs/reward-history.md`](docs/reward-history.md) for the experiment history,
observations, and next reward hypothesis.

See [`docs/protocol.md`](docs/protocol.md) for the schema-10 live-controller contract.
The first measured live-training reference is recorded in
[`docs/baseline.md`](docs/baseline.md).

Before changing rewards or architecture, qualify the complete live controller:

```powershell
uv run autodancer-qualify `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --mod-dir ".\mods\AutoDancer" `
  --num-instances 8 `
  --transitions-per-worker 125000 `
  --run-dir ".\runs\controller-qualification" `
  --device cuda
```

The gate requires 125,000 valid transitions from every worker with no natural
controller fault. Pre-soak phases can be resumed with `--resume`; the natural
soak always starts from zero after a failure.

## Experiment lineage

Architecture and reward experiments use Git-tracked immutable contracts plus a
local MLflow runtime store. Install the `lineage` extra, validate the registry,
and launch the local UI with:

```powershell
uv sync --extra train --extra lineage --extra test
uv run autodancer-experiment validate
uv run autodancer-experiment ui
```

Training and baseline evaluation accept `--experiment-id`, `--experiment-arm`,
and `--trial-id`. See [`docs/experiment-lineage.md`](docs/experiment-lineage.md)
for declaration, backfill, decision, and baseline-promotion workflows.
