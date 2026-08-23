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
`21 × 21 × 19` symbolic grid, a 20-value player vector, a `13 × 8` inventory,
a persistent `65 × 65 × 5` current-floor map memory, and an 11-value
legal-action mask. See [`docs/observation-parity.md`](docs/observation-parity.md)
for the human-information parity audit and remaining gaps.

The default versioned reward profile prioritizes floor progress over bounded
exploration and combat shaping. Pass `--reward-config weights.json` to override
individual weights; the exact profile is stored in checkpoints and must match
on resume. See [`docs/rewards.md`](docs/rewards.md) for the current defaults and
[`docs/reward-history.md`](docs/reward-history.md) for the experiment history,
observations, and next reward hypothesis.

See [`docs/protocol.md`](docs/protocol.md) for the schema-7 wire contract.
The first measured live-training reference is recorded in
[`docs/baseline.md`](docs/baseline.md).
