# AutoDancer

AutoDancer trains a recurrent PPO agent directly against independent
**Crypt of the NecroDancer** Bard games. There is no simulator, screenshot
capture, keyboard automation, or UI control.

One hidden coordinator uses SYNCHRONY's native multi-instance API to create a
fixed number of workers. Each worker has its own UID, config, command file,
engine log, seed, run ID, and GRU state. Python refuses to start around an
unowned NecroDancer process and never silently trains with fewer workers than
requested.

## Install

Install the project and training dependencies:

```powershell
uv sync --extra train --extra test
```

Copy [`mods/AutoDancer`](mods/AutoDancer) to the unpackaged SYNCHRONY mod
directory and enable it under **Customize → Mods**. The supported initial game
build is `v4.2.1-b5713`. Lua changes require a mod reload or a game restart.

## Train

Close every existing NecroDancer process first. The supervisor launches one
hidden coordinator plus exactly `--num-instances` native workers:

```powershell
uv run autodancer-train `
  --game-dir "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64" `
  --mod-dir "$env:LOCALAPPDATA\NecroDancer\mods\AutoDancer" `
  --num-instances 4 `
  --total-steps 1000000 `
  --run-dir ".\runs\bard-ppo" `
  --device auto
```

Rollouts contain 128 valid transitions per worker. PPO uses 32-step recurrent
chunks, four update epochs, action masking, periodic deterministic evaluation,
atomic checkpoints, and JSONL metrics. Continue an exact optimizer/model/RNG
state with `--resume .\runs\bard-ppo\latest.pt`.

The default action order is `up`, `right`, `down`, `left`, `wait`, `bomb`,
`item 1`, `item 2`, `throw`, `spell 1`, and `spell 2`. Observations contain a
`21 × 21 × 7` symbolic grid, a 16-value player vector, an `8 × 3` inventory,
and an 11-value legal-action mask.

See [`docs/protocol.md`](docs/protocol.md) for the schema-4 wire contract.
