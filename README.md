# AutoDancer

AutoDancer is a simulator-first reinforcement learning project for **Crypt of
the NecroDancer**. The target is a Bard policy that completes the four base-game
zones without human demonstrations.

The repository has two Gymnasium environments:

- `AutoDancer-Sim-v0` is a fast deterministic Python simulator.
- `AutoDancer-Live-v0` reads telemetry from a local SYNCHRONY mod and sends
  focused game input.

The policy receives symbolic data rather than pixels. Both environments can
render a `256 × 256` RGB image for inspection.

## Current scope

This release is the project foundation. It includes deterministic generation,
visibility, movement, wall digging, dagger attacks and throws, enemy movement,
health, stairs, spike traps, bombs, gold, floor and zone transitions, generated
boss floors, six curriculum tasks, raw event logs, shaped rewards, live trace
recording and comparison, recurrent PPO training, and recurrent policy export
for native Windows inference.

The simulator contains clean-room approximations for enemies in Zones 1 to 4. It
is not yet a frame-accurate copy of the full game. Add or change an exact rule
only with a version-pinned live conformance trace.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The full POSIX test suite
includes training-model tests, so install both extras:

```sh
uv sync --extra test --extra train
uv run pytest
uv run ruff check .
```

For simulator-only work, the base dependencies are enough. Add a platform extra
for screen capture and live input when needed.

Create and inspect a simulator environment:

```python
import gymnasium as gym
import autodancer

env = gym.make("AutoDancer-Sim-v0", task="navigation", render_mode="rgb_array")
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(1)
frame = env.render()
```

The action order is `up`, `right`, `down`, `left`, `wait`, `bomb`, `item 1`,
`item 2`, `throw`, `spell 1`, and `spell 2`.

## Observation contract

Simulator and live policies use the same grid, inventory, action mask, and
16-value player vector. The enemy-count feature means **visible enemies only**;
it never exposes the simulator's hidden global enemy count. Task identity is
explicit deployment context rather than hidden game state. The live adapter
re-derives visible enemy count, on-stairs state, task identity, and terminal
flags before handing each observation to a policy.

## Tasks and seeds

The task names are `navigation`, `single_enemy`, `mixed_room`, `floor`, `zone`,
and `all_zones`. Training uses one policy across these tasks. See
`src/autodancer/training/curriculum.py` for the adaptive task mixture and fixed
seed splits.

## Training on a Windows desktop with the RTX 3070

Sample Factory does not support native Windows. Run training in **WSL2** so the
Linux training process can use the desktop's NVIDIA GPU. Run the game and live
adapter in native Windows.

Inside WSL2:

```sh
uv sync --extra train --extra test
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
uv run autodancer-train --experiment=autodancer_baseline
```

Training automatically selects `gpu` when PyTorch can see CUDA and otherwise
falls back to `cpu`; pass `--device=gpu` to require CUDA explicitly. The default
configuration requests 64 environments, eight workers, and a GRU policy.
Simulator stepping and observation construction are CPU work, so profile
steps-per-second before increasing the worker count solely because a faster GPU
is available.

Training output goes to `runs/`. Keep validation and test seeds out of training.
The final report must show completion rate, furthest floor, deaths, turns, and
game score. Shaped reward is not a completion metric.

## Exporting the recurrent policy to native Windows

Export a checkpoint inside WSL2. The resulting TorchScript program has an
explicit GRU-state input and does not require Sample Factory at inference time:

```sh
uv run autodancer-export-policy \
  --experiment autodancer_baseline \
  --checkpoint-kind best \
  --output /mnt/c/AutoDancer/policies/autodancer_best.pt
```

Then run it from native Windows, where the game and SYNCHRONY are running:

```powershell
uv sync --extra windows --extra train
uv run autodancer-live-run `
  "C:\path\to\NecroDancer.log" `
  --policy "C:\AutoDancer\policies\autodancer_best.pt" `
  --task all_zones
```

Live inference chooses CUDA when available. The network is small, so
`--device=cpu` can be preferable while the game is sharing the GPU.

## Live game

The live path requires the base game and SYNCHRONY. Native Windows is the
maintained deployment path for this project. The existing macOS adapter remains
a legacy movement-only default and was not expanded for special Bard actions as
part of the Windows work. AMPLIFIED, Zone 5, and character DLC are outside the
current scope.

Read `mods/AutoDancer/README.md` before a live run. Keep the game focused, do not
provide manual input during automation, and pin every trace to the installed
game version and Steam build. The Windows controlled-restart key defaults to
**F8**. Bard special actions use the directional key chords defined in
`src/autodancer/live/windows.py`.

Protocol schema 2 gives every run a stable `run_id`, verifies consecutive turn
sequence numbers, reports explicit `running`, `won`, `dead`, and `aborted`
episode states, and applies a configurable client-side maximum-turn guard.
Successful live runs now terminate as completed episodes rather than waiting
forever for another turn.

The local Lua mod prints telemetry to the normal debug log. It does not open a
file, network socket, or IPC channel. AutoDancer contains no game assets and no
copied game source.

## Conformance work

Record a version-pinned live trace with the bounded symbolic explorer:

```powershell
uv run autodancer-record-trace `
  "C:\path\to\NecroDancer.log" `
  "traces\live\example.jsonl" `
  --task all_zones --max-turns 100
```

The recorder sends the configured restart key by default and refuses to
overwrite an existing trace unless `--overwrite` is passed. It always stores the
raw live observation as evidence. Use `--compare-observation`,
`--compare-events`, or `--compare-reward` only when those captured values are
intended to be simulator assertions.

Compare a curated trace with the simulator:

```sh
uv run autodancer-compare-trace traces/live/example.jsonl
```

Trace schema 2 supports partial observations, events, rewards, terminal flags,
and canonical simulator state. Repeated `--ignore` path/glob patterns should
cover only fields outside the mechanic under test. A raw live map will not
automatically match the clean-room generator, so conformance traces should be
reduced to one mechanic and state exactly which paths are authoritative. Legacy
trace-schema-1 state traces remain readable.

Report simulator results and live-game results separately.

Useful references:

- [Gymnasium custom environments](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/)
- [Sample Factory](https://www.samplefactory.dev/)
- [SYNCHRONY API](https://vortexbuffer.com/synchrony/docs/)
- [Brace Yourself Games mod policy](https://braceyourselfgames.com/mod-policy/)
