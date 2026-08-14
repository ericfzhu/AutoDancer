# AutoDancer

AutoDancer is a simulator-first reinforcement learning project for **Crypt of
the NecroDancer**. The target is a Bard policy that completes the four base-game
zones without human demonstrations.

The repository has two Gymnasium environments:

- `AutoDancer-Sim-v0` is a fast deterministic Python simulator.
- `AutoDancer-Live-v0` reads a local SYNCHRONY mod trace and sends macOS keys.

The first policy receives symbolic data. It does not receive pixels. Both
environments can render a `256 × 256` RGB image.

## Current scope

This release is the project foundation. It includes deterministic generation,
visibility, movement, wall digging, dagger attacks and throws, enemy movement,
health, stairs, spike traps, bombs, gold, floor and zone transitions, generated
boss floors, six curriculum tasks, raw event logs, shaped rewards, trace
comparison, and a recurrent PPO training entry point.

The simulator has clean-room approximations for enemies in Zones 1 to 4. It is
not yet a frame-accurate copy of the full game. Each exact rule must be added
with a live conformance trace. The Lua exporter also starts with a narrow safe
observation. It leaves unconfirmed values at zero.

## Setup

Native Windows supports the simulator, live game adapter, and PyTorch model
dependencies. Sample Factory itself is POSIX-only, so PPO training must run on
Linux, macOS, or WSL; the Windows training entry point exits with a clear error.

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --extra test
uv run pytest
```

Install training dependencies when you need them:

```sh
uv sync --extra train
```

Create and inspect an environment:

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

## Tasks and seeds

The task names are `navigation`, `single_enemy`, `mixed_room`, `floor`, `zone`,
and `all_zones`. Training uses one policy across these tasks. See
`src/autodancer/training/curriculum.py` for the adaptive task mixture and fixed
seed splits.

## Training

The default command starts Sample Factory recurrent PPO with 64 environments
and eight worker processes:

```sh
uv run autodancer-train --experiment=autodancer_baseline
```

Training output goes to `runs/`. Keep validation and test seeds out of training.
The final report must show completion rate, furthest floor, deaths, turns, and
game score. Shaped reward is not a completion metric.

## Live game

The live path requires the base game and SYNCHRONY. It supports Windows and
Apple Silicon macOS. It excludes AMPLIFIED, Zone 5, and character DLC.

Read `mods/AutoDancer/README.md` before a live run. The Python process needs
macOS Accessibility and Screen Recording permission. Keep the game focused.
Pin each trace to the installed game version and Steam build.

The local Lua mod prints telemetry to the normal debug log. It does not open a
file, network socket, or IPC channel. AutoDancer contains no game assets and no
copied game source.

## Conformance work

Store black-box JSON Lines traces in `tests/conformance`. Compare one with:

```sh
uv run autodancer-compare-trace tests/conformance/example.jsonl
```

Add or change a game rule only with a trace that checks state after every action.
Report simulator results and live-game results separately.

Useful references:

- [Gymnasium custom environments](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/)
- [Sample Factory](https://www.samplefactory.dev/)
- [SYNCHRONY API](https://vortexbuffer.com/synchrony/docs/)
- [Brace Yourself Games mod policy](https://braceyourselfgames.com/mod-policy/)

