# AutoDancer

AutoDancer connects a reinforcement-learning agent directly to one running
**Crypt of the NecroDancer** instance. It supports Bard only. There is no game
simulator in this repository.

The control loop is synchronous:

1. Python atomically writes one command to `bridge-command.txt`.
2. `Bridge.lua` polls that file and injects the corresponding SYNCHRONY action.
3. `AutoDancer.lua` logs the post-turn symbolic observation and echoes the
   Python session ID, command ID, requested action, and observed engine action.
4. Python rejects any transition that does not acknowledge the exact command.

The Gymnasium environment is `AutoDancer-Live-v0`. Its action order is `up`,
`right`, `down`, `left`, `wait`, `bomb`, `item 1`, `item 2`, `throw`, `spell 1`,
and `spell 2`. Observations contain a `21 × 21 × 7` symbolic grid, a 16-value
player vector, an `8 × 3` inventory, and an 11-value action mask.

## Local setup

Copy [`mods/AutoDancer`](mods/AutoDancer) to the unpackaged SYNCHRONY mod
directory and enable it under **Customize → Mods**. After changing Lua files,
reload the mod or restart the game.

For the current Windows installation, the two paths used by Python are:

```text
log:     X:\Steam\steamapps\common\Crypt of the NecroDancer\Necrodancer64\NecroDancer.log
command: C:\Users\cobra\AppData\Local\NecroDancer\mods\AutoDancer\bridge-command.txt
```

Create an environment for an already-running Bard run:

```python
from autodancer.envs.live import AutoDancerLiveEnv

env = AutoDancerLiveEnv(
    log_path=r"X:\Steam\steamapps\common\Crypt of the NecroDancer\Necrodancer64\NecroDancer.log",
    command_path=r"C:\Users\cobra\AppData\Local\NecroDancer\mods\AutoDancer\bridge-command.txt",
    attach_existing=True,
)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(0)
```

`reset()` publishes a Lua restart command unless `attach_existing=True`. No
keyboard focus, key injection, screenshot capture, trace comparison, or Python
game model is involved.

Run the bounded symbolic explorer against an existing Bard run:

```powershell
uv run autodancer-live-explore `
  "X:\Steam\steamapps\common\Crypt of the NecroDancer\Necrodancer64\NecroDancer.log" `
  "C:\Users\cobra\AppData\Local\NecroDancer\mods\AutoDancer\bridge-command.txt" `
  --max-turns 100
```

The next scaling step, after this single-instance handshake is reliable, is to
give each game process a distinct user-data/mod directory and therefore a
distinct command file and log. Each Python environment will then own exactly
one `(command_path, log_path)` pair.

See [`docs/protocol.md`](docs/protocol.md) for the wire contract.
