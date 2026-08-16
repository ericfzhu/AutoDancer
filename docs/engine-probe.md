# Real-engine phase-one benchmark

This probe answers one question before AutoDancer invests in multi-process game
training:

> How many trustworthy Bard transitions per second can one real NecroDancer
> process produce when the mod drives the engine directly?

It does not train a policy and it does not require Python-to-Lua IPC.

## Install

Copy the complete `mods/AutoDancer` directory to the SYNCHRONY unpackaged-mod
directory, enable it, and reload scripts. Start a Bard run and keep the game in a
normal playable state.

The benchmark uses three mod hotkeys:

- **F6** starts 256 scripted movement actions through `necro.client.Input.add`.
  The normal game loop controls pacing.
- **F7** starts the same action script through
  `ClientActionBuffer.addAction` followed by `Turn.process`. This is the
  experimental accelerated mode.
- **F5** stops the active probe.

The fixed action sequence is right, left, up, down. Edit `DEFAULT_COMMANDS`,
`PROCESS_BATCH_SIZE`, or `ACTION_SCRIPT` near the top of
`EngineProbe.lua` for a different stress test.

## Collect a run

Start the collector before pressing the probe hotkey:

```powershell
uv run autodancer-benchmark-engine collect `
  "C:\path\to\NecroDancer.log" `
  --timeout 120 `
  --capture "captures\normal-input.jsonl"
```

Press **F6** for the baseline. Repeat from an equivalent game state and press
**F7** for accelerated turn processing:

```powershell
uv run autodancer-benchmark-engine collect `
  "C:\path\to\NecroDancer.log" `
  --timeout 120 `
  --capture "captures\direct-process.jsonl"
```

The summary reports turns per second, command latency percentiles, command-ID
gaps, non-unit turn deltas, unpaired telemetry, and client-input action
mismatches.

A probe can end before 256 commands if Bard dies, the level becomes unavailable,
or direct turn processing fails. That is a useful result rather than a silent
success.

## Compare semantics

For the strongest comparison, run both modes from the same game build, seed, and
starting state with the same action script. Then compare the captured post-action
telemetry:

```powershell
uv run autodancer-benchmark-engine compare `
  "captures\normal-input.jsonl" `
  "captures\direct-process.jsonl"
```

The comparison ignores run IDs, telemetry sequence numbers, entity IDs, debug
entity lists, and timing. It compares the symbolic observation, normalized
events, seed, zone, floor, episode status, and terminal flags after every
command. The first divergent command is reported.

## Safety and interpretation

`Turn.process` is deliberately treated as experimental. A high turns-per-second
number is useful only when the captured state sequence remains equivalent to the
normal-input baseline. Test several seeds and include floor transitions, combat,
traps, and death before using accelerated stepping for reinforcement-learning
experience collection.

This probe uses supported Synchrony modules and does not alter the process clock,
patch executable memory, bypass the mod sandbox, or interact with leaderboards.
