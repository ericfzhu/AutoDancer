# AutoDancer SYNCHRONY mod

This is an unpackaged local mod. It prints one schema-2 telemetry record to the
normal game debug log after each turn and an explicit terminal record when a run
ends.

1. Copy the complete `AutoDancer` directory, including `mod.json`, into the
   SYNCHRONY unpackaged-mod directory.
2. Set `GAME_VERSION` and `STEAM_BUILD` in `scripts/AutoDancer.lua`.
3. Open **Customize → Mods** and enable AutoDancer.
4. Press **Shift+F7** after a script change.
5. Keep the game focused during a live run and avoid manual input.
6. Bind **F8** to a controlled restart for the default Windows adapter.

The exporter includes revealed terrain, visible entities, safe inventory fields,
raw events, action masks, a stable run ID, and explicit episode status. Unknown
or unconfirmed values stay zero. The policy receives only visible dynamic data;
task identity is supplied explicitly by Python.

The default Windows adapter supports movement and Bard multi-key actions. The
macOS default mapping remains movement-only and is retained mainly for legacy
development use.

## Phase-one real-engine probe

`EngineProbe.lua` is disabled until explicitly started. It benchmarks whether the
real game can supply reinforcement-learning transitions faster than normal play:

- **F6** drives a bounded movement script through `necro.client.Input.add` at
  normal game-loop pacing.
- **F7** drives the same script through `ClientActionBuffer.addAction` followed
  by experimental `Turn.process` stepping.
- **F5** stops either probe.

Plain **F7** starts the accelerated probe; **Shift+F7** reloads scripts.

For a formal comparison, start an **All Zones Seeded** run as Bard, enter a fixed
seed, stop on the first playable turn, and do not make any manual move before
starting the probe. Repeat the complete run with the same seed for:

1. normal-input baseline A;
2. normal-input baseline B;
3. direct-process candidate.

The two normal baselines must agree before the direct-process candidate is
interpreted. All captures must report the requested telemetry seed and matching
game build, character, mode, settings, action script, and command count.

Start `autodancer-benchmark-engine collect` before pressing F6 or F7. The
complete runbook in `docs/engine-probe.md` covers:

- native-Windows setup and log discovery;
- seeded run reproduction;
- capture and comparison commands;
- formal per-capture and per-seed acceptance criteria;
- exit codes and common failure reasons;
- throughput decision bands;
- the real-engine-primary, hybrid, direct-input-only, limited-pool, and
  simulator-first downstream paths.

Accelerated stepping is not valid merely because it is fast. Its post-turn
telemetry must match an equivalent repeated normal-input run, with contiguous
command IDs, unit turn deltas, no unpaired telemetry, and no action mismatches.

The current probe starts from a manually created seeded run. SYNCHRONY exposes
seeded game modes and generator seed options, so automatic seeded start/reset is
a feasible later step, but it is not implemented by this phase-one mod.

The mod does not open files, sockets, or IPC channels. It contains no game
assets and no copied game source.
