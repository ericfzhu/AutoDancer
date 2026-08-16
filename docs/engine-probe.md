# Real-engine phase-one benchmark

AutoDancer is currently deciding whether most reinforcement-learning experience
should come from the Python simulator, the real NecroDancer engine, or a hybrid
of both. This probe supplies the first evidence for that decision.

The primary question is:

> How many **trustworthy** Bard transitions per second can one real NecroDancer
> process produce when the SYNCHRONY mod drives the engine directly?

A fast result is useful only when it is reproducible and produces the same
post-action game states as the normal-input baseline.

The phase-one probe does not train a policy, launch multiple game processes, or
require Python-to-Lua IPC. It runs a deterministic action script inside one game
process while Python collects and checks the resulting telemetry.

## What this experiment tests

The benchmark evaluates these hypotheses separately:

1. `necro.client.Input.add` can execute the intended Bard action without Windows
   keyboard injection.
2. Each requested command produces exactly one completed logical game turn.
3. `ClientActionBuffer.addAction` followed by `Turn.process` can advance the
   game faster than the normal game loop.
4. Accelerated stepping produces the same post-turn symbolic state and events as
   normal direct input from an equivalent seeded starting state.
5. Command IDs, action acknowledgements, turn IDs, and telemetry records remain
   complete and correctly paired.
6. The achieved throughput is high enough to justify building a real-engine
   environment pool.

This phase does **not** yet test:

- neural-network inference or reinforcement-learning training;
- Python-to-Lua commands;
- automatic game-process launch or recovery;
- automatic seeded resets;
- two or more simultaneous game processes;
- external process-clock or speed-hack tools;
- rendering/audio suppression;
- long-run stability across millions of turns.

## Seeded runs and reproducibility

Crypt of the NecroDancer and SYNCHRONY support seeded runs. The public API
includes `GameSession.Mode.AllZonesSeeded`, manual seed modes, a
`LevelGenerator.Options.seed` field, and dungeon-seed accessors. This means an
automatic seeded start/reset path is feasible.

The current phase-one probe does **not** start the run itself. For now, start an
**All Zones Seeded** Bard run through the game UI and enter the same chosen seed
for every baseline/candidate pair. Begin each probe on the first playable turn
without making any manual move beforehand.

For every formal comparison, record all of the following:

- requested seed;
- seed reported by AutoDancer telemetry;
- game version and Steam build;
- AutoDancer Git commit;
- enabled mods and DLC;
- relevant game settings;
- character (`Bard`);
- mode (`All Zones Seeded`);
- action script and command count.

A pair is invalid if the telemetry seed differs from the requested seed. A pair
is also invalid if either run received input before the probe began.

The current capture format compares post-action records but does not yet store a
dedicated pre-action observation. Until that is added, use two independent
normal-input repetitions for each seed. Those baseline repetitions must agree
before the accelerated run is evaluated.

## End-to-end setup

### 1. Prepare the Python project

Run these commands from the repository root on native Windows:

```powershell
uv sync --extra test --extra windows
uv run pytest tests/test_engine_benchmark.py
```

The benchmark collector itself uses only the base project dependencies, but the
command above also verifies its local tests and installs the Windows live extras.

### 2. Install the unpackaged SYNCHRONY mod

1. Copy the complete `mods/AutoDancer` directory, including `mod.json`, into the
   SYNCHRONY unpackaged-mod directory.
2. Set `GAME_VERSION` and `STEAM_BUILD` near the top of
   `scripts/AutoDancer.lua` to the values for the installed game.
3. Open **Customize → Mods** and enable AutoDancer.
4. Reload scripts with **Shift+F7** after making a Lua change.
5. Confirm that plain **F7** is available for the accelerated probe. Do not
   confuse it with **Shift+F7**, which reloads scripts.
6. Confirm that the debug log contains lines beginning with
   `AUTODANCER_JSON:` after normal Bard actions.

Use the actual debug-log file in which those markers appear as `<LOG_PATH>` in
the commands below. The precise path can vary by installation and platform.

### 3. Confirm probe hotkeys

The benchmark uses:

- **F6** — 256 scripted movement actions through `necro.client.Input.add`; the
  normal game loop controls pacing.
- **F7** — the same script through `ClientActionBuffer.addAction` followed by
  experimental `Turn.process` stepping.
- **F5** — stop the active probe.

The fixed logical action cycle is `RIGHT`, `LEFT`, `UP`, `DOWN`. Edit
`DEFAULT_COMMANDS`, `PROCESS_BATCH_SIZE`, or `ACTION_SCRIPT` near the top of
`EngineProbe.lua` only when intentionally defining a different experiment. Keep
those values identical between a baseline and its candidate.

### 4. Choose and record a seed

Choose a fixed integer seed and create a capture directory containing it in the
file name, for example:

```powershell
$Seed = 123456789
New-Item -ItemType Directory -Force "captures\seed-$Seed" | Out-Null
```

Start an **All Zones Seeded** run as Bard with that seed. Stop on the first
playable turn and do not provide any gameplay input.

## Collect the normal-input baseline

Start the collector before pressing the probe hotkey:

```powershell
uv run autodancer-benchmark-engine collect `
  "<LOG_PATH>" `
  --timeout 120 `
  --capture "captures\seed-$Seed\baseline-a.jsonl" `
  --output "captures\seed-$Seed\baseline-a-summary.json"
```

Press **F6** once. Do not press movement keys or interact with another game
window while the probe is running.

Repeat the complete seeded run from the lobby using the same seed and first-turn
starting point, then collect a second F6 control:

```powershell
uv run autodancer-benchmark-engine collect `
  "<LOG_PATH>" `
  --timeout 120 `
  --capture "captures\seed-$Seed\baseline-b.jsonl" `
  --output "captures\seed-$Seed\baseline-b-summary.json"
```

Compare the two normal-input controls first:

```powershell
uv run autodancer-benchmark-engine compare `
  "captures\seed-$Seed\baseline-a.jsonl" `
  "captures\seed-$Seed\baseline-b.jsonl" `
  --output "captures\seed-$Seed\baseline-repeat-comparison.json"
```

If these do not match, the starting state or normal-input path is not yet
reproducible. Do not interpret an accelerated comparison for that seed.

## Collect the direct-process candidate

Recreate the same seeded Bard run again, stop on the first playable turn, and
start the collector:

```powershell
uv run autodancer-benchmark-engine collect `
  "<LOG_PATH>" `
  --timeout 120 `
  --capture "captures\seed-$Seed\direct-process.jsonl" `
  --output "captures\seed-$Seed\direct-process-summary.json"
```

Press **F7** once. Then compare it with the reproducible baseline:

```powershell
uv run autodancer-benchmark-engine compare `
  "captures\seed-$Seed\baseline-a.jsonl" `
  "captures\seed-$Seed\direct-process.jsonl" `
  --output "captures\seed-$Seed\process-comparison.json"
```

The comparison ignores run IDs, telemetry sequence numbers, entity IDs, debug
entity lists, and timing. It compares requested actions, the symbolic
observation, normalized events, seed, zone, floor, episode status, and terminal
flags after every command. It reports the first divergent command.

A probe may stop before 256 commands if Bard dies, the level becomes
unavailable, or direct processing fails. That is useful diagnostic evidence, but
it is not a completed movement smoke test.

## Formal acceptance criteria

### Per-capture integrity gate

Every capture used for a pass decision must satisfy all of these conditions:

```text
status == "completed"
commands == target_commands
command_ids_contiguous == true
non_unit_turn_deltas == 0
observed_action_mismatches == 0
unpaired_probe_turns == 0
malformed_probe_records == 0
telemetry seed == requested seed
```

Any failed condition invalidates that capture, regardless of its reported
turns-per-second.

### Seed-pair reproducibility gate

For each accepted seed:

1. `baseline-a` and `baseline-b` must be equivalent.
2. Both baselines must report the requested seed, game build, character, mode,
   and starting floor.
3. The direct-process capture must report the same metadata.
4. The direct-process capture must be equivalent to the accepted baseline.

### Phase-one smoke gate

The initial movement-only probe passes when at least **five fixed seeds** meet
all integrity and reproducibility requirements for 256 commands per capture:

```text
5 seeds × (2 normal baselines + 1 direct-process candidate)
```

A single successful seed is evidence that the mechanism can work; it is not
enough to choose the project architecture.

### Semantic coverage gate before RL experience collection

Before using accelerated real-engine turns for policy training, equivalent
seeded tests must also cover at least:

- unobstructed and blocked movement;
- wall digging;
- enemy movement and combat;
- traps;
- bombs, items, and special Bard actions;
- stairs and a floor transition;
- player death;
- run completion when practical.

The current `RIGHT, LEFT, UP, DOWN` script covers only the initial movement
smoke test. Additional bounded scripts or controlled scenarios are required for
this gate.

### Throughput decision bands

Correctness gates are absolute; throughput bands are planning heuristics. Use
**trusted turns per second**, meaning only turns from captures that passed every
integrity and equivalence check.

- **About 25 or more trusted turns/sec per instance:** proceed to a
  Python-to-Lua single-environment prototype and then a 1→2→4→8 process scaling
  test. This is in the range where eight instances could plausibly supply about
  100 million transitions in roughly one week.
- **About 5–25 trusted turns/sec per instance:** treat the real engine as a
  strong hybrid candidate for fine-tuning, conformance, and final evaluation,
  while retaining the simulator for bulk pretraining.
- **Below about 5 trusted turns/sec per instance, or unstable results:** retain
  simulator-first training unless later rendering/process optimizations change
  the result materially.

These bands can be revised when the target training-step budget is known.

## Exit codes

`autodancer-benchmark-engine collect` returns:

- `0` — the probe produced turns and finished with status `completed`;
- `1` — turns were captured, but the probe stopped, errored, or timed out;
- `2` — no probe turns were captured.

`autodancer-benchmark-engine compare` returns:

- `0` — both captures are equivalent under the current canonical comparison;
- `1` — the captures differ or have different lengths.

PowerShell exposes the code as `$LASTEXITCODE`. Preserve the JSON output even on
non-zero exits because it contains the diagnostic reason.

## Common failure reasons

- `game_not_ready` — the hotkey was pressed while the level was loading, the
  lobby/menu was active, input was blocked, or Bard was unavailable.
- `command_timeout` — a requested command did not complete within the probe's
  configured timeout.
- `turn_process_returned_without_turn` — experimental `Turn.process` returned
  without producing the expected turn event.
- `game_became_unavailable` — the run entered a loading/lobby state or otherwise
  became unavailable before reaching the target command count.
- `command_exception` — a SYNCHRONY API call raised an error; inspect the
  accompanying message in the probe record and game log.
- `stopped_by_hotkey` — F5 was pressed.
- `restarted_by_hotkey` — another probe was started while one was active.
- collector timeout with no turns — the collector was pointed at the wrong log,
  started after the relevant records, the mod was not loaded, or the probe never
  started.
- semantic mismatch at command 1 — first verify the seed, build, character,
  settings, first-turn starting point, and absence of manual input before
  treating it as a `Turn.process` difference.

## Downstream architecture paths

### Path A — equivalent, reliable, and fast

```text
Direct engine stepping passes all gates
        ↓
Build supported Python→Lua command transport
        ↓
Create one real-engine Gymnasium environment
        ↓
Add exact command/action acknowledgement and automatic seeded reset
        ↓
Add process supervision and crash recovery
        ↓
Scale 1 → 2 → 4 → 8 game processes
        ↓
Evaluate real-engine-primary training
```

The Python simulator can remain useful for cheap experiments and unit tests, but
the real engine becomes the behavioral authority.

### Path B — equivalent but only moderately fast

```text
Correct real-engine transitions, limited throughput
        ↓
Simulator bulk pretraining
        ↓
Small real-engine pool for fine-tuning
        ↓
Real-engine validation and final evaluation
```

This is the preferred hybrid path when fidelity is excellent but transition
volume is insufficient for the entire PPO workload.

### Path C — normal direct input works, accelerated stepping diverges

```text
Input.add baseline is reliable
Turn.process changes behavior or ordering
        ↓
Investigate target-turn/action-buffer semantics
        ↓
Minimize rendering and audio overhead
        ↓
Only then test controlled external time scaling
```

Do not apply an external 20× process clock until the normal direct-input capture
is reproducible. Every external speed setting must pass the same seeded
state-equivalence gates.

### Path D — one process works, scaling does not

```text
Single-process benchmark passes
Multi-process throughput or reliability collapses
        ↓
Use one or a few real instances for fine-tuning/evaluation
        ↓
Keep the simulator for high-volume experience
```

Potential scaling bottlenecks include CPU, rendering, memory, Steam process
behavior, logs, resets, and per-instance state isolation.

### Path E — real-engine stepping remains slow or unstable

```text
Neither direct path is operationally suitable for bulk training
        ↓
Continue simulator-first training
        ↓
Use the real game for conformance traces,
policy-transfer tests, and final evaluation
```

This is still a useful experimental result because it prevents spending further
engineering effort on an infeasible environment pool.

## Safety and interpretation

`Turn.process` is deliberately experimental. A high turns-per-second number is
not a pass when any state, event, action, or sequence check fails.

The probe uses supported SYNCHRONY modules. It does not alter the process clock,
patch executable memory, bypass the mod sandbox, or interact with leaderboards.
External time scaling, if investigated later, is a separate experiment and must
be evaluated against the accepted normal-input seeded baseline.
