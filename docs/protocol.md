# Live protocol

The local SYNCHRONY mod prints one line after each completed turn and a final
line when a run ends. Each line starts with `AUTODANCER_JSON:`. Python tails the
normal NecroDancer debug log. The mod does not open a file, network connection,
or IPC connection.

## Schema 2

Each record contains:

- `schema_version`: `2`
- `run_id`: stable for the complete run and different after a restart
- `sequence`: starts at `0` after restart and increases by one
- `kind`: `reset`, `turn`, or `terminal`
- `game.version` and `game.steam_build`
- `character`: must be `Bard`
- `seed`, `zone`, and `floor`
- `observation.grid`: `21 × 21 × 7` integers
- `observation.player`: 16 integers
- `observation.inventory`: `8 × 3` integers
- `observation.action_mask`: 11 binary values
- `events`: raw game events, separate from rewards
- `episode_status`: `running`, `won`, `dead`, or `aborted`
- `terminated`: true for `won` and `dead`
- `truncated`: true for `aborted`
- `metrics`: confirmed counters such as turns, completion, and deaths

Python rejects malformed shapes, non-integer observations, out-of-range enum
values, non-binary action masks, negative event amounts, invalid event objects,
inconsistent terminal flags, sequence gaps, unannounced run-ID changes, missing
sequences, and unpinned game builds. Terminal records may mask every action;
running records may not.

Tiles outside player visibility have no actor, health, item, trap, or status
values. The player vector uses visible enemy count rather than a hidden global
enemy count. Python re-derives visible enemy count, on-stairs state, task
identity, won, and dead before exposing an observation to a policy.

## Episode lifecycle

- A new run emits sequence `0`, kind `reset`, status `running`.
- Normal completed inputs emit kind `turn`, status `running`.
- Player death emits kind `terminal`, status `dead`, and a `failure` event.
- Final all-zones completion emits kind `terminal`, status `won`, and a `success`
  event whose data contains `task_complete: true`.
- A non-victory run completion not already classified as death emits kind
  `terminal`, status `aborted`, and a `failure` event.
- The Python environment additionally truncates at its configured maximum turn
  count, preventing an evaluation loop from running forever if the game fails
  to produce a terminal signal.

The `run_id` plus sequence check prevents stale records, floor transitions, and
new runs from being silently mixed. It does not prove that a record was caused
by the exact key chord Python just sent; manual input during an automated run
must still be avoided. A future protocol version should echo a command ID or
action to close that remaining correlation gap.

## Conformance traces

The live protocol version and the conformance-trace version are separate.
Newly recorded conformance traces use trace schema 2 and identify the live
protocol version in their header. The comparison tool also reads legacy
trace-schema-1 state traces.

A schema-2 recording stores raw live observations as evidence. Captured values
become simulator assertions only when the corresponding compare option is
selected or when a curated trace adds partial `observation`, `state`, `events`,
or `reward` fields.

## Confirmed live semantics

- A freshly loaded exporter labels sequence `0` as `reset`, including a hot
  reload. Floor changes within a run keep the same run ID and increasing
  sequence.
- Terrain value `3` is detected from tile metadata (`name == "Stairs"` or a
  positive `descent`), rather than from a build-specific tile ID.
- Entering stairs starts a short descent delay; the floor field changes only
  when the next level has loaded.
- Enemy damage and kills are emitted only for actor victims. Destroyed chests
  and crates emit `container_opened` and do not receive enemy rewards.
- Zone 1 live traces distinguish `Skeleton2`/`Skeleton2Headless` (actor 13) and
  `Dragon` (actor 14) from ordinary skeletons and generic bosses.
