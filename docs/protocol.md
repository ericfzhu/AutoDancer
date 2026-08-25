# Python/Lua live protocol

Schema 10 identifies every pipe message with `instance_id`, supervisor
`session_id`, and an attempt-specific `launch_id`. Python owns a
current-user-only Windows named pipe for each process. Pipe names include all
three identities, so stale launches cannot satisfy readiness or send records to
replacement workers. Lua retains a received command until the engine accepts it
in a safe state.

## Readiness

The first pipe message from every process is a validated `HELLO`:

```json
{
  "message_type": "hello",
  "schema_version": 10,
  "instance_id": "worker-0000",
  "session_id": "...",
  "launch_id": "...",
  "game_version": "v4.2.1-b5713",
  "steam_build": "22938426"
}
```

Python validates the connection, identities, schema, and pinned game build. The
log readiness marker remains diagnostic evidence and never makes a worker ready.

## Commands

Each command is one ASCII line:

```text
RESET <session_id> <command_id> <seed>
ACTION <session_id> <command_id> <logical_action>
```

`RESET` starts Bard in normal All Zones mode using the exact requested seed.
Logical actions are integers `0..10`. Process creation and replacement are
owned directly by the Python supervisor; no coordinator game process exists.

The logical action order is `up`, `right`, `down`, `left`, `wait`, `bomb`,
`item 1`, `item 2`, `throw`, `spell 1`, and `spell 2`. In a running Bard
record, the four directions and wait must always be enabled. Each remaining
mask bit must exactly match the corresponding inventory slot and cooldown
state; Python rejects the complete record if it does not.

## Command lifecycle and transition acknowledgement

Every outstanding command emits lifecycle messages: `received`, `deferred`,
`accepted`, `input_observed`, `turn_completed`, and `telemetry_sent`, or a
terminal `command_error`. While a command is pending, a one-second heartbeat
reports the pending command, loading/lobby/cutscene/player gates, level identity,
and game tick. Lifecycle and heartbeat messages never advance the transition
sequence.

An action-driven transition echoes:

```json
{
  "message_type": "transition",
  "schema_version": 10,
  "instance_id": "worker-0000",
  "session_id": "...",
  "launch_id": "...",
  "role": "worker",
  "bridge": {
    "kind": "ACTION",
    "session_id": "...",
    "command_id": 1,
    "requested_action": 0,
    "engine_action": 3,
    "observed_action": 3
  }
}
```

A reset record acknowledges `kind`, session, command ID, and seed. Python also
requires its top-level seed to match. It rejects stale sessions or launches,
duplicate or out-of-order acknowledgements, unexpected run-ID or seed changes,
malformed records, and records belonging to another worker. It also requires a
running observation's zone and floor to match the transition metadata, ensuring
that the first policy action after a floor change uses the current floor. For an
action, `observed_action` is
mandatory and must equal `engine_action`; an echoed request without an engine
observation is not considered proof that the action occurred.

## Symbolic observation

Transition records are UTF-8 JSON messages sent directly over the worker's
duplex named pipe and are limited to 256 KiB. Reads preserve Windows message
boundaries, including `ERROR_MORE_DATA`; adjacent messages are never combined.
Reads and writes have deadlines. Debug logs carry readiness and fatal
diagnostics only; Python does not tail them for live transitions.

The worker emits a player-centred `21 × 21 × 29` grid. Its channels are terrain
class/type, actor class/type/current HP/max HP, item class/type, trap class,
visibility, status flags, facing, beat-delay counter and interval, visible
freeze/confusion duration, charge state and direction, and shield direction.
These are raw visible cues rather than engine decisions or AI targets. Exact
types use a deterministic 12-bit hash with zero reserved for absence; the
coarse classes remain available so the policy can generalize across related
enemies and items. Ten additional channels describe visible object class/type,
interaction flags, price currency and amount, visible health cost, trap
activation/failure animation time, attack-tell animation time, and explosive
state. Interaction flags distinguish interactable, locked, active-shrine,
active-sale, and shopliftable state. Hidden container and shrine contents are
never serialized.

Inventory is `13 × 8` and covers weapon, two action slots, shovel, two spells,
bombs, misc, body, head, feet, torch, and ring. Each row carries coarse and
exact type, quantity, weapon damage, turn and kill cooldowns, readiness, and
toggle state. The 21 player fields include song elapsed, total, and remaining
time in deciseconds, the visible song-end state, and effective shopkeeper-song
volume in basis points. This is the music-layer contribution a player hears,
not a hidden shop location. All values are range-checked before entering the
policy.

The schema-9 observation representation remains unchanged inside schema-10
transition envelopes. It carries absolute `observation.map_bounds` and may carry
`observation.revealed_map`, a raw `65 × 65` terrain snapshot anchored at the
floor's spawn position. Lua emits it
periodically and immediately while the Map item is held. Zero is unknown;
non-zero cells are terrain the game has actually marked revealed. Python
combines these snapshots with the local visibility grid and Bard's own path
into an absolute-coordinate history, then renders the derived `map_memory`
policy input as a `65 × 65` viewport centred on Bard. If the full level bounds
exceed that supported capacity, collection fails with a map-capacity error
instead of silently clipping state or restarting workers. Distance travelled
from the spawn does not consume additional capacity.
