# Python/Lua live protocol

Schema 5 identifies every message with `instance_id` and `role`. Python owns a
current-user-only Windows named pipe for each process. Pipe names include the
supervisor session and worker identity, so commands cannot cross worker slots.
Lua retains a received command until the engine accepts it in a safe state.

## Readiness

Every process prints one structured marker in its own engine log:

```text
AUTODANCER_READY:{"schema_version":5,"instance_id":"worker-0000","role":"worker",...}
```

Python discovers logs by marker identity, validates schema and pinned game
build, and rejects cross-worker records.

## Commands

Each command is one ASCII line:

```text
SPAWN <session_id> <command_id> <worker_id>
CLOSE <session_id> <command_id> <worker_id>
RESET <session_id> <command_id> <seed>
ACTION <session_id> <command_id> <logical_action>
```

`RESET` starts Bard in normal All Zones mode using the exact requested seed.
Logical actions are integers `0..10`. Process creation and replacement are
owned by the Python supervisor; the coordinator never controls the game UI.

## Transition acknowledgement

An action-driven transition echoes:

```json
{
  "schema_version": 5,
  "instance_id": "worker-0000",
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
requires its top-level seed to match. It rejects stale sessions, duplicate or
out-of-order acknowledgements, unexpected run-ID changes, malformed records,
and records belonging to another worker.

## Symbolic observation

The worker emits a player-centred `21 × 21 × 11` grid. Its channels are terrain
class/type, actor class/type/current HP/max HP, item class/type, trap class,
visibility, and status flags. Exact types use a deterministic 12-bit hash with
zero reserved for absence; the coarse classes remain available so the policy
can generalize across related enemies and items.

Inventory is `8 × 4`: coarse item class, exact type, quantity, and weapon
damage. Player features and the legal-action mask retain their existing shapes.
All values are range-checked before entering the policy.
