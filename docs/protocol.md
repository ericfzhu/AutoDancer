# Python/Lua live protocol

Schema 4 identifies every message with `instance_id` and `role`. The
coordinator reads `bridge-command.coordinator.txt`; worker `worker-0000` reads
`bridge-command.worker-0000.txt`, and so on. Commands remain in the mounted
file until the engine accepts them in a safe state.

## Readiness

Every process prints one structured marker in its own engine log:

```text
AUTODANCER_READY:{"schema_version":4,"instance_id":"worker-0000","role":"worker",...}
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

`SPAWN` creates an independent SYNCHRONY instance with a unique UID and config
name. `RESET` starts Bard in normal All Zones mode using the exact requested
seed. Logical actions are integers `0..10`.

## Transition acknowledgement

An action-driven transition echoes:

```json
{
  "schema_version": 4,
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
