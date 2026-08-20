# Python/Lua live protocol

Protocol schema 3 controls one running game instance through two local files:

- Python atomically replaces `bridge-command.txt` in the installed mod folder.
- Lua prints `AUTODANCER_JSON:` records to `NecroDancer.log`.

## Commands

Each command is one ASCII line:

```text
ACTION <session_id> <command_id> <logical_action>
RESTART <session_id> <command_id> -1
```

Logical actions are integers `0..10`. Command IDs increase within a randomly
generated Python session. Atomic replacement prevents Lua from reading a
partially written command.

## Transition acknowledgement

An agent-driven turn record contains:

```json
{
  "bridge": {
    "session_id": "...",
    "command_id": 1,
    "requested_action": 0,
    "engine_action": 3,
    "observed_action": 3
  }
}
```

The live Gym environment verifies the session ID, command ID, and requested
action before returning from `step()`. Missing, stale, manual, duplicated, or
out-of-order transitions raise `ProtocolError`.

The rest of each record contains a stable `run_id`, consecutive `sequence`,
`kind` (`reset`, `turn`, or `terminal`), pinned game build, Bard character,
seed, zone, floor, symbolic observation, raw events, terminal status, and
metrics. Running observations must expose at least one legal action.

For multiple instances, each game process must use a separate mod/user-data
root and log. The protocol itself does not require shared state between
instances.
