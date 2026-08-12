# Live protocol

The local SYNCHRONY mod prints one line after each completed turn. The line starts
with `AUTODANCER_JSON:`. Python tails the normal NecroDancer debug log. The mod
does not open a file, network connection, or IPC connection.

Each record uses protocol schema 1. It contains:

- `schema_version`: `1`
- `sequence`: starts at `0` after restart and increases by one
- `kind`: `reset` or `turn`
- `game.version` and `game.steam_build`
- `character`: must be `Bard`
- `seed`, `zone`, and `floor`
- `observation.grid`: `21 × 21 × 7` integers
- `observation.player`: 16 integers
- `observation.inventory`: `8 × 3` integers
- `observation.action_mask`: 11 values, each `0` or `1`
- `events`: raw game events, separate from rewards
- `terminated` and `truncated`
- `metrics`: turns, deaths, floor progress, and game score when available

Unknown values are zero. Tiles outside player visibility have no actor, health,
item, trap, or status values. Python rejects records with a missing sequence or
an unpinned game build.

The live mod scaffold only exports fields that have confirmed public API access.
Add a new field after a live trace confirms its meaning.
