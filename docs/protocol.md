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
- `metrics`: confirmed counters such as turns

Unknown values are zero. Tiles outside player visibility have no actor, health,
item, trap, or status values. Python rejects records with a missing sequence or
an unpinned game build.

The live mod scaffold only exports fields that have confirmed public API access.
Add a new field after a live trace confirms its meaning.

## Confirmed live semantics

- A freshly loaded exporter labels sequence `0` as `reset`, including hot reloads
  on floors after floor 1. Floor changes within a run keep the sequence increasing.
- Terrain value `3` is detected from tile metadata (`name == "Stairs"` or positive
  `descent`), rather than from a build-specific tile ID.
- Entering stairs starts a short descent delay; the floor field changes only when
  the next level has loaded.
- Enemy damage and kills are emitted only for actor victims. Destroyed chests and
  crates emit `container_opened` and do not receive enemy rewards.
- A player death record has health zero, `terminated: true`, a `player_damage`
  event, and a `failure` event.
- Zone 1 live traces distinguish `Skeleton2`/`Skeleton2Headless` (actor 13) and
  `Dragon` (actor 14) from ordinary skeletons and generic bosses. The observed
  Skeleton2 had 2 HP and dealt 3 damage; the observed Dragon had 4 HP.
