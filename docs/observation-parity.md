# Player-information parity

The policy should receive the gameplay facts available to an attentive human
player, while never reading unrevealed engine state. This is distinct from
reward shaping: information makes a decision possible; reward says which
outcomes matter.

## Current parity

Schema 7 provides a persistent 65×65 current-floor memory anchored at Bard's spawn.
It combines the game's revealed terrain with Bard's position, visit count, and
visit recency. Dynamic enemies and items disappear when they leave sight, so
the map cannot become a hidden-state oracle. A periodic Lua snapshot also
captures distant terrain revealed by the Map item. The local 21×21 observation
remains the authoritative combat view.

This approximates what the HUD minimap and ordinary human spatial memory make
available. The reward tracker still owns its independent exact novelty sets;
those sets prevent reward farming but are not exposed directly.

Schema 7 also closes the three highest-priority gaps from the previous audit:
visible enemy facing/timing/status/attack cues; all thirteen HUD equipment
slots with cooldown and readiness state; and the visible song deadline. It
exports raw cues only, never hidden enemy targets or pathfinding state.

## Remaining disadvantages, in priority order

1. **Bomb and trap timing.** Bomb fuse state and stateful trap phase are not
   represented completely even when visible.
2. **Economy and interaction context.** Shop prices, shrine identity/state,
   chest locks, and other visible interaction costs are only partially encoded
   through hashed types.
3. **Multiple objects on one cell.** The fixed grid keeps one actor and one item
   summary per cell. Overlapping visible objects can therefore be lost.
4. **Map extent.** Standard Bard All Zones is represented on a spawn-centred
   65×65 canvas. Live validation must detect clipping before supporting unusual
   level-generation mods or larger custom levels.
5. **Audio-only cues.** Any gameplay-relevant cue that has no symbolic visual
   equivalent must be identified and exported as state rather than raw audio.

## Design rule

Prefer raw, player-visible state over handcrafted conclusions. For example,
export enemy facing and phase rather than `safe_action=LEFT`, and export the
revealed map rather than a prescribed direction to the nearest frontier. This
keeps navigation and tactics learnable while eliminating accidental sensory
handicaps.
