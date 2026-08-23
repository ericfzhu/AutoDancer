# Player-information parity

The policy should receive the gameplay facts available to an attentive human
player, while never reading unrevealed engine state. This is distinct from
reward shaping: information makes a decision possible; reward says which
outcomes matter.

## Current parity

Schema 6 adds a persistent 65×65 current-floor memory anchored at Bard's spawn.
It combines the game's revealed terrain with Bard's position, visit count, and
visit recency. Dynamic enemies and items disappear when they leave sight, so
the map cannot become a hidden-state oracle. A periodic Lua snapshot also
captures distant terrain revealed by the Map item. The local 21×21 observation
remains the authoritative combat view.

This approximates what the HUD minimap and ordinary human spatial memory make
available. The reward tracker still owns its independent exact novelty sets;
those sets prevent reward farming but are not exposed directly.

## Remaining disadvantages, in priority order

1. **Enemy state and intent.** The grid carries type and health but not every
   visible facing direction, movement phase, shield orientation, charge state,
   freeze/confusion state, or attack telegraph. These are visible and often
   determine the only safe Bard action.
2. **Incomplete equipment state.** The eight encoded slots omit some visible
   equipment categories such as armour, head, feet, and torch. Exact identity,
   durability, charges, cooldowns, and activation readiness should be explicit
   for every occupied HUD slot.
3. **Song and floor deadline.** A player can see and hear song progress and
   knows how much time remains before forced descent. Sequence number is not a
   reliable substitute because song length and tempo vary.
4. **Bomb and trap timing.** Bomb fuse state and stateful trap phase are not
   represented completely even when visible.
5. **Economy and interaction context.** Shop prices, shrine identity/state,
   chest locks, and other visible interaction costs are only partially encoded
   through hashed types.
6. **Multiple objects on one cell.** The fixed grid keeps one actor and one item
   summary per cell. Overlapping visible objects can therefore be lost.
7. **Map extent.** Standard Bard All Zones is represented on a spawn-centred
   65×65 canvas. Live validation must detect clipping before supporting unusual
   level-generation mods or larger custom levels.
8. **Audio-only cues.** Any gameplay-relevant cue that has no symbolic visual
   equivalent must be identified and exported as state rather than raw audio.

## Design rule

Prefer raw, player-visible state over handcrafted conclusions. For example,
export enemy facing and phase rather than `safe_action=LEFT`, and export the
revealed map rather than a prescribed direction to the nearest frontier. This
keeps navigation and tactics learnable while eliminating accidental sensory
handicaps.
