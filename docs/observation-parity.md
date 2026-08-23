# Player-information parity

The policy should receive the gameplay facts available to an attentive human
player, while never reading unrevealed engine state. This is distinct from
reward shaping: information makes a decision possible; reward says which
outcomes matter.

## Current parity

Schema 9 provides persistent current-floor memory through a 65×65 viewport
centred on Bard. Absolute-coordinate terrain and traversal history are retained
for the whole floor, so remembered areas leave and re-enter the viewport as Bard
moves. It combines the game's revealed terrain with Bard's position, visit count,
and visit recency. Dynamic enemies and items disappear when they leave sight, so
the map cannot become a hidden-state oracle. A periodic Lua snapshot also
captures distant terrain revealed by the Map item. The local 21×21 observation
remains the authoritative combat view.

This approximates what the HUD minimap and ordinary human spatial memory make
available. The reward tracker still owns its independent exact novelty sets;
those sets prevent reward farming but are not exposed directly.

Schema 9 retains the three highest-priority additions from Schema 7:
visible enemy facing/timing/status/attack cues; all thirteen HUD equipment
slots with cooldown and readiness state; and the visible song deadline. It
exports raw cues only, never hidden enemy targets or pathfinding state.

It also encodes visible explosives and trap/tell animation timing, plus chest,
shrine, shopkeeper, sale, lock, shoplifting, currency-price, and health-price
state. Affordability is not precomputed: the policy can compare displayed cost
with its visible health and gold. Hidden container and shrine contents remain
excluded.

Every record now carries the engine's absolute level bounds. Python verifies
that the full floor dimensions fit the fixed viewport capacity and stops with a
non-recoverable capacity error if they do not. Distance from the initial spawn
does not count as extra map size. Training can no longer continue with silently
clipped map memory.

The audio audit found one material non-visual cue: shopkeeper singing can be
heard before the shopkeeper is visible. The policy receives the engine-computed
effective shop music-layer volume, allowing the same warmer/colder search as a
player without revealing direction or position. Transient sounds are not
exported because the pre-play event does not expose final listener attenuation;
turning their source coordinates into observations could reveal sounds the
player did not actually hear.

## Remaining disadvantages, in priority order

1. **Multiple objects on one cell.** The fixed grid keeps one actor, one item,
   and one prioritized object summary per cell. Overlapping visible objects can
   therefore be lost.
2. **Unsupported custom map extent.** Standard Bard All Zones is represented by
   a player-centred 65×65 viewport over persistent coordinate history. Larger
   custom levels fail clearly and require an architecture change rather than
   receiving a clipped observation.
3. **Transient positional audio.** Enemy and effect sounds are omitted unless a
   future post-attenuation hook can prove that the local player heard them.

## Design rule

Prefer raw, player-visible state over handcrafted conclusions. For example,
export enemy facing and phase rather than `safe_action=LEFT`, and export the
revealed map rather than a prescribed direction to the nearest frontier. This
keeps navigation and tactics learnable while eliminating accidental sensory
handicaps.
