# Training architecture

AutoDancer trains directly against live Bard workers. Schema 7 and policy
architecture 4 are a deliberate compatibility boundary. Architecture-2
checkpoints can only be used through the explicit partial warm-start path.

## Observation encoding

Each grid cell combines compact embeddings for coarse semantic classes with
shared 4096-entry embeddings for deterministic exact-type hashes. Health,
maximum health, their ratio, enemy timing/status values, and visible attack
cues are encoded as embeddings and numeric features. This avoids
the former seven independent 32768-entry tables while preserving distinctions
such as enemy variants and equipment types.

The policy processes that representation in parallel:

- a residual CNN preserves local geometry and produces a global spatial view;
- a separate CNN reads a persistent 65×65 floor map anchored at the spawn,
  including revealed terrain, Bard's position, visit counts, and visit recency;
- a two-layer attention encoder reads up to 64 salient actor, item, trap, stair,
  and player tokens with explicit row and column positions;
- a slot-aware attention encoder reads all thirteen HUD inventory slots,
  including quantities, cooldowns, readiness, and toggle state;
- an MLP reads player state including song elapsed/remaining time, while a
  context encoder reads the previous action and reward.

These streams fuse to 512 features. A 512-unit LSTM maintains separate hidden
and cell state across partial observations. Independent two-layer actor and
critic heads produce masked logits for 11 actions and the state value. The
persistent map contains no unseen terrain or stale off-screen entities. The
default architecture has 6,401,258 trainable parameters.

## Recurrent PPO contract

Rollouts store the exact LSTM state at every transition, including both hidden
and cell tensors, plus previous-action and previous-reward inputs. Episode
boundaries reset all recurrent context. PPO trains 32-step recurrent chunks
from the stored initial state and keeps the existing clipped objective, GAE,
entropy bonus, gradient clipping, and synchronous fixed-capacity workers.

Checkpoints include the architecture version and complete model configuration.
Resume requires an exact architecture and PPO configuration match. A deliberate
architecture-2 warm start transfers compatible portions of the old local,
player, and inventory encoders plus recurrent and actor weights. New tactical,
map, equipment, and song inputs, the expanded fusion layer, and critic are
initialized afresh.
