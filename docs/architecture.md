# Training architecture

AutoDancer trains directly against live Bard workers. Schema 5 and policy
architecture 2 are a deliberate compatibility boundary: schema-4 observations
and older checkpoints are rejected rather than silently reinterpreted.

## Observation encoding

Each grid cell combines compact embeddings for coarse semantic classes with
shared 4096-entry embeddings for deterministic exact-type hashes. Health,
maximum health, and their ratio are encoded as numeric features. This avoids
the former seven independent 32768-entry tables while preserving distinctions
such as enemy variants and equipment types.

The policy processes that representation in parallel:

- a residual CNN preserves local geometry and produces a global spatial view;
- a two-layer attention encoder reads up to 64 salient actor, item, trap, stair,
  and player tokens with explicit row and column positions;
- a slot-aware attention encoder reads the eight inventory slots;
- an MLP reads player state, while a context encoder reads the previous action
  and reward.

These streams fuse to 512 features. A 512-unit LSTM maintains separate hidden
and cell state across partial observations. Independent two-layer actor and
critic heads produce masked logits for 11 actions and the state value. The
default model has 5,953,167 trainable parameters.

## Recurrent PPO contract

Rollouts store the exact LSTM state at every transition, including both hidden
and cell tensors, plus previous-action and previous-reward inputs. Episode
boundaries reset all recurrent context. PPO trains 32-step recurrent chunks
from the stored initial state and keeps the existing clipped objective, GAE,
entropy bonus, gradient clipping, and synchronous fixed-capacity workers.

Checkpoints include the architecture version and complete model configuration.
Resume requires an exact architecture and PPO configuration match, preventing
accidental loading of weights against different observation semantics.
