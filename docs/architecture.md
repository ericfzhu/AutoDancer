# Training architecture

AutoDancer trains directly against live Bard workers. Schema 9 and policy
architecture 6 are a deliberate compatibility boundary. Architecture-2
checkpoints can only be used through the explicit partial warm-start path.

## Observation encoding

Each grid cell combines compact embeddings for coarse semantic classes with
shared 4096-entry embeddings for deterministic exact-type hashes. Health,
maximum health, their ratio, enemy timing/status values, and visible attack
cues, visible objects, interaction state, prices, and animation timing are
encoded as embeddings and numeric features. This avoids
the former seven independent 32768-entry tables while preserving distinctions
such as enemy variants and equipment types.

The policy processes that representation in parallel:

- a residual CNN preserves local geometry and produces a global spatial view;
- a separate CNN reads a player-centred 65×65 viewport over persistent
  absolute-coordinate floor memory, including revealed terrain, Bard's
  position, visit counts, and visit recency;
- a two-layer attention encoder reads up to 64 salient actor, item, trap, stair,
  and player tokens with explicit row and column positions;
- a slot-aware attention encoder reads all thirteen HUD inventory slots,
  including quantities, cooldowns, readiness, and toggle state;
- an MLP reads player state including song elapsed/remaining time and audible
  shop-music volume, while a context encoder reads the previous action and
  reward.

These streams fuse to 512 features. A 512-unit LSTM maintains separate hidden
and cell state across partial observations. Independent two-layer actor and
critic heads produce masked logits for 11 actions and the state value. The
persistent map contains no unseen terrain or stale off-screen entities. The
default architecture has 6,478,798 trainable parameters.

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
map, equipment, song, object, price, and animation inputs, the expanded fusion
layer, and critic are initialized afresh. Architectures 4 and 5 also have
explicit partial upgrades for interaction-state and audio additions.

## Architecture 7 experimental compatibility path

Architecture 7 is an implemented but rejected experiment, not the promoted default. It
preserves a complete Architecture-2 actor-critic—including its local encoders,
fusion layer, 512-unit LSTM, actor, and critic—and adds a separate residual
encoder for schema-9-only state:

- tactical, hazard, object, price, and animation grid channels;
- the player-centred 65×65 explicit floor-memory viewport;
- song timing and shop-audio player fields; and
- expanded inventory slots and per-slot state.

The new branch projects to the existing 512-value latent and is multiplied by
`tanh(adapter_gate)` before the original LSTM. The scalar gate initializes to
exactly zero, so a loaded A2 checkpoint is the initial function. The bounded
gate can then learn how much residual information to admit without replacing
the proven latent interface.

The A2-to-A7 warm start copies every source tensor, including the critic, into
the preserved `base` module. It resets only the optimizer and initializes only
the adapter. `architecture7_parity.py` is a mandatory preflight: on the real V2
checkpoint it currently reports zero error for logits, values, next recurrent
state, reset-containing sequences, deterministic actions, and arbitrary
perturbations of all new inputs while the gate is closed.

The paired pilot keeps Reward V2 and the existing action contract fixed. Its
predeclared seeds, diagnostics, and gameplay gates are recorded in
`reward-history.md`; passing static parity does not itself promote A7.

The completed three-seed pilot confirmed exact initialization parity but did
not confirm effective sensory adaptation. Final scalar gates had magnitudes
between `0.000030` and `0.001276`; aggregate progress improved slightly from
1.000 to 1.033, but stair discovery remained zero and unchanged directional
attempts worsened from 76.7% to 91.7%. A2 therefore remains the baseline. A
future compatibility design must preserve zero initial output while allowing
the new branch to receive useful gradients immediately.

## Architecture 8 controlled residual

Architecture 8 is the next candidate, not a promoted default. It retains the
complete A2 actor-critic and replaces A7's scalar gate with a zero-initialized
512-by-512 residual projection after the schema-9 sensory adapter. At
initialization the new branch contributes exactly zero, preserving A2 logits,
values, recurrent state, and deterministic actions. The projection itself
receives gradients immediately; once it opens, gradients reach the adapter.

For the first ten PPO updates the preserved A2 base is frozen. This turns the
first 10,240 transitions into a direct test of whether the new path learns,
instead of allowing ordinary PPO drift in the old path to masquerade as an
architecture improvement. A2 legacy-action and current-action controls use the
same source checkpoint, reward, seed, budget, and checkpoint cadence. The full
predeclared protocol is in `docs/architecture8-controls.md`.
