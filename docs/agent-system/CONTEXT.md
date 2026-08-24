# AutoDancer agent context

This glossary is the small amount of repository context assumed by the atlas.

- **A2–A7** — policy architecture versions. A2–A6 are implemented historical
  contracts. A7 is proposed and must not be described as current behavior.
- **Bard** — the only supported training character. Bard advances the world
  once per accepted action and does not require beat-timed input.
- **All Zones** — the normal full-run game mode used for training and
  evaluation; Daily Challenge is excluded.
- **Worker** — one supervisor-owned, isolated NecroDancer process with its own
  profile, pipe, seed stream, episode, and recurrent state.
- **Transition** — one acknowledged command plus the authoritative resulting
  observation, events, reward inputs, identities, and terminal state.
- **Schema** — the versioned Lua-to-Python observation and protocol contract.
- **Spatial memory** — the explicit player-visible 65×65 floor map maintained
  by `FloorMapMemory`.
- **Temporal memory** — the learned 512-unit LSTM hidden and cell state.
- **Task reward** — floor, zone, victory, death, and abort signals tied directly
  to episode outcomes.
- **Shaping reward** — bounded intermediate learning signals such as new
  positions, combat credit, items, or stair potential.
- **Policy version** — one frozen set of weights used to collect a complete
  rollout before the next PPO update.
- **Retained baseline** — Reward V2 with Architecture A2, currently the best
  supported policy according to held-out gameplay gates.
