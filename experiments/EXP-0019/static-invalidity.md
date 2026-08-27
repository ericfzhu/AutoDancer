# EXP-0019 pre-execution invalidity finding

EXP-0019 was stopped before calibration or optimization because static inspection
of the supported live game build disproved a defining curriculum assumption.

## Pinned evidence

- Installed `NecroDancer.wsp` SHA-256:
  `52f7ddb310749c44c1f9740bd9bf1318cc137abed43ab3a4de2077f83d33e831`
- Decompressed
  `scripts/necro/game/data/enemy/bosses/DeathMetal.lua` bytecode SHA-256:
  `0733006bca14adcebb0cdff7ce5cec3d6d4e5c1c827160832f5eef6f587437b`
- Supported game identity: NecroDancer `v4.2.1-b5713`, Steam build
  `22938426`.

The repository's read-only WSP16 reader extracted the pinned module, and the
existing LuaJIT bytecode decompiler exposed these transition semantics:

1. Death Metal's default maximum health is 9. Its phase thresholds are 6, 4,
   and 2 health.
2. The phase variants have different entity types, AI, spells, cooldowns,
   beat-delay behavior, shield state, teleport behavior, and summons.
3. Phase selection and `object.convert(...)` occur in the
   `objectTakeDamage/deathMetalHitTriggers` handler after legal damage.
4. `boss1hp-player*` instead writes `entity.health.health = 1` directly after
   level load. It neither emits the damage transition nor converts the entity.
   It also applies the write to `bossAdd` entities.

The resulting state is therefore a phase-1 entity at phase-4 health, with phase
conversion side effects omitted. It is outside the ordinary game transition
distribution. A successful episode against that state cannot establish a
late-fight Death Metal subskill or justify health-only assistance reduction.

## Decision consequence

- The queued frozen player-8 calibration and conditional EXP-0019 optimization
  watchers were stopped before they launched a game worker.
- EXP-0019 is rejected pre-execution; no calibration, training, or evaluation
  seed was consumed.
- EXP-0016 through EXP-0018 remain controller-valid evidence for the explicitly
  mutated profile only. Their Zone 2 entries are not reachable-state boss
  competence and cannot support normal-start promotion.
- The active schema-10 controller qualification remains valid because its final
  natural soak does not use the boss-health profile.

The replacement curriculum must use a legal natural-prefix handoff: a guide
acts through ordinary engine transitions, the learner takes over at a declared
observed phase boundary, guide transitions are excluded from PPO reward/return,
and the handoff preserves the complete live state. Only after that mechanism is
qualified may boss assistance reduction resume.
