# EXP-0020 pre-execution rejection

EXP-0020 was rejected before calibration, training, or evaluation consumed any experiment seed.

The declared `DeathMetalGuideV1` reward described itself as a boss guide but the implementation credited `enemy_damage` and `enemy_kill` for every hostile entity. The live event already carries authoritative `data.boss` and `data.boss_add` roles, but V1 did not inspect them.

This is material rather than hypothetical. In the selected parent policy's held-out assisted evaluation, one 12-episode wave recorded 50 enemy-damage points and 50 kills, while only 10 damage points and 10 kills were attributed to the boss. Thus 80% of those combat events were not boss progress. At V1's weights, incidental enemies could consume the full `+5` floor-level combat budget without advancing Death Metal's phase objective.

That proxy conflicts with EXP-0020's question and primary phase-acquisition outcome. Running it would not distinguish boss learning from generic combat farming. The experiment is therefore statically invalid and must not be repaired in place after immutable registration.

The successor must:

- use a new experiment ID and reward component version;
- restrict guide combat credit to events whose authoritative telemetry has `data.boss == true`;
- keep generic enemies and boss adds available as actions needed for survival, but give them no direct proxy reward;
- retain boss phase depth and real Zone 2 entry—not shaped return—as selection criteria;
- preserve the exact legal, player-health-only controller and curriculum requirements.

Evidence: `runs/assisted-death-metal/evaluation/seed-68002/stochastic-98001/report.json`, whose aggregate reports `enemy_damage=50`, `enemy_kills=50`, `boss_damage=10`, and `boss_kills=10`.
