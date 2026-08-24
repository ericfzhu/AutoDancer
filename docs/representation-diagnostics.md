# Representation diagnostics

Broad gameplay evaluation cannot establish whether a policy actually uses a
particular observation source. AutoDancer therefore tests representation paths
before promoting another architecture pilot.

## Measurements

`autodancer-representation` loads one or more checkpoints and measures ten
observation groups independently:

- established A2 inputs: local terrain, actors, local items/traps, player,
  inventory, and recurrent action/reward context;
- schema-9 additions: tactical grid fields, persistent map memory, extended
  player fields, and extended inventory/equipment fields.

For each group, the tool performs a controlled single-group counterfactual and
measures changes in policy logits, critic value, recurrent state, and argmax
actions. It separately backpropagates an output objective and measures the
gradient norm at that group's dedicated encoder parameters. Action masks are
excluded because they constrain legal outputs directly rather than entering the
learned representation.

The generated inputs stay within decoder-supported numeric domains, but they
are deliberately synthetic and need not represent a naturally co-occurring
game state. The test establishes wiring, gradient reach, and relative influence;
it does not claim that a feature improves gameplay.

## Materiality rule

Numerically nonzero is not sufficient. A group is `material` only when both its
counterfactual sensitivity and encoder gradient norm reach at least 1% of the
median established-input group in the same checkpoint. A supported path below
that threshold is `trace`. Other statuses are `inactive`, `gradient_only`,
`sensitivity_only`, or `unsupported`.

The 1% threshold is a screening gate, not an assertion that every useful input
must always change the selected action. Candidates failing it should first pass
a smaller representation-learning test instead of receiving a long live pilot.
It was introduced after the A7 gameplay result, so its A7 use is descriptive;
it is predeclared only for future architecture candidates.

## Command

```powershell
uv run autodancer-representation `
  ".\runs\reward-v2-250k\final.pt" `
  ".\runs\architecture7-v2-pilot\training\seed-35001\final.pt" `
  ".\runs\architecture7-v2-pilot\training\seed-35002\final.pt" `
  ".\runs\architecture7-v2-pilot\training\seed-35003\final.pt" `
  --output ".\runs\architecture7-v2-pilot\representation.json"
```

PPO also records one pre-clipping gradient snapshot per update as
`gradient_<group>`. This adds only one parameter scan per update rather than one
scan per minibatch.

## Measured A7 result

The existing A2 checkpoint showed material sensitivity and gradient reach for
all six inputs it supports. As designed, it ignored all four schema-9 additions.

| A7 checkpoint | Adapter gate | Largest new/base sensitivity ratio | Largest new/base gradient ratio | Material new groups |
| --- | ---: | ---: | ---: | ---: |
| Seed 35001 | `-1.2760e-3` | `4.8768e-4` | `4.3876e-2` | `0 / 4` |
| Seed 35002 | `+2.6548e-4` | `1.1792e-4` | `6.7195e-3` | `0 / 4` |
| Seed 35003 | `-3.0464e-5` | `1.9537e-5` | `2.9763e-4` | `0 / 4` |

Every A7-only path was numerically active after training, but none was
material. None of the controlled new-input perturbations changed an argmax
action. This confirms the earlier interpretation: A7 mostly fine-tuned its A2
base while the new representation remained too weak to test whether richer
observations improve gameplay.
