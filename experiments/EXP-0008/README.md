# EXP-0008: Architecture 8 training-horizon test

EXP-0007 established that A8 integrates and uses the rich schema-9 observations,
but it did not improve held-out progression within 30,720 transitions. This
experiment tests the narrower explanation that A8 needs a longer adaptation
horizon.

Both trainable arms resume their exact EXP-0007 model, critic, optimizer, update,
and RNG states and continue to 250,880 transitions. Reward V2, the controller,
the observations, PPO, action contract, seed, capacity, and evaluation protocol
remain fixed. Passing advances A8 to a separate multiseed confirmation; it does
not promote A8 directly.

Runtime evidence is written to `runs/architecture8-horizon` and MLflow. The
recoverable launcher is `tools/run-architecture8-horizon.ps1`.
