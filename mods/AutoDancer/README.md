# AutoDancer SYNCHRONY mod

This unpackaged mod is the only control and telemetry interface used by Python.
It is Bard-only.

Install this complete directory as:

```text
%LOCALAPPDATA%\NecroDancer\mods\AutoDancer
```

Enable **AutoDancer** under **Customize → Mods**, then start or attach to a Bard
run. Python writes `bridge-command.txt` beside `mod.json`; the Lua bridge polls
that file, injects one action, and emits the resulting symbolic transition to
`NecroDancer.log`.

The mod never accepts an uncorrelated transition as an agent step. Every
agent-driven turn contains the exact Python `session_id`, monotonic
`command_id`, logical action, engine action, and observed input action.

There are no probe hotkeys, keyboard automation, screenshots, or simulator
comparison paths.
