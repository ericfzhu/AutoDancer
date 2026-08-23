# AutoDancer SYNCHRONY mod

This unpackaged Bard-only mod is Python's sole control and telemetry interface.
Install this complete directory at:

```text
%LOCALAPPDATA%\NecroDancer\mods\AutoDancer
```

Enable **AutoDancer** under **Customize → Mods**. Python starts a hidden
coordinator, which reads `bridge-command.coordinator.txt` and creates native
independent workers. Each worker reads its own predictable file, such as
`bridge-command.worker-0000.txt`, and emits schema-7 symbolic transitions in
its engine-assigned log.

The Python supervisor precreates all worker command files before launch. Every
transition identifies its worker and acknowledges the exact session, command,
action or reset seed. There are no probe hotkeys, keyboard controls,
screenshots, or simulator paths.
