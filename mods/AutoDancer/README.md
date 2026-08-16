# AutoDancer SYNCHRONY mod

This is an unpackaged local mod. It prints one schema-2 telemetry record to the
normal game debug log after each turn and an explicit terminal record when a run
ends.

1. Copy the complete `AutoDancer` directory, including `mod.json`, into the
   SYNCHRONY unpackaged-mod directory.
2. Set `GAME_VERSION` and `STEAM_BUILD` in `scripts/AutoDancer.lua`.
3. Open **Customize → Mods** and enable AutoDancer.
4. Press Shift+F7 after a script change.
5. Keep the game focused during a live run and avoid manual input.
6. Bind **F8** to a controlled restart for the default Windows adapter.

The exporter includes revealed terrain, visible entities, safe inventory fields,
raw events, action masks, a stable run ID, and explicit episode status. Unknown
or unconfirmed values stay zero. The policy receives only visible dynamic data;
task identity is supplied explicitly by Python.

The default Windows adapter supports movement and Bard multi-key actions. The
macOS default mapping remains movement-only and is retained mainly for legacy
development use.

The mod does not open files, sockets, or IPC channels. It contains no game
assets and no copied game source.
