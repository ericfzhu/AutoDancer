# AutoDancer SYNCHRONY mod

This is an unpackaged local mod. It prints one schema-1 telemetry record to the
normal game debug log after each turn.

1. Copy the complete `AutoDancer` directory, including `mod.json`, into the
   SYNCHRONY unpackaged-mod directory.
2. Set `GAME_VERSION` and `STEAM_BUILD` in `scripts/AutoDancer.lua`.
3. Open **Customize → Mods** and enable AutoDancer.
4. Press Shift+F7 after a script change.
5. Keep the game focused during a live run.

The initial exporter is deliberately narrow. It exports the player centre tile,
player position, health, level identity, and safe actions. Other visible tiles,
entities, inventory values, raw events, and special action masks must be added
with matching traces. Unknown values stay zero. This prevents hidden game data
from reaching a policy.

Python sends F8 as the default restart key. Bind F8 to a controlled restart in
the game, or configure another key in the platform action sender. On macOS, give
the Python host Accessibility and Screen Recording permission before a live run.

The mod does not open files, sockets, or IPC channels. It contains no game asset
and no copied game source.

