# Installed-engine inspection and training-port feasibility

## Assessment

Follow-up: the [fixed-action pacing test](engine-pacing-test.md) has now run.
Uncapping and skipping game drawing produced about 1.3x one-worker replay
throughput with substantially higher CPU consumption. See that report for the
measurements; the rest of this document records the preceding file inspection.

The installed game has substantial general-purpose client machinery around its
turn simulation. A training runtime could plausibly remove much of that machinery.
The inspection does **not** establish how much CPU time it consumes or support
a measured 10x or 100x simulator-speed claim. LuaJIT and native components already
exist, so a language rewrite alone is not the compelling opportunity.

The strongest next experiment is an accelerated host using the existing game
rules: explicitly control simulation time, advance turns without waiting on
interactive input/render pacing, and profile the remaining work. The existing
fast-forward and snapshot mechanisms make this worth investigating before a
full reimplementation.

## What was inspected

Read-only inspection of the installed v4.2.1-b5713 configuration, executable and
libraries, base WSP archive, v4.2.1 patch archive, inherited game user configuration,
recent training-worker log, and this repository's launcher and Lua/native bridge.
The base archive has 1,846 entries including 1,276 Lua entries; the patch has 319
entries. These are archive counts, not counts of active runtime handlers.
Core modules contain LuaJIT bytecode with useful names and debug metadata. Selected
functions were inspected through LuaJIT's bytecode introspection API without
executing their game module bodies. Patched module strings were inspected too;
base-only disassemblies are not treated as authoritative for overridden modules.

No installed executable, archive, setting, mod, or training configuration was
changed. No new live profiling/uncapped-speed measurement was performed in this
inspection. Binary strings and handler names establish available machinery, not
its runtime cost. There is no native source-level optimization audit here.

## Findings

| Area | Evidence | Training implication |
| --- | --- | --- |
| Execution runtime | `lua51.dll` identifies LuaJIT 2.1.1748749768; core game scripts are LuaJIT bytecode. Native `necrolevel.dll`, engine, and FFI interfaces are present. | Already beyond a plain interpreted prototype. This does not prove all hot paths are JIT compiled. |
| Graphics | A recent successful training-worker log initializes SDL, a Direct3D 11/BGFX renderer, shaders, and bitmap fonts. | Worker windows are hidden, but graphics initialization still happens. Rendering's steady-state cost remains unmeasured. |
| Existing launcher reductions | Asset auto-reload and automatic DLC enabling are disabled; Galaxy is disabled; Steam is disabled on all but its designated worker; window size is reduced to 320x180. | Some straightforward background reductions are already applied. Disabling automatic DLC enabling does not establish that all content modules are absent. |
| Frame pacing | Base configuration declares 60 FPS. GameWindow contains frame-limit, VSync, and reduce-input-lag settings; native interfaces include frame-limit setters and sleep-time reporting. Inherited user JSON has no explicit FPS/VSync override. | Frame/input pacing is a concrete candidate, but effective runtime FPS, VSync, and sleep fraction were not measured. |
| Action entry | AutoDancer reads commands from `event.tick`, accepts one outstanding action, then inserts it through `GameInput.add` and the client action buffer. | Actions are coupled to the interactive client's scheduling. The bridge does not directly call a standalone simulation step. |
| Tick and turn pipeline | Tick/turn metadata includes networking, music, visual updates, snapshot/checksum work, input, AI, combat, and item/environment rules. | There is removable surrounding work, but stage names do not reveal CPU percentages or whether every stage runs meaningfully in a particular episode. |
| Fast-forward | `necro.client.FastForward` processes a queue; its render hook sets `renderGame=false` while active. Replay seek and server fast-forward modules also exist. | The engine already has machinery for processing state apart from ordinary visible playback. This is not yet an arbitrary-action accelerated training API. |
| Headless recognition | `system.game.Session.isHeadless` tests whether the graphics bridge is absent. Audio/image preload and visual modules reference headless state. | There is an internal concept of a host without graphics. No working desktop headless launch switch was established. |
| Snapshots | Snapshot metadata covers entities, tile/collision/visibility/object maps, and globals. Worker logs actually report snapshot broadcasts around resets. | Reusing internal restoration could avoid replay or restart work. Exact RNG, clock, external-state and Python bookkeeping restoration still needs validation. |
| Profiling facilities | Performance interfaces expose tick time, render time, total frame time, memory categories and LuaJIT profiling; executable symbols include sleep-time reporting. | We should obtain an actual cost breakdown using these hooks before estimating the removable fraction. Availability is not proof of safe mod access. |

The recent log also contains a bootstrap `Array.lua` undeclared-variable error
before later successful initialization and training. Its contribution to startup
cost is unknown; do not interpret it as a measured steady-state bottleneck.

## Costs a port could target

1. Replace one OS process/full game client per environment with compact environment
   state in a shared host, retaining independent RNG and state ownership.
2. Advance turns on demand without sleeping for the next interactive tick or
   rendering frame; provide an explicit simulation clock.
3. Omit graphics assets/rendering, physical audio output, menus, platform services,
   and networking/replay work unnecessary for a single local training world.
4. Exchange observations in reusable numeric buffers instead of building Lua
   observation tables and passing records through the native pipe/Python boundary.
5. Restore a qualified episode state directly, avoiding level initialization and
   guide replay where complete restoration can be established.

These savings overlap. They cannot be estimated independently and multiplied.
The measured full game worker occupies roughly 1.2 GiB RSS in the scaling pilot;
that is a promising target for reduction, not a measurement of the minimum state
size or evidence that a particular number of lightweight environments will fit.

Removing audio/animation requires particular care: current observations contain
music time, song length/end state, shop music information, and animation-derived
state. Removing the output devices is distinct from removing these semantics.
A simulation clock and parity tests must preserve the intended game/observation
contract or explicitly declare a changed task.

## Quantitative limits from current measurements

The latest eight-worker training loop takes 63.390 s: 52.625 s collection and
10.765 s everything else, including 10.687 s PPO. If **all collection work** got
faster while other work stayed fixed:

| Hypothetical collection speedup | Training loop | End-to-end speedup |
| --- | ---: | ---: |
| 2x | 37.08 s | 1.71x |
| 10x | 16.03 s | 3.96x |
| 100x | 11.29 s | 5.61x |
| Instantaneous collection | 10.77 s | 5.89x |

These are optimistic mathematical scenarios, **not port-performance predictions**.
A game-only speedup improves only part of collection; policy inference, observation
handling, and coordination remain. Therefore even a 10x faster game cannot by
itself make this fixed training configuration 10x faster overall. The learner and
inference architecture would also need improvement.

The slowest worker in the batched-warm-up pilot spent 16.838 s on 436 guide game
steps (38.6 ms/call), 9.955 s on 128 learner game steps (77.8 ms/call), and 7.843 s
on nine resets (871 ms/reset). Those are end-to-end worker interface wall times
under eight-worker load, including transport/Python effects; they are not pure
simulation CPU times. A 10x reduction in those step paths would require roughly
3.9–7.8 ms per call under comparable conditions. Static files cannot establish
whether that target is achievable.

## Recommended feasibility gate

Use a bounded, no-policy replay of qualified actions to remove inference from the
measurement. Compare default pacing with a verified raised/disabled frame limit,
then an accelerated path with render work skipped. Measure effective tick rate,
sleep, simulation, rendering, observation serialization, reset, and resident memory.
Use isolated worker profiles and preserve the installed configuration.

Require parity of game state, RNG-dependent outcomes, rewards, termination, and
policy inputs on the same qualified traces. Validate a deterministic clock rather
than silently dropping time-dependent observation fields. If a lightweight host
reaches the desired throughput while passing parity, a full rules port may be
unnecessary. If it does not, a narrowly scoped Death Metal simulator is a cheaper
next feasibility test than an all-game rewrite, but its success would not establish
fidelity for other enemies, items, level generation, characters, or DLC.

Local evidence is under `runs/performance-project/engine-inspection/`, including
file hashes in `assessment.json`, module inventories/constants, and selected
bytecode listings. The training/scaling numbers come from the earlier measured
pilots, not from a new profiler run.
