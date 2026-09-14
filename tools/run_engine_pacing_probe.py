"""Fixed-action engine pacing experiment using isolated, disposable worker mods."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

from autodancer.live.supervisor import AutoDancerSupervisor, SupervisorConfig
from autodancer.training.demonstrations import normalized_observation_digest
from autodancer.training.trace_prefix import QualifiedTracePrefixBank

PROBE = r"""
local Window = require "system.game.Window"
local Performance = require "system.debug.Performance"
local Timer = require "system.utils.Timer"
local configured = false
local started = nil
local ticks = 0
local tickTime = 0
local renderTime = 0
local frameTime = 0
local gameRenders = 0
local skips = 0
event.render.add("countAutoDancerProbeGameRenders", "overlay", function()
    gameRenders = gameRenders + 1
end)
event.tick.add("measureAutoDancerEnginePacing", {order="input", sequence=-150}, function()
    if not configured then
        if MODE ~= "default" then
            Window.setVSync(false)
            Window.setFramerateLimit(0)
        end
        configured = true
        local stats = Performance.getRenderPerformance()
        if type(stats) == "table" then
            for key, value in pairs(stats) do
                if type(value) == "number" then
                    print("AUTODANCER_RENDER_STAT:" .. tostring(key) .. "=" .. tostring(value))
                end
            end
        end
    end
    local now = Timer.getGlobalTime()
    started = started or now
    ticks = ticks + 1
    tickTime = tickTime + Performance.getTickTime()
    renderTime = renderTime + Performance.getRenderTime()
    frameTime = frameTime + Performance.getTotalFrameTime()
    if now - started >= 1 then
        print("AUTODANCER_PACING:" .. string.format(
            '{"time":%.6f,"ticks":%d,"elapsed":%.6f,"limit":%.6f,"fps":%.6f,"tick_raw_mean":%.9f,"render_raw_mean":%.9f,"frame_raw_mean":%.9f,"game_renders":%d,"skips":%d}',
            now, ticks, now-started, Window.getFramerateLimit(), Performance.getFramerate(),
            tickTime/ticks, renderTime/ticks, frameTime/ticks, gameRenders, skips))
        started = now
        ticks = 0
        tickTime = 0
        renderTime = 0
        frameTime = 0
        gameRenders = 0
        skips = 0
    end
end)
if MODE == "uncapped-no-game-render" then
    event.renderUI.add("skipAutoDancerProbeGameRender",
        {order="fastForward", sequence=1000}, function(ev)
        ev.renderGame = false
        skips = skips + 1
    end)
end
"""


class ProbeSupervisor(AutoDancerSupervisor):
    mode: str = "default"

    def _validate_installation(self):
        # Do not synchronize the experimental probe into the shared installed mod.
        for path in (
            self.config.executable,
            self.config.game_dir / "autodancer_native.dll",
            self.config.mod_dir / "scripts/Bridge.lua",
        ):
            if not path.is_file():
                raise FileNotFoundError(path)


def observation_signature(obs):
    return {
        "normalized": normalized_observation_digest(obs),
        "fields": {
            key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for key, value in obs.items()
        },
        "player": obs["player"].tolist(),
    }


def trial(args, mode, bank):
    target = args.output / mode
    target.mkdir()
    host = args.output / "game-host"
    mod_root = host / "probe-mods"
    mod = mod_root / "AutoDancer"
    if not host.exists():
        host.mkdir()
        for path in args.game_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
                shutil.copy2(path, host / path.name)
        for name in ("versions", "dlc"):
            shutil.copytree(args.game_dir / name, host / name)
        shutil.copytree(Path("mods/AutoDancer"), mod)
    config = json.loads((args.game_dir / "config.json").read_text())
    game = config["wos"]["game"]
    game["assets"]["external"]["path"] = args.game_dir.parent.resolve().as_posix()
    game["mods"]["scriptWhitelist"] += ["system.game.Window", "system.debug.Performance"]
    game["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(config))
    probe = mod / "scripts/EnginePacingProbe.lua"
    probe.write_text(PROBE.replace("MODE", json.dumps(mode)))
    entry = mod / "scripts/AutoDancer.lua"
    entry.write_text(
        'require "AutoDancer.scripts.EnginePacingProbe"\n'
        + Path("mods/AutoDancer/scripts/AutoDancer.lua").read_text()
    )
    # Preserve the 60-Hz watchdog's intended wall-time limits at any frame rate.
    # This affects only the copied bridge, identically in every measured mode.
    bridge = Path("mods/AutoDancer/scripts/Bridge.lua").read_text()
    bridge = 'local ProbeTimer = require "system.utils.Timer"\n' + bridge
    bridge = bridge.replace(
        "received_tick = tickCount,",
        "received_tick = tickCount, received_time = ProbeTimer.getGlobalTime(),",
    )
    bridge = bridge.replace(
        "accepted_tick = tickCount,",
        "accepted_tick = tickCount, accepted_time = ProbeTimer.getGlobalTime(),",
    )
    bridge = bridge.replace(
        "tickCount - queuedCommand.received_tick >= RESET_WATCHDOG_TICKS",
        "ProbeTimer.getGlobalTime() - queuedCommand.received_time >= RESET_WATCHDOG_TICKS / 60",
    )
    for state in ("pending", "completed"):
        bridge = bridge.replace(
            f"tickCount - {state}.accepted_tick >= ACTION_WATCHDOG_TICKS",
            f"ProbeTimer.getGlobalTime() - {state}.accepted_time >= ACTION_WATCHDOG_TICKS / 60",
        )
    (mod / "scripts/Bridge.lua").write_text(bridge)
    supervisor = ProbeSupervisor(
        SupervisorConfig(
            game_dir=host,
            mod_dir=mod,
            startup_timeout=20,
            num_instances=1,
            affinity_policy="none",
            steam_presence_worker=0,
            curriculum_commands_enabled=True,
            diagnostic_root=target / "diagnostics",
            profile_root=target / "profiles",
        )
    )
    supervisor.mode = mode
    started = time.perf_counter()
    result = {"mode": mode, "episodes": []}
    try:
        with supervisor:
            result["startup_seconds"] = time.perf_counter() - started
            worker = supervisor.environment(supervisor.worker_ids[0])
            process = psutil.Process(supervisor.workers[supervisor.worker_ids[0]].pid)
            try:
                for repetition in range(args.repeats):
                    for trace in bank.traces:
                        t0 = time.perf_counter()
                        obs, info = worker.reset(
                            seed=trace.seed, options=trace.reset_spec.reset_options()
                        )
                        reset_seconds = time.perf_counter() - t0
                        episode = {
                            "repetition": repetition,
                            "seed": trace.seed,
                            "reset_seconds": reset_seconds,
                            "steps": [],
                            "reset_signature": observation_signature(obs),
                        }
                        result["episodes"].append(episode)
                        cpu0 = sum(process.cpu_times()[:2])
                        replay_started = time.perf_counter()
                        for index, action in enumerate(trace.actions):
                            if not obs["action_mask"][action]:
                                episode["error"] = f"action masked at {index}"
                                break
                            step_started = time.perf_counter()
                            obs, reward, terminated, truncated, info = worker.step(action)
                            latency = time.perf_counter() - step_started
                            signature = observation_signature(obs)
                            episode["steps"].append(
                                {
                                    "seconds": latency,
                                    "action": action,
                                    "reward": reward,
                                    "terminated": terminated,
                                    "truncated": truncated,
                                    "status": info.get("episode_status"),
                                    "qualified_digest_match": signature["normalized"]
                                    == trace.turn_digests[index + 1],
                                    "signature": signature,
                                }
                            )
                            if terminated or truncated:
                                break
                        episode["replay_seconds"] = time.perf_counter() - replay_started
                        episode["worker_cpu_seconds"] = sum(process.cpu_times()[:2]) - cpu0
                        episode["worker_rss_bytes"] = process.memory_info().rss
                        episode["expected_steps"] = len(trace.actions)
                        episode["qualified_reset_match"] = (
                            episode["reset_signature"]["normalized"] == trace.turn_digests[0]
                        )
                        print(
                            json.dumps(
                                {
                                    "mode": mode,
                                    "seed": trace.seed,
                                    "repetition": repetition,
                                    "steps": len(episode["steps"]),
                                    "seconds": episode["replay_seconds"],
                                }
                            ),
                            flush=True,
                        )
            finally:
                worker.close()
            handle = supervisor.workers[supervisor.worker_ids[0]]
            result["restarts"] = handle.restart_count
            # Logging flushes every 50 ms; read after replay has completed.
            time.sleep(0.1)
            log = handle.log_path.read_text(encoding="utf-8", errors="replace")
            (target / "game.log").write_text(log, encoding="utf-8")
            result["probe_samples"] = [
                json.JSONDecoder().raw_decode(line.split("AUTODANCER_PACING:", 1)[1])[0]
                for line in log.splitlines()
                if "AUTODANCER_PACING:{" in line
            ]
            if not result["probe_samples"]:
                raise RuntimeError("Probe produced no runtime measurements; mode not verified")
            renders = sum(s["game_renders"] for s in result["probe_samples"])
            skips = sum(s["skips"] for s in result["probe_samples"])
            if mode == "uncapped-no-game-render" and (renders or not skips):
                raise RuntimeError("Game-render suppression was not verified")
            if mode != "uncapped-no-game-render" and not renders:
                raise RuntimeError("Baseline game-render counter did not observe rendering")
            if mode != "default" and any(s["limit"] != 0 for s in result["probe_samples"]):
                raise RuntimeError("Uncapped mode did not retain frame limit zero")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=Path(r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("default", "uncapped", "uncapped-no-game-render"),
        default=["default", "uncapped", "uncapped-no-game-render"],
    )
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("fresh output directory and positive repeats required")
    args.output.mkdir(parents=True)
    bank = QualifiedTracePrefixBank.load(
        "runs/qualified-death-metal-trace-search/demonstration-bank.json",
        "runs/qualified-death-metal-trace-search/qualification.json",
        tail_actions=1,
    )
    results = []
    for mode in args.modes:
        result = trial(args, mode, bank)
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if "error" in result:
            print(result["error"], flush=True)
            return 1
        latencies = [s["seconds"] for e in result["episodes"] for s in e["steps"]]
        print(
            json.dumps(
                {
                    "mode": mode,
                    "step_calls_per_second": len(latencies) / sum(latencies),
                    "median_step_ms": statistics.median(latencies) * 1000,
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
