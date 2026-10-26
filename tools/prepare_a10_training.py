"""Bounded live A10 learning preflight and an unstarted, pinned experiment recipe.

Uses the production training/evaluation entry points with an isolated game host.
The smoke checkpoint is diagnostic only; the generated experiment starts fresh.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import psutil
import torch
from probe_throw_semantics import PrivateSupervisor

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.rewards import load_reward_config
from autodancer.training import baseline, train
from autodancer.training.model import WorldActorCritic, model_from_spec
from autodancer.training.ppo import PPOConfig, RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]


def digest(state):
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = dict(
            peak_python_rss=0,
            peak_children_rss=0,
            minimum_available_ram=psutil.virtual_memory().available,
        )
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        process = psutil.Process()
        while not self.stop.is_set():
            children = 0
            for child in process.children(recursive=True):
                try:
                    children += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples["peak_python_rss"] = max(
                self.samples["peak_python_rss"], process.memory_info().rss
            )
            self.samples["peak_children_rss"] = max(self.samples["peak_children_rss"], children)
            self.samples["minimum_available_ram"] = min(
                self.samples["minimum_available_ram"], psutil.virtual_memory().available
            )
            self.stop.wait(0.2)


def private_host(game, output):
    host = output / "game-host"
    host.mkdir()
    for path in game.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(path, host / path.name)
    for name in ("versions", "dlc"):
        shutil.copytree(game / name, host / name)
    mod = host / "probe-mods/AutoDancer"
    shutil.copytree(ROOT / "mods/AutoDancer", mod)
    config = json.loads((game / "config.json").read_text())
    config["wos"]["game"]["assets"]["external"]["path"] = game.parent.as_posix()
    config["wos"]["game"]["mods"]["loadPaths"][0] = dict(
        location="WORKING_DIRECTORY", name="unpackaged", package=False, path="probe-mods"
    )
    atomic_json(host / "config.json", config)
    return host, mod


def train_args(host, mod, reward, output, steps, seed_pool, *, max_turns):
    return [
        "--game-dir",
        str(host),
        "--mod-dir",
        str(mod),
        "--run-dir",
        str(output),
        "--architecture",
        "10",
        "--equipment-profile",
        "internal-v1",
        "--equipment-capacity",
        "64",
        "--world-capacity",
        "256",
        "--action-contract",
        "current",
        "--num-instances",
        "2",
        "--device",
        "cuda",
        "--seed",
        "27",
        "--total-steps",
        str(steps),
        "--training-seed-pool",
        seed_pool,
        "--curriculum-target-level",
        "2",
        "--curriculum-profile",
        "normal",
        "--reward-config",
        str(reward),
        "--gamma",
        "0.99",
        "--learning-rate",
        "0.0003",
        "--rollout-length",
        "128",
        "--sequence-length",
        "32",
        "--max-turns",
        str(max_turns),
        "--checkpoint-interval",
        "256",
        "--evaluation-interval",
        "0",
        "--affinity",
        "none",
        "--steam-presence-worker",
        "0",
    ]


def evaluation_args(host, mod, reward, checkpoint, output, seeds, max_steps):
    return [
        "--game-dir",
        str(host),
        "--mod-dir",
        str(mod),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--num-instances",
        "2",
        "--device",
        "cuda",
        "--seeds",
        ",".join(map(str, seeds)),
        "--max-steps",
        str(max_steps),
        "--policy-mode",
        "stochastic",
        "--policy-seed",
        "2701",
        "--action-contract",
        "current",
        "--curriculum-target-level",
        "2",
        "--reward-config",
        str(reward),
        "--affinity",
        "none",
        "--steam-presence-worker",
        "0",
    ]


def write_launcher(output, recipe):
    def quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location -LiteralPath {quote(ROOT)}",
        "$env:OMP_NUM_THREADS = '4'",
        "$env:MKL_NUM_THREADS = '4'",
        f"$python = {quote(ROOT / '.venv/Scripts/python.exe')}",
        # The generated recipe is never executed by this preflight.
        "# Fresh experiment: deliberately excludes the smoke checkpoint.",
    ]
    for module, arguments in recipe:
        lines.append("& $python -m " + module + " " + " ".join(quote(a) for a in arguments))
        lines.append("if ($LASTEXITCODE -ne 0) { throw 'Experiment stage failed' }")
    (output / "launch-training.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, game = args.output.resolve(), args.game_dir.resolve()
    if output.exists():
        parser.error("Output must be fresh")
    if not torch.cuda.is_available():
        parser.error("Pinned preflight requires CUDA")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    monitor = MemoryMonitor()
    monitor.thread.start()
    paths = [
        Path(__file__),
        *ROOT.joinpath("src/autodancer").rglob("*.py"),
        *ROOT.joinpath("src/autodancer/data").glob("*.json"),
        *ROOT.joinpath("mods/AutoDancer").rglob("*.lua"),
        ROOT / "mods/AutoDancer/mod.json",
        game / "config.json",
        game / "lua51.dll",
        game / "autodancer_native.dll",
        game / "NecroDancer.wsp",
        game / "versions/v4.2.1.wsp",
        *game.joinpath("dlc").glob("*.necromod"),
    ]
    report = dict(
        passed=False,
        actual_training_started=False,
        purpose="bounded-learning-preflight",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes={str(p): sha256_file(p) for p in paths},
        stages=[],
    )
    status_path = output / "preflight.json"
    atomic_json(status_path, report)
    supervisors, samples, initial_hashes = [], [], []

    class ObservedSupervisor(PrivateSupervisor):
        def __init__(self, config):
            super().__init__(replace(config, profile_root=output / "profiles"))
            supervisors.append(self)

        def environment(self, *a, **kw):
            env = super().environment(*a, **kw)
            reset = env.reset

            def observed_reset(*a, **kw):
                obs, info = reset(*a, **kw)
                if not samples:
                    samples.append({key: value.copy() for key, value in obs.items()})
                return obs, info

            env.reset = observed_reset
            return env

    class ObservedModel(WorldActorCritic):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            initial_hashes.append(digest(self.state_dict()))

    def stage(name, module, arguments):
        print(json.dumps(dict(stage=name, status="starting")), flush=True)
        before = time.monotonic()
        with (
            patch.object(module, "AutoDancerSupervisor", ObservedSupervisor),
            patch.object(sys, "argv", [module.__name__, *arguments]),
        ):
            assert module.main() == 0
        report["stages"].append(dict(name=name, seconds=time.monotonic() - before))
        atomic_json(status_path, report)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        host, mod = private_host(game, output)
        # Preserve the existing bounded exploration recipe while disabling the
        # boss-specific potential for a first-floor objective.
        reward = replace(
            load_reward_config(ROOT / "configs/reward-trial-exploration-v5.json"),
            boss_progress_potential_per_damage=0,
        )
        reward_path = output / "reward.json"
        atomic_json(reward_path, asdict(reward))
        smoke_args = train_args(
            host, mod, reward_path, output / "smoke", 512, "330000-330015", max_turns=128
        )
        with patch.object(train, "WorldActorCritic", ObservedModel):
            stage("train_two_updates", train, smoke_args)
        checkpoint = output / "smoke/final.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["global_step"] == 512 and payload["updates"] == 2
        assert digest(payload["model"]) != initial_hashes[0]
        assert all(torch.isfinite(value).all() for value in payload["model"].values())
        restored = RecurrentPPO(
            model_from_spec(payload["architecture"], initialize=False),
            PPOConfig(**payload["config"]),
            device=torch.device("cuda"),
            checkpoint_metadata=payload["checkpoint_metadata"],
        )
        restored.load(checkpoint)
        assert_equal(payload["optimizer"], restored.optimizer.state_dict())
        expected = model_from_spec(payload["architecture"], initialize=False).cuda().eval()
        expected.load_state_dict(payload["model"])
        restored.model.eval()
        sample = {k: torch.from_numpy(v).unsqueeze(0).cuda() for k, v in samples[0].items()}
        with torch.inference_mode():
            assert_equal(
                expected.step(sample, expected.initial_state(1, device="cuda")),
                restored.model.step(sample, restored.model.initial_state(1, device="cuda")),
            )
        report["checkpoint_roundtrip"] = dict(model_outputs_exact=True, optimizer_exact=True)
        del restored, expected, payload, sample
        gc.collect()
        torch.cuda.empty_cache()
        resume_args = train_args(
            host, mod, reward_path, output / "resume", 768, "330000-330015", max_turns=128
        )
        stage("resume_one_update", train, [*resume_args, "--resume", str(checkpoint)])
        final = output / "resume/final.pt"
        resumed = torch.load(final, map_location="cpu", weights_only=False)
        assert resumed["global_step"] == 768 and resumed["updates"] == 3
        report["smoke_global_steps"], report["smoke_updates"] = 768, 3
        report["architecture"] = resumed["architecture"]
        report["checkpoint_metadata"] = resumed["checkpoint_metadata"]
        del resumed
        eval_path = output / "evaluation.json"
        stage(
            "heldout_evaluation",
            baseline,
            evaluation_args(host, mod, reward_path, final, eval_path, [340000, 340001], 64),
        )
        evaluation = json.loads(eval_path.read_text())
        assert evaluation["controller_valid"]
        assert evaluation["reward_semantics_matches_checkpoint"]
        report["evaluation"] = {
            k: evaluation[k]
            for k in (
                "controller_valid",
                "reward_semantics_matches_checkpoint",
                "policy_consumes_reward_feedback",
            )
        }
        rows = []
        for folder in ("smoke", "resume"):
            rows.extend(
                json.loads(line)
                for line in (output / folder / "metrics.jsonl").read_text().splitlines()
            )
            assert json.loads((output / folder / "run-status.json").read_text())["completed"]
        assert len(rows) == 3
        for row in rows:
            assert all(math.isfinite(v) for v in row.values() if type(v) in (float, int))
            assert (
                row["optimizer_steps"] > 0
                and row["gradient_world"] > 0
                and row["gradient_task"] > 0
            )
        report["learning_metrics"] = [
            {k: v for k, v in row.items() if type(v) in (float, int)} for row in rows
        ]
        report["worker_restarts"] = sum(
            h.restart_count for s in supervisors for h in s.workers.values()
        )
        assert report["worker_restarts"] == 0
        assert all(not h.failure_history for s in supervisors for h in s.workers.values())
        report["source_files_unchanged"] = all(
            sha256_file(Path(p)) == h for p, h in report["source_hashes"].items()
        )
        assert report["source_files_unchanged"]
        experiment = output / "experiment"
        train_command = train_args(
            game,
            ROOT / "mods/AutoDancer",
            reward_path,
            experiment,
            65536,
            "310000-310999",
            max_turns=1024,
        )
        train_command[train_command.index("--checkpoint-interval") + 1] = "8192"
        eval_seeds = list(range(320000, 320016))
        eval_command = evaluation_args(
            game,
            ROOT / "mods/AutoDancer",
            reward_path,
            experiment / "final.pt",
            output / "experiment-evaluation.json",
            eval_seeds,
            1024,
        )
        recipe = [
            ("autodancer.training.train", train_command),
            ("autodancer.training.baseline", eval_command),
        ]
        manifest = dict(
            status="prepared_not_started",
            training_seeds=[310000, 310999],
            evaluation_seeds=eval_seeds,
            smoke_training_seeds=[330000, 330015],
            smoke_evaluation_seeds=[340000, 340001],
            total_training_steps=65536,
            workers=2,
            objective="first_floor_completion",
            fresh_initialization=True,
            cpu_threads=4,
            reward_sha256=sha256_file(reward_path),
            commands=recipe,
        )
        atomic_json(output / "experiment.json", manifest)
        write_launcher(output, recipe)
        report["passed"] = True
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=2)
        report["resources"] = {
            **monitor.samples,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved": torch.cuda.max_memory_reserved(),
        }
        atomic_json(status_path, report)
        print(json.dumps(dict(passed=report["passed"], report=str(status_path))), flush=True)


if __name__ == "__main__":
    main()
