"""Bounded, game-free performance measurements; never launches live workers."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from autodancer.constants import (
    ACTION_COUNT,
    GRID_CHANNELS,
    GRID_SIZE,
    INVENTORY_FEATURES,
    INVENTORY_SLOTS,
    MAP_CHANNELS,
    MAP_SIZE,
    PLAYER_FEATURES,
)
from autodancer.training.model import START_ACTION, evaluate_sequence_batched, model_from_spec


def summarize_metrics(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No metrics in {path}")
    # Only use a single uninterrupted process log. Subtract its initial global
    # step so a resumed process in a separate file is also supported.
    segments: list[float] = []
    previous_step = int(rows[0]["global_step"]) - round(
        rows[0]["collector_seconds"] * rows[0]["collector_steps_per_second"]
    )
    base_step = previous_step
    previous_elapsed = 0.0
    for row in rows:
        step = int(row["global_step"])
        elapsed = (step - base_step) / float(row["steps_per_second"])
        if elapsed < previous_elapsed or step <= previous_step:
            # Ambiguous resumes cannot be reconstructed reliably from old logs.
            raise ValueError("Metrics contain a restart/overlap; summarize separate process logs")
        segments.append(elapsed - previous_elapsed)
        previous_elapsed, previous_step = elapsed, step
    elapsed = sum(segments)
    collection = sum(float(row["collector_seconds"]) for row in rows)
    result: dict[str, Any] = {
        "path": str(path),
        "updates": len(rows),
        "learner_transitions": previous_step - base_step,
        "training_loop_seconds": elapsed,
        "collection_seconds": collection,
        "collection_fraction": collection / elapsed,
        "outside_collection_seconds": elapsed - collection,
        "mean_fragment_straggler_seconds": statistics.mean(
            row["fragment_straggler_seconds"] for row in rows
        ),
        "scope": "Single-process training loop; excludes initial launch and final evaluation",
        "timing_note": "Worker times overlap across slots; prefix_total contains prefix stages",
    }
    for field in ("ppo_update_seconds", "evaluation_and_reset_seconds", "imitation_update_seconds"):
        result[field] = (
            sum(float(row.get(field, 0)) for row in rows)
            if any(field in row for row in rows)
            else None
        )
    stages: dict[str, float] = {}
    for row in rows:
        for worker in row.get("worker_stage_timings", []):
            for name, value in worker.items():
                if name.endswith(("_seconds", "_calls")):
                    stages[name] = stages.get(name, 0.0) + float(value)
    result["summed_worker_stages"] = stages
    return result


def benchmark_observations(
    batch: int, length: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Synthetic nonempty inputs with production shapes; not gameplay evidence."""
    grid = torch.zeros(batch, length, GRID_SIZE, GRID_SIZE, GRID_CHANNELS, dtype=torch.int16)
    grid[..., 0] = 1
    grid[..., 1] = 23
    grid[..., 9] = 2
    grid[:, :, 10, 10, 2] = 1
    grid[:, :, 10, 10, 3] = 123
    grid[:, :, 9, 10, 2] = 2
    grid[:, :, 9, 10, 3] = 456
    grid[..., 4:6] = 3
    memory = torch.zeros(batch, length, MAP_SIZE, MAP_SIZE, MAP_CHANNELS, dtype=torch.int16)
    memory[..., :2] = 1
    player = torch.ones(batch, length, PLAYER_FEATURES, dtype=torch.int32)
    player[..., :2] = 6
    inventory = torch.zeros(batch, length, INVENTORY_SLOTS, INVENTORY_FEATURES, dtype=torch.int16)
    inventory[..., 1] = 123
    mask = torch.ones(batch, length, ACTION_COUNT, dtype=torch.bool)
    mask[..., -2:] = False
    return {
        key: value.to(device)
        for key, value in {
            "grid": grid,
            "map_memory": memory,
            "player": player,
            "inventory": inventory,
            "action_mask": mask,
            "previous_action": torch.full((batch, length), START_ACTION),
            "previous_reward": torch.linspace(-0.1, 0.1, batch * length).reshape(batch, length),
        }.items()
    }


def sequence_benchmark(
    checkpoint: Path,
    *,
    device: torch.device,
    batch: int,
    length: int,
    encoder_batches: list[int],
    repeats: int,
) -> dict[str, Any]:
    if min(batch, length, repeats) <= 0 or any(size <= 0 for size in encoder_batches):
        raise ValueError("Batch sizes, length, and repeats must be positive")
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = model_from_spec(payload["architecture"], initialize=False).to(device).train()
    model.load_state_dict(payload["model"])
    del payload
    obs = benchmark_observations(batch, length, device)
    actions = torch.zeros(batch, length, device=device, dtype=torch.long)
    initial = model.initial_state(batch, device=device)
    starts = torch.zeros(batch, length, device=device, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, length // 2] = True
    stored = torch.randn(batch, length, 2, model.hidden_size, device=device) * 0.1

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(size: int) -> tuple[torch.Tensor, ...]:
        model.zero_grad(set_to_none=True)
        inputs = (obs, actions, initial, starts, stored)
        outputs = (
            model.evaluate_sequence(*inputs)
            if size == 0
            else evaluate_sequence_batched(model, *inputs, encoder_batch_size=size)
        )
        log_probs, entropy, values = outputs
        loss = -log_probs.mean() - 0.01 * entropy.mean() + 0.5 * (values - 5).square().mean()
        loss.backward()
        return tuple(value.detach() for value in outputs)

    reference = tuple(value.cpu() for value in run(0))
    reference_grads = {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    results = []
    for size in [0, *dict.fromkeys(encoder_batches)]:
        try:
            output = run(size)
            max_error = max(
                float((a.cpu() - b).abs().max()) for a, b in zip(output, reference, strict=True)
            )
            output_pass = all(
                torch.allclose(a.cpu(), b, atol=2e-5, rtol=2e-4)
                for a, b in zip(output, reference, strict=True)
            )
            gradient_error = 0.0
            gradient_pass = True
            for name, p in model.named_parameters():
                if name not in reference_grads:
                    gradient_pass &= p.grad is None
                    continue
                if p.grad is None:
                    gradient_pass = False
                    continue
                actual, expected = p.grad.detach().cpu(), reference_grads[name]
                gradient_error = max(gradient_error, float((actual - expected).abs().max()))
                gradient_pass &= torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)
            del output
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()
                run(size)
                synchronize()
                durations.append(time.perf_counter() - start)
            result = {
                "encoder_batch_size": size,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "max_output_error": max_error,
                "max_gradient_error": gradient_error,
                "parity_pass": bool(output_pass and gradient_pass),
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
            }
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            result = {
                "encoder_batch_size": size,
                "error": "CUDA out of memory",
                "parity_pass": False,
            }
        results.append(result)
    baseline = results[0].get("median_seconds")
    for result in results:
        if baseline and "median_seconds" in result:
            result["speedup"] = baseline / result["median_seconds"]
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "architecture": model.architecture_spec(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "batch": batch,
        "sequence_length": length,
        "scope": "Synthetic forward/backward minibatch; excludes optimizer and live collection",
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("metrics", nargs="+", type=Path)
    sequence = commands.add_parser("sequence")
    sequence.add_argument("checkpoint", type=Path)
    sequence.add_argument("--device", default="cuda")
    sequence.add_argument("--batch", type=int, default=8)
    sequence.add_argument("--length", type=int, default=32)
    sequence.add_argument("--encoder-batches", type=int, nargs="+", default=[16, 32])
    sequence.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.command == "summarize":
        report = {"runs": [summarize_metrics(path) for path in args.metrics]}
    else:
        report = sequence_benchmark(
            args.checkpoint,
            device=torch.device(args.device),
            batch=args.batch,
            length=args.length,
            encoder_batches=args.encoder_batches,
            repeats=args.repeats,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
