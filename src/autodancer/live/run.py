"""Run an exported recurrent policy against the native live game adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from autodancer.envs.live import AutoDancerLiveEnv
from autodancer.evaluation import run_episode, summarize
from autodancer.tasks import TASKS


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


class RecurrentTorchPolicy:
    """Callable policy that owns and resets the exported GRU state."""

    def __init__(
        self,
        policy: str | Path | nn.Module,
        *,
        device: str = "auto",
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        self.device = resolve_device(device)
        self.deterministic = deterministic
        self.generator = torch.Generator(device=self.device.type)
        self.generator.manual_seed(seed)
        self.metadata: dict[str, Any] = {}
        if isinstance(policy, (str, Path)):
            path = Path(policy)
            self.module = torch.jit.load(str(path), map_location=self.device)
            metadata_path = path.with_suffix(path.suffix + ".json")
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            self.module = policy.to(self.device)
        self.module.eval()
        self._initial_state = self._read_initial_state()
        self.rnn_state = self._initial_state.clone()

    def _read_initial_state(self) -> Tensor:
        state = getattr(self.module, "initial_rnn_state", None)
        if isinstance(state, Tensor):
            return state.detach().to(self.device).clone()
        rnn_size = int(self.metadata.get("rnn_size", 0))
        if rnn_size <= 0:
            raise RuntimeError(
                "The exported policy has no initial_rnn_state and no sidecar rnn_size"
            )
        return torch.zeros(1, rnn_size, device=self.device)

    def reset(self) -> None:
        self.rnn_state = self._initial_state.clone()

    @staticmethod
    def _batch(value: np.ndarray, device: torch.device) -> Tensor:
        return torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device)

    def __call__(self, observation: dict[str, np.ndarray]) -> int:
        mask = self._batch(observation["action_mask"], self.device).bool()
        if not bool(mask.any()):
            raise RuntimeError("The live observation masks every action")
        with torch.inference_mode():
            logits, next_state = self.module(
                self._batch(observation["grid"], self.device),
                self._batch(observation["player"], self.device),
                self._batch(observation["inventory"], self.device),
                self._batch(observation["action_mask"], self.device),
                self.rnn_state,
            )
            logits = logits.masked_fill(~mask, -1.0e9)
            if self.deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                probabilities = torch.softmax(logits, dim=-1)
                action = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=self.generator,
                    ).item()
                )
            self.rnn_state = next_state.detach()
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an exported policy in NecroDancer")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="all_zones")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--turn-timeout", type=float, default=5.0)
    parser.add_argument("--attach-existing", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.episodes <= 0:
        parser.error("--episodes must be positive")
    max_turns = arguments.max_turns or TASKS[arguments.task].max_turns
    if max_turns <= 0:
        parser.error("--max-turns must be positive")

    policy = RecurrentTorchPolicy(
        arguments.policy,
        device=arguments.device,
        deterministic=not arguments.stochastic,
        seed=arguments.seed,
    )
    environment = AutoDancerLiveEnv(
        log_path=arguments.log_path,
        attach_existing=arguments.attach_existing,
        task=arguments.task,
        max_turns=max_turns,
        turn_timeout=arguments.turn_timeout,
    )
    try:
        results = []
        for episode in range(arguments.episodes):
            results.append(
                run_episode(
                    environment,
                    policy,
                    seed=arguments.seed + episode,
                    max_steps=max_turns + 1,
                )
            )
            environment.attach_existing = False
        report = summarize(results, source="live")
        report["device"] = str(policy.device)
        report["policy"] = str(arguments.policy)
        print(json.dumps(report, indent=2))
        return 0
    finally:
        environment.close()


if __name__ == "__main__":
    raise SystemExit(main())
