"""Sample Factory recurrent PPO entry point."""

from __future__ import annotations

import sys
from typing import Any

from autodancer.training.curriculum import CurriculumEnv


def default_training_device() -> str:
    """Prefer CUDA in Linux/WSL, while retaining a safe CPU fallback."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def make_environment(
    full_env_name: str,
    cfg: Any,
    env_config: dict[str, Any] | None,
    render_mode: str | None = None,
):
    del full_env_name, cfg
    config = env_config or {}
    worker = config.get("worker_index", 0)
    vector = config.get("vector_index", config.get("env_id", 0))
    return CurriculumEnv(
        stream=f"worker-{worker}-env-{vector}", split="train", render_mode=render_mode
    )


def register_components() -> None:
    from sample_factory.algo.utils.context import global_model_factory
    from sample_factory.envs.env_utils import register_env

    from autodancer.training.model import make_actor_critic, make_encoder

    register_env("autodancer_curriculum", make_environment)
    factory = global_model_factory()
    factory.register_encoder_factory(make_encoder)
    factory.register_actor_critic_factory(make_actor_critic)


def parse_arguments(argv: list[str] | None = None):
    from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(argument == "--env" or argument.startswith("--env=") for argument in arguments):
        arguments.append("--env=autodancer_curriculum")
    parser, _ = parse_sf_args(arguments)
    parser.set_defaults(
        env="autodancer_curriculum",
        train_dir="runs",
        device=default_training_device(),
        num_workers=8,
        num_envs_per_worker=8,
        worker_num_splits=2,
        use_rnn=True,
        rnn_size=256,
        rnn_type="gru",
        rollout=32,
        recurrence=32,
        batch_size=2048,
        normalize_input=False,
        normalize_returns=False,
        with_wandb=False,
    )
    return parse_full_cfg(parser, arguments)


def main() -> int:
    if sys.platform == "win32":
        print(
            "Sample Factory does not support native Windows. Train AutoDancer in "
            "Linux or WSL2; WSL2 can use the host NVIDIA GPU when CUDA is configured.",
            file=sys.stderr,
        )
        return 2
    from sample_factory.algo.utils.misc import ExperimentStatus
    from sample_factory.train import run_rl

    register_components()
    cfg = parse_arguments()
    if cfg.device == "gpu":
        try:
            import torch
        except ImportError:
            print("PyTorch is required for GPU training", file=sys.stderr)
            return 2
        if not torch.cuda.is_available():
            print(
                "GPU training was requested, but torch.cuda.is_available() is false.",
                file=sys.stderr,
            )
            return 2
    status = run_rl(cfg)
    return 0 if status == ExperimentStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
