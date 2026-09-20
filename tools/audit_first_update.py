"""EXP-0039: capture one live batch and audit two updates on identical inputs."""

from __future__ import annotations

import copy
import json
import random
import runpy
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.model import evaluate_sequence_batched
from autodancer.training.ppo import RecurrentPPO, generalized_advantage_estimate
from tools import run_bounded_learning_pilot as runtime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/first-update-audit-exp0039"
ORIGINAL_UPDATE = RecurrentPPO.update


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


@torch.no_grad()
def replay(ppo, rollout):
    training = ppo.model.training
    ppo.model.eval()
    chunks = [
        (w, t)
        for w in range(rollout.actions.shape[1])
        for t in range(0, rollout.actions.shape[0], 32)
    ]
    deltas, values = [], []
    for offset in range(0, len(chunks), 8):
        selected = chunks[offset : offset + 8]

        def gather(value, selected=selected):
            return torch.stack([value[t : t + 32, w] for w, t in selected]).to(ppo.device)

        logp, _, value = evaluate_sequence_batched(
            ppo.model,
            {k: gather(v) for k, v in rollout.observations.items()},
            gather(rollout.actions).long(),
            torch.stack([rollout.hiddens[t, w] for w, t in selected]).to(ppo.device),
            gather(rollout.episode_starts).bool(),
            gather(rollout.hiddens),
            encoder_batch_size=16,
        )
        deltas.append((logp - gather(rollout.old_log_probs)).flatten().cpu())
        values.append((value - gather(rollout.values)).flatten().cpu())
    ppo.model.train(training)
    delta = torch.cat(deltas)
    return {
        "sampled_action_approx_kl": float((delta.exp() - 1 - delta).mean()),
        "logprob_max_abs_difference": float(delta.abs().max()),
        "value_max_abs_difference": float(torch.cat(values).abs().max()),
        "ratio_outside_clip_fraction": float(
            ((delta.exp() - 1).abs() > ppo.config.clip_range).float().mean()
        ),
    }


def audited_update(self, rollout):
    assert self.updates == 0 and rollout.actions.numel() == 1024
    self.save(OUT / "initial.pt")
    payload = {}
    for field in fields(rollout):
        value = getattr(rollout, field.name)
        payload[field.name] = (
            {k: v.detach().cpu() for k, v in value.items()}
            if isinstance(value, dict)
            else value.detach().cpu()
        )
    torch.save(payload, OUT / "rollout.pt")
    initial_model = copy.deepcopy(self.model).cpu()
    before = replay(self, rollout)
    assert before["logprob_max_abs_difference"] < 0.001, before
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        rollout.next_value,
        terminations=rollout.terminations,
        truncation_values=rollout.truncation_values,
        gamma=self.config.gamma,
        gae_lambda=self.config.gae_lambda,
    )
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    by_action = {}
    for action in rollout.actions.unique().tolist():
        mask = rollout.actions == action
        by_action[str(action)] = {
            "count": int(mask.sum()),
            "raw_advantage_mean": float(advantages[mask].mean()),
            "normalized_advantage_mean": float(normalized[mask].mean()),
            "positive_normalized_fraction": float((normalized[mask] > 0).float().mean()),
        }
    state = rng_state()
    result = ORIGINAL_UPDATE(self, rollout)
    after_rng = rng_state()
    original_after = replay(self, rollout)
    guarded = RecurrentPPO(initial_model, replace(self.config, target_kl=0.02), device=self.device)
    guarded.sequence_encoding_batch_size = self.sequence_encoding_batch_size
    restore_rng(state)
    guarded_metrics = ORIGINAL_UPDATE(guarded, rollout)
    guarded_after = replay(guarded, rollout)
    restore_rng(after_rng)
    historical = json.loads(
        (ROOT / "runs/bounded-navigation-learning-exp0038/training/metrics.jsonl")
        .read_text()
        .splitlines()[0]
    )
    selected = (
        "value_mean",
        "return_mean",
        "advantage_raw_mean",
        "advantage_raw_std",
        "reward_max_abs",
        "approx_kl",
        "clip_fraction",
        "optimizer_steps",
    )
    atomic_json(
        OUT / "audit.json",
        {
            "pre_update_replay": before,
            "original_after": original_after,
            "guarded_after": guarded_after,
            "original_metrics": result,
            "guarded_metrics": guarded_metrics,
            "reward_nonzero_count": int((rollout.rewards != 0).sum()),
            "termination_count": int(rollout.terminations.sum()),
            "boundary_count": int(rollout.dones.sum()),
            "value_mean": float(rollout.values.mean()),
            "return_mean": float(returns.mean()),
            "by_action": by_action,
            "historical_first_update": {k: historical[k] for k in selected},
            "capture_sha256": sha256_file(OUT / "rollout.pt"),
            "initial_sha256": sha256_file(OUT / "initial.pt"),
            "limitation": (
                "Fresh capture, not recovery of the historical batch. "
                "Same-batch KL intervention is not a gameplay evaluation "
                "or causal proof of EXP-0038 failure."
            ),
        },
    )
    return result


def child():
    command = json.loads(
        (ROOT / "runs/bounded-navigation-learning-exp0038/training-invocation.json").read_text()
    )["command"]
    args = command[3:]
    for flag, value in {
        "--experiment-id": "EXP-0039",
        "--experiment-arm": "capture",
        "--trial-id": "first-update-217001",
        "--run-dir": str(OUT / "training"),
        "--total-steps": "1024",
        "--checkpoint-interval": "1024",
    }.items():
        args[args.index(flag) + 1] = value
    RecurrentPPO.update = audited_update
    sys.argv = ["autodancer.training.train", *args]
    runpy.run_module("autodancer.training.train", run_name="__main__")


def main():
    if "--child" in sys.argv:
        child()
        return
    OUT.mkdir(parents=True, exist_ok=False)
    store = ExperimentStore(ROOT / "experiments")
    spec = store.load("EXP-0039")
    runtime._load_qualification(
        ROOT / spec.data["source"]["controller_qualification"],
        runtime.GAME,
        ROOT / "mods/AutoDancer",
    )
    files = [
        Path(__file__).resolve(),
        spec.path,
        ROOT / spec.data["source"]["checkpoint"],
        ROOT / "configs/reward-death-metal-potential-v5.json",
    ]
    files += list((ROOT / "src/autodancer").rglob("*.py"))
    files += list((ROOT / "mods/AutoDancer").rglob("*.lua"))
    manifest = {
        "experiment_id": "EXP-0039",
        "spec_sha256": spec.digest,
        "budget_seconds": 600,
        "hashes": {str(p): sha256_file(p) for p in files},
    }
    atomic_json(OUT / "manifest.json", manifest)
    runtime.OUT = OUT
    store.set_status("EXP-0039", "running")
    started = time.monotonic()
    try:
        invocation = runtime.execute(
            "training", "tools.audit_first_update", ["--child"], started + 600, training=True
        )
        assert all(sha256_file(Path(p)) == h for p, h in manifest["hashes"].items())
        manifest["invocation"] = invocation
        atomic_json(OUT / "status.json", {"complete": True, "valid": True})
    except BaseException as error:
        atomic_json(OUT / "status.json", {"complete": False, "valid": False, "error": str(error)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
