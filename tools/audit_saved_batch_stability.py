"""EXP-0040: bounded offline step-size and frozen-feature critic diagnostics."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import ExperimentStore, atomic_json
from autodancer.training.model import (
    evaluate_sequence_batched,
    is_critic_parameter,
    model_from_spec,
    set_actor_trainable,
)
from autodancer.training.ppo import (
    PPOConfig,
    RecurrentPPO,
    RolloutBatch,
    generalized_advantage_estimate,
)
from tools import run_bounded_learning_pilot as runtime
from tools.audit_first_update import replay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/saved-batch-stability-exp0040"
SOURCE = ROOT / "runs/first-update-audit-exp0039"


def load_ppo(saved):
    model = model_from_spec(saved["architecture"], initialize=False)
    ppo = RecurrentPPO(model, PPOConfig(**saved["config"]), device=torch.device("cuda"))
    ppo.sequence_encoding_batch_size = 16
    ppo.load(SOURCE / "initial.pt")
    return ppo


@torch.no_grad()
def recurrent_features(ppo, rollout):
    """Cache exactly the recurrent features consumed by the critic in PPO replay."""
    features = torch.empty((*rollout.actions.shape, ppo.model.hidden_size))
    chunks = [(w, t) for w in range(8) for t in range(0, 128, 32)]
    observed = []
    handle = ppo.model.base.critic.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().cpu())
    )
    ppo.model.eval()
    try:
        for offset in range(0, len(chunks), 8):
            selected = chunks[offset : offset + 8]

            def gather(value, selected=selected):
                return torch.stack([value[t : t + 32, w] for w, t in selected]).to(ppo.device)

            evaluate_sequence_batched(
                ppo.model,
                {k: gather(v) for k, v in rollout.observations.items()},
                gather(rollout.actions).long(),
                torch.stack([rollout.hiddens[t, w] for w, t in selected]).to(ppo.device),
                gather(rollout.episode_starts).bool(),
                gather(rollout.hiddens),
                encoder_batch_size=16,
            )
            batch = observed.pop()
            assert not observed
            for row, (worker, start) in enumerate(selected):
                features[start : start + 32, worker] = batch[row]
    finally:
        handle.remove()
    return features


def observed_terminal_returns(rollout, gamma):
    """Only label prefixes whose true termination is contained in the saved batch."""
    targets = torch.zeros_like(rollout.rewards)
    mask = torch.zeros_like(rollout.dones, dtype=torch.bool)
    for worker in range(rollout.actions.shape[1]):
        known, future = False, 0.0
        for step in reversed(range(rollout.actions.shape[0])):
            if bool(rollout.dones[step, worker]):
                known = bool(rollout.terminations[step, worker])
                future = 0.0
            if known:
                future = float(rollout.rewards[step, worker]) + gamma * future
                targets[step, worker] = future
                mask[step, worker] = True
    return targets, mask


def finite_metrics(metrics):
    assert all(math.isfinite(float(value)) for value in metrics.values())


def child():
    spec = ExperimentStore(ROOT / "experiments").load("EXP-0040")
    settings = spec.data["evaluation"]
    saved = torch.load(SOURCE / "initial.pt", map_location="cpu", weights_only=False)
    rollout = RolloutBatch(**torch.load(SOURCE / "rollout.pt", weights_only=False))
    result = {"new_gameplay_steps": 0, "ppo_arms": {}}
    for name, rate in zip(
        ("guarded_original", "guarded_tenth", "guarded_hundredth"),
        settings["learning_rates"],
        strict=True,
    ):
        ppo = load_ppo(saved)
        before = replay(ppo, rollout)
        assert before["logprob_max_abs_difference"] < settings["replay_logprob_tolerance"]
        ppo.config = replace(ppo.config, learning_rate=rate, target_kl=0.02)
        for group in ppo.optimizer.param_groups:
            group["lr"] = rate
        metrics = ppo.update(rollout)
        after = replay(ppo, rollout)
        finite_metrics(metrics)
        finite_metrics(after)
        result["ppo_arms"][name] = {
            "learning_rate": rate,
            "before": before,
            "metrics": metrics,
            "after": after,
            "below_diagnostic_threshold": after["sampled_action_approx_kl"] < 0.02,
        }
        atomic_json(OUT / "partial-results.json", result)
        print(json.dumps({"arm": name, "after": after}), flush=True)
        del ppo
        torch.cuda.empty_cache()

    ppo = load_ppo(saved)
    set_actor_trainable(ppo.model, False)
    frozen = {
        k: v.detach().cpu().clone()
        for k, v in ppo.model.named_parameters()
        if not is_critic_parameter(k)
    }
    features = recurrent_features(ppo, rollout).to(ppo.device)
    _, targets = generalized_advantage_estimate(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        rollout.next_value,
        terminations=rollout.terminations,
        truncation_values=rollout.truncation_values,
        gamma=ppo.config.gamma,
        gae_lambda=ppo.config.gae_lambda,
    )
    targets = targets.to(ppo.device)
    observed, mask = observed_terminal_returns(rollout, ppo.config.gamma)
    assert int(mask.sum()) == 92 and int(rollout.terminations.sum()) == 1
    head = ppo.model.base.critic

    @torch.no_grad()
    def errors():
        predictions = head(features).squeeze(-1)
        report = {}
        for name, workers in (
            ("fit", settings["fit_workers"]),
            ("heldout", settings["heldout_workers"]),
        ):
            report[f"{name}_fixed_target_mse"] = float(
                (predictions[:, workers] - targets[:, workers]).square().mean()
            )
        report["observed_terminal_prefix_mse"] = float(
            (predictions.cpu()[mask] - observed[mask]).square().mean()
        )
        report["observed_terminal_prefix_prediction_mean"] = float(predictions.cpu()[mask].mean())
        finite_metrics(report)
        return report

    with torch.no_grad():
        assert (head(features).squeeze(-1).cpu() - rollout.values).abs().max() < 0.001
    before_errors = errors()
    before_policy = replay(ppo, rollout)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings["critic_learning_rate"])
    workers = settings["fit_workers"]
    losses = []
    for _ in range(settings["critic_steps"]):
        prediction = head(features[:, workers]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, targets[:, workers])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), ppo.config.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert all(math.isfinite(value) for value in losses)
    assert all(
        torch.equal(value.detach().cpu(), frozen[name])
        for name, value in ppo.model.named_parameters()
        if name in frozen
    )
    after_policy = replay(ppo, rollout)
    for key in (
        "logprob_max_abs_difference",
        "sampled_action_approx_kl",
        "ratio_outside_clip_fraction",
    ):
        assert after_policy[key] == before_policy[key], key
    result["critic_head"] = {
        "before": before_errors,
        "after": errors(),
        "losses": losses,
        "actor_parameters_bitwise_unchanged": True,
        "policy_before": before_policy,
        "policy_after": after_policy,
        "observed_terminal_transitions": int(mask.sum()),
        "observed_terminal_episodes": 1,
        "limitation": (
            "Fixed bootstrapped targets are not ground truth. The observed-return subset "
            "is one correlated failure, not independent validation of gameplay calibration. "
            "No calibrated-critic PPO arm was run because final bootstrap observations "
            "were not saved."
        ),
    }
    atomic_json(OUT / "results.json", result)
    print(json.dumps({"critic_head": result["critic_head"]}), flush=True)


def main():
    if "--child" in sys.argv:
        child()
        return
    store = ExperimentStore(ROOT / "experiments")
    spec = store.load("EXP-0040")
    audit = json.loads((SOURCE / "audit.json").read_text())
    assert sha256_file(SOURCE / "initial.pt") == audit["initial_sha256"]
    assert sha256_file(SOURCE / "rollout.pt") == audit["capture_sha256"]
    previous = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in previous["hashes"].items():
        path = (
            SOURCE / "executed-audit.py"
            if Path(name).name == "audit_first_update.py"
            else Path(name)
        )
        assert sha256_file(path) == digest, path
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [
        spec.path,
        Path(__file__).resolve(),
        ROOT / "tools/audit_first_update.py",
        ROOT / "tools/run_bounded_learning_pilot.py",
        SOURCE / "initial.pt",
        SOURCE / "rollout.pt",
    ]
    paths += list((ROOT / "src/autodancer").rglob("*.py"))
    manifest = {
        "experiment_id": "EXP-0040",
        "spec_sha256": spec.digest,
        "hashes": {str(p): sha256_file(p) for p in paths},
        "budget_seconds": 600,
    }
    atomic_json(OUT / "manifest.json", manifest)
    runtime.OUT = OUT
    store.set_status("EXP-0040", "running")
    started = time.monotonic()
    try:
        manifest["invocation"] = runtime.execute(
            "offline", "tools.audit_saved_batch_stability", ["--child"], started + 600
        )
        assert all(sha256_file(Path(p)) == digest for p, digest in manifest["hashes"].items())
        atomic_json(OUT / "status.json", {"complete": True, "valid": True})
    except BaseException as error:
        atomic_json(OUT / "status.json", {"complete": False, "valid": False, "error": str(error)})
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    main()
