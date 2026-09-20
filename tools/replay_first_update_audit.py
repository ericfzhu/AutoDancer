"""Recover the saved EXP-0039 diagnostic offline after post-update bookkeeping failed."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.training.model import model_from_spec
from autodancer.training.ppo import PPOConfig, RecurrentPPO, RolloutBatch
from tools.audit_first_update import OUT, replay


def main():
    started = time.monotonic()
    manifest = json.loads((OUT / "manifest.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    for path, digest in manifest["hashes"].items():
        artifact = Path(path)
        if artifact.name == "audit_first_update.py":
            artifact = OUT / "executed-audit.py"
        assert sha256_file(artifact) == digest, path
    assert sha256_file(OUT / "initial.pt") == audit["initial_sha256"]
    assert sha256_file(OUT / "rollout.pt") == audit["capture_sha256"]
    metrics = json.loads((OUT / "training/metrics.jsonl").read_text().splitlines()[0])
    assert metrics["global_step"] == 1024 and metrics["updates"] == 1
    assert metrics["worker_restarts"] == metrics["collector_recoveries_total"] == 0
    saved = torch.load(OUT / "initial.pt", map_location="cpu", weights_only=False)
    rollout = RolloutBatch(**torch.load(OUT / "rollout.pt", weights_only=False))
    report = {"live_attempt_status_preserved": True, "new_gameplay_steps": 0, "arms": {}}
    for name, target in (("original", None), ("guarded", 0.02)):
        model = model_from_spec(saved["architecture"], initialize=False)
        ppo = RecurrentPPO(model, PPOConfig(**saved["config"]), device=torch.device("cuda"))
        ppo.sequence_encoding_batch_size = 16
        ppo.load(OUT / "initial.pt")
        before = replay(ppo, rollout)
        assert before["logprob_max_abs_difference"] < 0.001
        ppo.config = replace(ppo.config, target_kl=target)
        result = ppo.update(rollout)
        after = replay(ppo, rollout)
        assert all(math.isfinite(float(v)) for v in result.values())
        report["arms"][name] = {"before": before, "metrics": result, "after": after}
        del ppo, model
        torch.cuda.empty_cache()
    report["input_hashes"] = {
        str(OUT / name): sha256_file(OUT / name)
        for name in ("initial.pt", "rollout.pt", "audit.json", "manifest.json", "executed-audit.py")
    }
    report["replayer_sha256"] = sha256_file(Path(__file__))
    report["elapsed_seconds"] = time.monotonic() - started
    assert report["elapsed_seconds"] + manifest["elapsed_seconds"] < 600
    report["valid_offline_diagnostic"] = True
    atomic_json(OUT / "offline-validation.json", report)
    print(
        json.dumps(
            {
                k: {"after": v["after"], "optimizer_steps": v["metrics"]["optimizer_steps"]}
                for k, v in report["arms"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
