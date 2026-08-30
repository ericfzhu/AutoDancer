from __future__ import annotations

from pathlib import Path

import pytest

from autodancer.training.death_metal_phase3_compare import (
    SOURCE_SHA256,
    _aggregate,
    _prefix_contract,
    _validate_prefix,
)


def _episode(seed: int, *, acquired: bool, completed: bool = False) -> dict:
    return {
        "seed": seed,
        "status": (
            "curriculum_complete" if completed else "dead" if acquired else "prefix_failed"
        ),
        "turns": 20 if acquired else 0,
        "action_counts": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "natural_prefix": {
            "acquired": acquired,
            "guide_turns": 40,
        },
    }


def test_phase3_aggregate_keeps_failed_acquisitions_in_unconditional_denominator() -> None:
    report = {
        "trained": {
            "results": [
                _episode(1, acquired=True, completed=True),
                _episode(2, acquired=True),
                _episode(3, acquired=False),
                _episode(4, acquired=False),
            ]
        }
    }

    result = _aggregate([report])

    assert result["acquisition_rate"] == 0.5
    assert result["unconditional_completion_rate"] == 0.25
    assert result["conditional_completion_rate"] == 0.5
    assert result["distinct_completion_seeds"] == [1]


def test_phase3_contract_is_bound_to_exact_guide_bytes(tmp_path: Path) -> None:
    contract = {
        **_prefix_contract(),
        "guide_checkpoint": str(tmp_path / "guide.pt"),
        "guide_checkpoint_sha256": SOURCE_SHA256,
    }
    _validate_prefix(contract, path=tmp_path / "report.json")

    contract["guide_checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="guide hash mismatch"):
        _validate_prefix(contract, path=tmp_path / "report.json")


def test_exp0023_launcher_declares_exact_legal_phase3_contract() -> None:
    source = Path("tools/run-exp0020-legal-death-metal-guide.ps1").read_text(
        encoding="utf-8"
    )
    wrapper = Path("tools/run-exp0023-legal-phase3-potential.ps1").read_text(
        encoding="utf-8"
    )

    assert '"EXP-0023"' in source and '"EXP-0023"' in wrapper
    assert "reward-death-metal-potential-v5.json" in source
    assert '"--natural-prefix-target-phase", "3"' in source
    assert '"--natural-prefix-guide-mode", "stochastic"' in source
    assert '"--natural-prefix-policy-seed", "87001"' in source
    assert '"--natural-prefix-recurrent-state", "warm"' in source
    assert "10c1be7bd9e76e4fe3ec7265cc6f712170a3fa1c07b851014c40d2ab111e3b89" in source
