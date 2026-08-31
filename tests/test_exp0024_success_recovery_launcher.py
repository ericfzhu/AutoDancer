from pathlib import Path


def test_exp0024_success_recovery_replays_exact_checkpoint_and_qualifies_actions() -> None:
    source = Path("tools/recover-exp0024-success-traces.ps1").read_text(encoding="utf-8")
    assert "checkpoint-00092160.pt" in source
    assert "--resume $checkpoint" in source
    assert "--seed 94001" in source
    assert "--total-steps $TargetSteps" in source
    assert "--training-seed-pool $seedCsv" in source
    assert 'status -eq "curriculum_complete"' in source
    assert "successful_action_sequence" in source
    assert "stochastic-96003" in source
    assert "Preserved update-90 search report" in source
    assert "successfulSeeds.Count -lt 3" in source
    assert '"autodancer.training.demonstration_replay", "build"' in source
    assert "demonstration_replay qualify" in source
    assert "--recurrent-output $recurrent" in source
    assert "qualifiedSeeds.Count -lt 3" in source
    assert "exact-rng-schedule-continuation-action-log-recovery" in source
