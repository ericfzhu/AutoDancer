from pathlib import Path


def test_trace_tail_pilot_has_diverse_conditional_acquisition_gate() -> None:
    source = Path("tools/run-qualified-trace-tail-pilot.ps1").read_text(encoding="utf-8")
    assert "[int]$RequestedTailActions = 1" in source
    assert "[int[]]$CandidateTailActions = @()" in source
    assert "[int[]]$CalibrationPolicySeeds" in source
    assert "runs\\qualified-death-metal-tail1-warm-boundary" in source
    assert '"calibrating-source"' in source
    assert "--source-reference" in source
    assert "No trace-tail candidate lies inside the 10-90 percent live competence band" in source
    assert "source_calibration" in source
    assert "experiment_id = $ExperimentId" in source
    assert "[int]$SteamPresenceWorker = 0" in source
    assert "--steam-presence-worker $SteamPresenceWorker" in source
    assert '"--steam-presence-worker", "$SteamPresenceWorker"' in source
    assert "a8-live-calibrated-tail1" in source
    assert "--training-level-distribution-version" in source
    assert "qualified-trace-tail-v5" in source
    assert '"waiting-for-controller-qualification"' in source
    assert "Test-CurrentQualification" in source
    assert "Controller qualification heartbeat is stale" in source
    assert "qualified_distinct_seed_count -lt 3" in source
    assert "--trace-prefix-bank" in source
    assert "--trace-prefix-qualification" in source
    assert "--trace-prefix-recurrent-state" in source
    assert '"warm"' in source
    assert "completion_rate_at_least_10_percent" in source
    assert "at_least_two_reproducible_trials" in source
    assert "at_least_three_distinct_completion_seeds" in source
    assert '"expand_trace_tail"' in source
    assert '"retain_tail_boundary"' in source
    assert "autodancer.training.baseline" in source
    assert "stochastic-98001" in source
    assert "frozen final-policy episodes" in source
    assert "process_started_at" in source
    assert "Trace-search handoff heartbeat is stale" in source
    assert "ToUnixTimeSeconds" in source
    assert "DateTimeOffset]::Parse" not in source


def test_adaptive_trace_tail_experiment_predeclares_boundary_search() -> None:
    specification = Path("experiments/EXP-0028/experiment.yaml").read_text(encoding="utf-8")
    readme = Path("experiments/EXP-0028/README.md").read_text(encoding="utf-8")

    assert "qualified-trace-tail-v6" in specification
    assert "conditional-trace-tail-v15" in specification
    assert "candidate_tail_actions: [1, 2, 3, 4, 5, 6, 7, 8" in specification
    assert "competence_band: [0.10, 0.90]" in specification
    assert "assisted trace-tail completion never counts as normal-start Zone 2" in specification
    assert "-CandidateTailActions (1..16)" in readme


def test_expanded_trace_tail_experiment_reaches_near_full_boss_start() -> None:
    specification = Path("experiments/EXP-0029/experiment.yaml").read_text(encoding="utf-8")
    readme = Path("experiments/EXP-0029/README.md").read_text(encoding="utf-8")

    assert "qualified-trace-tail-v7" in specification
    assert "conditional-trace-tail-v16" in specification
    assert "candidate_tail_actions: [20, 24, 28, 32, 36, 40" in specification
    assert "76, 80, 81]" in specification
    assert "learner_turn_cap: 128" in specification
    assert "worker 0 alone initializes Steam presence" in specification
    assert "-LearnerTurnCap 128" in readme
    assert "-SteamPresenceWorker 0" in readme


def test_retained_trace_window_predeclares_balanced_expansion_and_retention() -> None:
    launcher = Path("tools/run-qualified-trace-tail-pilot.ps1").read_text(
        encoding="utf-8"
    )
    specification = Path("experiments/EXP-0030/experiment.yaml").read_text(
        encoding="utf-8"
    )
    assert "--trace-prefix-tail-window" in launcher
    assert "RetainedTailActions" in launcher
    assert "SourceCheckpointOverride" in launcher
    assert "retained_boundary_at_least_80_percent" in launcher
    assert "qualified-trace-window-v8" in specification
    assert "conditional-trace-window-v17" in specification
    assert "assisted completion never counts as normal-start Zone 2" in specification
