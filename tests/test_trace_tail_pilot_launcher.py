from pathlib import Path


def test_trace_tail_pilot_has_diverse_conditional_acquisition_gate() -> None:
    source = Path("tools/run-qualified-trace-tail-pilot.ps1").read_text(encoding="utf-8")
    assert "[int]$RequestedTailActions = 1" in source
    assert "runs\\qualified-death-metal-tail1-warm-boundary" in source
    assert '"calibrating-source"' in source
    assert "--source-reference" in source
    assert "outside the 10-90 percent live competence band" in source
    assert "source_calibration" in source
    assert 'experiment_id = "EXP-0027"' in source
    assert source.count("--steam-presence-worker") == 3
    assert "a8-live-calibrated-tail1" in source
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
