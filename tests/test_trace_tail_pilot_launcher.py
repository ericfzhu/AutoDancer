from pathlib import Path


def test_trace_tail_pilot_has_diverse_conditional_acquisition_gate() -> None:
    source = Path("tools/run-qualified-trace-tail-pilot.ps1").read_text(encoding="utf-8")
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
