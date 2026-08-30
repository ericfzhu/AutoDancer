from pathlib import Path


def test_exp0024_trace_handoff_waits_selects_and_preserves_seed_split() -> None:
    source = Path("tools/run-exp0024-qualified-trace-handoff.ps1").read_text(encoding="utf-8")
    assert 'pipeline.status -eq "complete"' in source
    assert "comparison.selected_trial" in source
    assert "training-only-full-boss-completions" in source
    assert "run-qualified-trace-search.ps1" in source
    assert "seed-selection.json" in source
    assert "heldout-selection.json" in source
    assert "No EXP-0024 training checkpoint produced" in source
