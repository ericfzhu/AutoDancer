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
    assert '"checkpoint-00092160.pt"' in source
    assert '"checkpoint-00061440.pt"' in source
    assert "Every competence-window checkpoint failed" in source
    assert 'published["failed_candidates"]' in source
    assert "if (-not $competenceMiss)" in source
    assert "No successful full-reset traces|Trace search exhausted" in source
    assert "process_started_at" in source
    assert 'Write-HandoffStatus "waiting-for-exp0024"' in source
