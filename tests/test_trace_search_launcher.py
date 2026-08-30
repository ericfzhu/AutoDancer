from pathlib import Path


def test_trace_search_launcher_separates_training_and_excluded_seeds() -> None:
    source = Path("tools/run-qualified-trace-search.ps1").read_text(encoding="utf-8")
    assert "ExcludedSeedSelection" in source
    assert "overlaps excluded evaluation seeds" in source
    assert "autodancer.training.baseline" in source
    assert "--policy-mode stochastic" in source
    assert "--curriculum-profile player20" in source
    assert 'autodancer.training.demonstration_replay", "build' in source
    assert "autodancer.training.demonstration_replay qualify" in source
    assert "No successful full-reset traces" in source
    assert "MinQualifiedDistinctSeeds" in source
    assert "Qualified traces cover only" in source
    assert "qualified_distinct_seeds" in source
    assert "96001..96032" in source
    assert "Trace search exhausted" in source
    assert "if ($candidateDistinctSeeds.Count -ge $MinQualifiedDistinctSeeds)" in source
