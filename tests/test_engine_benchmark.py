from __future__ import annotations

import json
from pathlib import Path

from autodancer.live.benchmark import (
    PROBE_MARKER,
    TELEMETRY_MARKER,
    ProbeCollector,
    compare_captures,
    summarize,
    write_capture,
)
from autodancer.live.benchmark import main as benchmark_main


def _line(marker: str, value: dict) -> str:
    return marker + json.dumps(value) + "\n"


def _telemetry(position: int) -> dict:
    return {
        "game": {"version": "4.2.1", "steam_build": "22938426"},
        "seed": 7,
        "character": "Bard",
        "zone": 1,
        "floor": 1,
        "observation": {
            "grid": [[[0]]],
            "player": [position],
            "inventory": [],
            "action_mask": [1],
        },
        "events": [{"kind": "player_moved", "amount": 0, "entity_id": 91}],
        "episode_status": "running",
        "terminated": False,
        "truncated": False,
    }


def _start(probe_id: str = "probe-1", mode: str = "process") -> dict:
    return {
        "schema_version": 1,
        "kind": "start",
        "probe_id": probe_id,
        "mode": mode,
        "status": "running",
        "target_commands": 2,
        "completed_commands": 0,
        "probe_elapsed_seconds": 0,
    }


def _turn(command: int, *, action: int = 1, elapsed: float = 0.01) -> dict:
    return {
        "schema_version": 1,
        "kind": "turn",
        "probe_id": "probe-1",
        "mode": "process",
        "status": "running",
        "target_commands": 2,
        "completed_commands": command,
        "command_id": command,
        "requested_action": action,
        "engine_action": 1,
        "observed_action": 1,
        "turn_before": command - 1,
        "turn_after": command,
        "turn_delta": 1,
        "command_elapsed_seconds": elapsed,
        "probe_elapsed_seconds": elapsed * command,
    }


def _finish() -> dict:
    return {
        "schema_version": 1,
        "kind": "finish",
        "probe_id": "probe-1",
        "mode": "process",
        "status": "completed",
        "reason": "target_reached",
        "target_commands": 2,
        "completed_commands": 2,
        "probe_elapsed_seconds": 0.02,
    }


def test_collector_pairs_probe_turn_with_preceding_telemetry() -> None:
    collector = ProbeCollector()
    collector.feed_line(_line(PROBE_MARKER, _start()))
    collector.feed_line(_line(TELEMETRY_MARKER, _telemetry(1)))
    collector.feed_line(_line(PROBE_MARKER, _turn(1)))
    collector.feed_line(_line(TELEMETRY_MARKER, _telemetry(2)))
    collector.feed_line(_line(PROBE_MARKER, _turn(2)))
    collector.feed_line(_line(PROBE_MARKER, _finish()))

    summary = summarize(collector, python_elapsed=0.03)
    assert collector.done
    assert len(collector.turns) == 2
    assert collector.turns[0]["telemetry"]["observation"]["player"] == [1]
    assert summary["turns_per_second"] == 100.0
    assert summary["command_ids_contiguous"]
    assert summary["non_unit_turn_deltas"] == 0


def test_capture_comparison_ignores_entity_ids_but_detects_state_change(
    tmp_path: Path,
) -> None:
    baseline_collector = ProbeCollector()
    candidate_collector = ProbeCollector()
    for collector in (baseline_collector, candidate_collector):
        collector.feed_line(_line(PROBE_MARKER, _start()))
        collector.feed_line(_line(TELEMETRY_MARKER, _telemetry(1)))
        collector.feed_line(_line(PROBE_MARKER, _turn(1)))
        collector.feed_line(_line(TELEMETRY_MARKER, _telemetry(2)))
        collector.feed_line(_line(PROBE_MARKER, _turn(2)))
        collector.feed_line(_line(PROBE_MARKER, _finish()))

    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    write_capture(baseline, baseline_collector, summarize(baseline_collector, 0.02))
    write_capture(candidate, candidate_collector, summarize(candidate_collector, 0.02))
    assert compare_captures(baseline, candidate)["equivalent"]

    changed = ProbeCollector()
    changed.feed_line(_line(PROBE_MARKER, _start()))
    changed.feed_line(_line(TELEMETRY_MARKER, _telemetry(99)))
    changed.feed_line(_line(PROBE_MARKER, _turn(1)))
    changed.feed_line(_line(TELEMETRY_MARKER, _telemetry(2)))
    changed.feed_line(_line(PROBE_MARKER, _turn(2)))
    changed.feed_line(_line(PROBE_MARKER, _finish()))
    write_capture(candidate, changed, summarize(changed, 0.02))
    result = compare_captures(baseline, candidate)
    assert not result["equivalent"]
    assert result["first_mismatch"]["reason"] == "telemetry_state"


def test_capture_comparison_rejects_empty_and_incomplete_captures(tmp_path: Path) -> None:
    empty_a = tmp_path / "empty-a.jsonl"
    empty_b = tmp_path / "empty-b.jsonl"
    empty_a.write_text("", encoding="utf-8")
    empty_b.write_text("", encoding="utf-8")

    empty_result = compare_captures(empty_a, empty_b)
    assert not empty_result["equivalent"]
    assert empty_result["first_mismatch"]["reason"] == "capture_integrity"
    assert "missing_finish" in empty_result["first_mismatch"]["baseline_errors"]
    assert benchmark_main(["compare", str(empty_a), str(empty_b)]) == 1

    partial_collector = ProbeCollector()
    partial_collector.feed_line(_line(PROBE_MARKER, _start()))
    partial_collector.feed_line(_line(TELEMETRY_MARKER, _telemetry(1)))
    partial_collector.feed_line(_line(PROBE_MARKER, _turn(1)))
    partial_a = tmp_path / "partial-a.jsonl"
    partial_b = tmp_path / "partial-b.jsonl"
    partial_summary = summarize(partial_collector, 0.02)
    write_capture(partial_a, partial_collector, partial_summary)
    write_capture(partial_b, partial_collector, partial_summary)

    partial_result = compare_captures(partial_a, partial_b)
    assert not partial_result["equivalent"]
    assert partial_result["first_mismatch"]["reason"] == "capture_integrity"
    assert "status_not_completed" in partial_result["first_mismatch"]["baseline_errors"]
