"""Measure and compare real-engine AutoDancer probe runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TELEMETRY_MARKER = "AUTODANCER_JSON:"
PROBE_MARKER = "AUTODANCER_PROBE:"
_JSON_DECODER = json.JSONDecoder()


def _decode_marked(line: str, marker: str) -> dict[str, Any] | None:
    index = line.find(marker)
    if index < 0:
        return None
    payload = line[index + len(marker) :].lstrip()
    try:
        value, _ = _JSON_DECODER.raw_decode(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(slots=True)
class ProbeCollector:
    requested_probe_id: str | None = None
    selected_probe_id: str | None = None
    start: dict[str, Any] | None = None
    finish: dict[str, Any] | None = None
    latest_telemetry: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    malformed_probe_records: int = 0
    unpaired_probe_turns: int = 0

    @property
    def done(self) -> bool:
        return self.finish is not None

    def _selects(self, probe: Mapping[str, Any]) -> bool:
        probe_id = probe.get("probe_id")
        if not isinstance(probe_id, str) or not probe_id:
            self.malformed_probe_records += 1
            return False
        if self.requested_probe_id is not None and probe_id != self.requested_probe_id:
            return False
        if self.selected_probe_id is None:
            self.selected_probe_id = probe_id
        return probe_id == self.selected_probe_id

    def feed_line(self, line: str) -> None:
        telemetry = _decode_marked(line, TELEMETRY_MARKER)
        if telemetry is not None:
            self.latest_telemetry = telemetry
            return

        probe = _decode_marked(line, PROBE_MARKER)
        if probe is None or not self._selects(probe):
            return
        kind = probe.get("kind")
        if kind == "start":
            self.start = probe
            return
        if kind == "turn":
            if self.latest_telemetry is None:
                self.unpaired_probe_turns += 1
                return
            self.turns.append(
                {"kind": "turn", "probe": probe, "telemetry": self.latest_telemetry}
            )
            self.latest_telemetry = None
            return
        if kind in {"finish", "error"}:
            self.finish = probe


def collect_live(
    path: Path,
    *,
    timeout: float,
    from_start: bool,
    probe_id: str | None,
) -> tuple[ProbeCollector, float]:
    collector = ProbeCollector(requested_probe_id=probe_id)
    offset = 0
    if not from_start and path.exists():
        offset = path.stat().st_size
    started = time.monotonic()
    deadline = started + timeout

    while time.monotonic() < deadline and not collector.done:
        if path.exists():
            size = path.stat().st_size
            if size < offset:
                offset = 0
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                while True:
                    line_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        offset = line_offset
                        break
                    offset = handle.tell()
                    collector.feed_line(line)
                    if collector.done:
                        break
        if not collector.done:
            time.sleep(0.01)
    return collector, time.monotonic() - started


def summarize(collector: ProbeCollector, python_elapsed: float) -> dict[str, Any]:
    probes = [entry["probe"] for entry in collector.turns]
    command_ids = [int(item.get("command_id", -1)) for item in probes]
    expected_ids = list(range(1, len(command_ids) + 1))
    latencies = [
        float(item["command_elapsed_seconds"])
        for item in probes
        if isinstance(item.get("command_elapsed_seconds"), (int, float))
        and float(item["command_elapsed_seconds"]) >= 0
    ]
    turn_deltas = [
        int(item.get("turn_delta", 0))
        for item in probes
        if isinstance(item.get("turn_delta"), (int, float))
    ]
    observed_mismatches = sum(
        1
        for item in probes
        if item.get("observed_action") is not None
        and item.get("observed_action") != item.get("engine_action")
    )
    terminal = collector.finish or (probes[-1] if probes else {})
    probe_elapsed = terminal.get("probe_elapsed_seconds")
    if not isinstance(probe_elapsed, (int, float)) or probe_elapsed <= 0:
        probe_elapsed = None
    commands = len(probes)

    return {
        "probe_id": collector.selected_probe_id,
        "mode": (collector.start or terminal).get("mode"),
        "status": terminal.get(
            "status", "timeout" if not collector.done else "unknown"
        ),
        "reason": terminal.get("reason"),
        "commands": commands,
        "target_commands": (collector.start or terminal).get("target_commands"),
        "probe_elapsed_seconds": probe_elapsed,
        "python_elapsed_seconds": python_elapsed,
        "turns_per_second": commands / probe_elapsed if probe_elapsed else None,
        "python_observed_turns_per_second": (
            commands / python_elapsed if python_elapsed > 0 else None
        ),
        "latency_seconds": {
            "minimum": min(latencies) if latencies else None,
            "median": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "maximum": max(latencies) if latencies else None,
        },
        "command_ids_contiguous": command_ids == expected_ids,
        "turn_delta_counts": {
            str(value): turn_deltas.count(value) for value in sorted(set(turn_deltas))
        },
        "non_unit_turn_deltas": sum(value != 1 for value in turn_deltas),
        "observed_action_mismatches": observed_mismatches,
        "unpaired_probe_turns": collector.unpaired_probe_turns,
        "malformed_probe_records": collector.malformed_probe_records,
    }


def write_capture(
    path: Path, collector: ProbeCollector, summary: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if collector.start is not None:
            handle.write(
                json.dumps({"kind": "start", "probe": collector.start}, sort_keys=True)
                + "\n"
            )
        for turn in collector.turns:
            handle.write(
                json.dumps(turn, sort_keys=True, separators=(",", ":")) + "\n"
            )
        if collector.finish is not None:
            handle.write(
                json.dumps(
                    {"kind": "finish", "probe": collector.finish}, sort_keys=True
                )
                + "\n"
            )
        handle.write(
            json.dumps({"kind": "summary", "summary": dict(summary)}, sort_keys=True)
            + "\n"
        )


def _canonical_event(event: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "kind": event.get("kind"),
        "amount": event.get("amount", 0),
    }
    if "data" in event:
        result["data"] = event["data"]
    return result


def canonical_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    events = telemetry.get("events", [])
    return {
        "game": telemetry.get("game"),
        "seed": telemetry.get("seed"),
        "character": telemetry.get("character"),
        "zone": telemetry.get("zone"),
        "floor": telemetry.get("floor"),
        "observation": telemetry.get("observation"),
        "events": [
            _canonical_event(event) for event in events if isinstance(event, Mapping)
        ],
        "episode_status": telemetry.get("episode_status"),
        "terminated": telemetry.get("terminated"),
        "truncated": telemetry.get("truncated"),
    }


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_capture(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("kind") == "turn":
            if not isinstance(value.get("probe"), dict) or not isinstance(
                value.get("telemetry"), dict
            ):
                raise ValueError(f"Invalid turn record at {path}:{line_number}")
            turns.append(value)
    return turns


def compare_captures(baseline: Path, candidate: Path) -> dict[str, Any]:
    left = load_capture(baseline)
    right = load_capture(candidate)
    compared = min(len(left), len(right))
    first_mismatch: dict[str, Any] | None = None

    for index in range(compared):
        left_probe = left[index]["probe"]
        right_probe = right[index]["probe"]
        if left_probe.get("requested_action") != right_probe.get("requested_action"):
            first_mismatch = {
                "command": index + 1,
                "reason": "requested_action",
                "baseline": left_probe.get("requested_action"),
                "candidate": right_probe.get("requested_action"),
            }
            break
        left_state = canonical_telemetry(left[index]["telemetry"])
        right_state = canonical_telemetry(right[index]["telemetry"])
        left_digest = _digest(left_state)
        right_digest = _digest(right_state)
        if left_digest != right_digest:
            first_mismatch = {
                "command": index + 1,
                "reason": "telemetry_state",
                "baseline_digest": left_digest,
                "candidate_digest": right_digest,
            }
            break

    if first_mismatch is None and len(left) != len(right):
        first_mismatch = {
            "command": compared + 1,
            "reason": "length",
            "baseline_turns": len(left),
            "candidate_turns": len(right),
        }

    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "baseline_turns": len(left),
        "candidate_turns": len(right),
        "compared_turns": compared,
        "equivalent": first_mismatch is None,
        "first_mismatch": first_mismatch,
    }


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(dict(payload), indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark direct real-engine turn stepping"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Tail one live probe run")
    collect_parser.add_argument("log_path", type=Path)
    collect_parser.add_argument("--timeout", type=float, default=60.0)
    collect_parser.add_argument("--from-start", action="store_true")
    collect_parser.add_argument("--probe-id")
    collect_parser.add_argument("--capture", type=Path)
    collect_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two captured probe runs"
    )
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--output", type=Path)

    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "compare":
        result = compare_captures(arguments.baseline, arguments.candidate)
        _write_or_print(result, arguments.output)
        return 0 if result["equivalent"] else 1

    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    collector, python_elapsed = collect_live(
        arguments.log_path,
        timeout=arguments.timeout,
        from_start=arguments.from_start,
        probe_id=arguments.probe_id,
    )
    result = summarize(collector, python_elapsed)
    if arguments.capture is not None:
        write_capture(arguments.capture, collector, result)
    _write_or_print(result, arguments.output)
    if not collector.turns:
        return 2
    return 0 if collector.done and result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
