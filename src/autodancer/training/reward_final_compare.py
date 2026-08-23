"""Compare the selected long V4 run with V2 on final unseen seeds."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report["trained"].items() if key != "results"}


def _rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    episodes = max(int(summary["episodes"]), 1)
    return (
        float(summary["mean_progress"]),
        -float(summary["death_rate"]),
        -float(summary["idle_rate"]),
        float(summary["enemy_kills"]) / episodes,
        float(summary["item_pickups"]) / episodes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the final V4 gameplay ordering")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reference = _summary(json.loads(arguments.reference.read_text(encoding="utf-8")))
    candidate = _summary(json.loads(arguments.candidate.read_text(encoding="utf-8")))
    promoted = _rank(candidate) > _rank(reference)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_name": arguments.candidate_name,
        "reference": reference,
        "candidate": candidate,
        "promoted": promoted,
        "decision": f"promote_{arguments.candidate_name}" if promoted else "retain_v2",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(arguments.output)
    print(json.dumps({"decision": report["decision"], "output": str(arguments.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
