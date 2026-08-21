"""Start a normal All Zones Bard run through the installed Lua bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autodancer.live.bridge import FileCommandBridge
from autodancer.live.protocol import JsonlTurnSource, validate_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Start an All Zones Bard run")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("command_path", type=Path)
    parser.add_argument("--timeout", type=float, default=20.0)
    arguments = parser.parse_args()

    source = JsonlTurnSource(arguments.log_path)
    source.reset_sequence()
    FileCommandBridge(arguments.command_path).start()
    record = source.read(arguments.timeout)
    validate_record(record)
    if record["kind"] != "reset" or record["character"] != "Bard":
        raise RuntimeError("START did not produce a Bard reset record")
    print(json.dumps({
        "run_id": record["run_id"],
        "seed": record.get("seed"),
        "zone": record.get("zone"),
        "floor": record.get("floor"),
        "character": record["character"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
