"""Whitelist the exact native bridge export module in the pinned game config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

SUPPORTED_SHA256 = "c864f2af92f9be9e4a1da89f4562fdce2cdf9de95423176d63a1cbd91c673a89"
NEEDLE = b'"system.utils.*"]'
REPLACEMENT = b'"system.utils.*","system.game.AutoDancerNative"]'


def patch_config(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SUPPORTED_SHA256:
        raise ValueError(f"unsupported config.json SHA-256 {digest}")
    if data.count(NEEDLE) != 1:
        raise ValueError("could not locate the pinned script whitelist")
    destination.write_bytes(data.replace(NEEDLE, REPLACEMENT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    patch_config(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
