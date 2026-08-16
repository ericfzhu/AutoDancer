"""Named random-number channels keep unrelated systems reproducible."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


class RandomChannels:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._channels: dict[str, np.random.Generator] = {}

    def channel(self, name: str) -> np.random.Generator:
        if name not in self._channels:
            digest = hashlib.blake2b(
                f"autodancer:{self.seed}:{name}".encode(), digest_size=8
            ).digest()
            child_seed = int.from_bytes(digest, "little")
            self._channels[name] = np.random.default_rng(child_seed)
        return self._channels[name]

    def snapshot(self) -> dict[str, Any]:
        """Return serializable state for every random channel used so far."""
        return {
            name: generator.bit_generator.state
            for name, generator in sorted(self._channels.items())
        }
