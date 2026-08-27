"""Canonical progression ordering for the four-level NecroDancer zones."""

from __future__ import annotations

LEVELS_PER_ZONE = 4


def level_progress(zone: int, floor: int) -> int:
    """Return the one-based sequential level index for a zone/floor pair."""
    zone = int(zone)
    floor = int(floor)
    if zone <= 0 or floor <= 0:
        return 0
    return (zone - 1) * LEVELS_PER_ZONE + floor


def deeper_level(
    current: tuple[int, int], candidate: tuple[int, int]
) -> tuple[int, int]:
    """Select the pair that is farther through the sequential All Zones run."""
    return candidate if level_progress(*candidate) > level_progress(*current) else current
