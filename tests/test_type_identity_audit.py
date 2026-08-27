from __future__ import annotations

import json

import pytest

from autodancer.training.type_identity_audit import _load_catalog, audit_catalog, lua_type_id


def test_lua_type_id_matches_bridge_algorithm() -> None:
    assert lua_type_id("") == 1
    assert lua_type_id("Gold") == lua_type_id("gold")
    assert lua_type_id("Bard") == 2450


def test_audit_reports_collisions_only_within_a_channel() -> None:
    # These two names collide under the exact 4,095-bucket Lua hash.
    first, second = "aa", "bah"
    assert lua_type_id(first) == lua_type_id(second)
    report = audit_catalog(
        {
            "actor_type": [first, second, "Bard"],
            "item_type": [first],
            "object_type": [second],
        }
    )
    actor = report["channels"]["actor_type"]
    assert actor["collision_groups"] == 1
    assert actor["collisions"] == [
        {"type_id": lua_type_id(first), "names": [first, second]}
    ]
    assert report["channels"]["item_type"]["collision_groups"] == 0
    assert report["channels"]["object_type"]["collision_groups"] == 0
    assert report["summary"]["collision_free"] is False


def test_catalog_validation_rejects_case_insensitive_duplicates(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 1, "channels": {"actor_type": ["Bard", "bard"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        _load_catalog(path)
