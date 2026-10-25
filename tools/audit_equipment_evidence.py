"""Read saved prototype evidence and installed handler bytes; never launch the game."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from catalog_game_controls import archive_entries

from autodancer.equipment_state import CAPABILITY_COMPONENTS, EQUIPMENT_FIELDS, equipment_registry
from autodancer.live.qualify import _wsp_entry_payload

HANDLERS = (
    "scripts/necro/game/item/Weapon.lua",
    "scripts/necro/game/item/ActionItem.lua",
    "scripts/necro/game/item/MovementItem.lua",
    "scripts/necro/game/item/ItemBan.lua",
    "scripts/necro/game/data/item/ItemBans.lua",
)


def runtime_coverage(catalog_path: Path, qualification_paths: list[Path]) -> dict:
    """Separate registration, source rules, pickups and effect evidence.

    The harness supports the registered superset. No generation allowlist is
    inferred from a finite sample, and no forced fixture is called natural loot.
    """
    from resolve_bard_runtime import validate_records

    rows = json.loads(catalog_path.read_text())
    validate_records(rows)
    records = {r["name"]: r["data"] for r in rows}
    registry, digest = equipment_registry()
    names = {n[5:] for n in records if n.startswith("item:")}
    if names != set(registry):
        raise ValueError("Runtime catalog differs from the pinned registry")
    progression = records["catalog_context"]["progression_enabled"]
    summaries, pickup, captured, effect_cases = [], {}, set(), set()
    for path in qualification_paths:
        report = json.loads(path.read_text())
        if not (
            report.get("passed") is True
            and report.get("training") is False
            and report.get("original_files_unchanged") is True
            and report.get("worker_restarts") == 0
        ):
            raise ValueError("Unqualified report: " + str(path))
        summaries.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cases": len(report["cases"]),
            }
        )
        for case in report["cases"]:
            if case["name"].startswith("pickup:"):
                pickup[case["name"][7:]] = case["collected"]
            if case["name"].startswith("effect:"):
                effect_cases.add(case["name"][7:])
            for state in [case["reset"], *case["steps"]]:
                for item in state["equipment"]["items"]:
                    captured.add(item["name"])
                    if any(f["applicable"] and not f["known"] for f in item["facts"].values()):
                        raise ValueError("Unknown applicable field in " + str(path))
    if pickup and set(pickup) != names:
        raise ValueError("Incomplete pickup sweep")
    items = []
    for name in sorted(names):
        item, catalog = records["item:" + name], records["catalog:" + name]
        pools = {}
        for component in catalog["components"]:
            value = records["component:" + name + ":" + component]
            if component.startswith("itemPool") and isinstance(value, dict) and "weights" in value:
                pools[component] = value["weights"]
        unlock_key = records.get("component:" + name + ":itemUnlockable", {}).get("key", name)
        if catalog.get("unlock_key") != unlock_key:
            raise ValueError("Refresh runtime catalog with authoritative unlock keys")
        unlock_allows = not progression or (
            "itemUnlockable" not in catalog["components"]
            or catalog["pool_unlocked"]
            and not catalog["pool_excluded"]
        )
        items.append(
            {
                "name": name,
                "catalog_id": registry[name],
                "slot": item["slot"],
                "phase_bans": item["blocked"],
                "unlock_key": unlock_key,
                "progression_gate_allows": unlock_allows,
                "pool_weights": pools,
                "positive_pool_candidate": bool(
                    any(any(w > 0 for w in weights) for weights in pools.values())
                    and not item["blocked"]["GENERATE_ITEM_POOL"]
                    and unlock_allows
                ),
                "captured_while_carried_or_selected": name in captured,
                "ground_fixture_collected": pickup.get(name),
            }
        )
    return {
        "registry_digest": digest,
        "registered_items": len(names),
        "progression_enabled": progression,
        "positive_pool_candidates": sum(i["positive_pool_candidate"] for i in items),
        "pickup_cases": len(pickup),
        "successful_ground_fixtures": sum(pickup.values()),
        "non_walk_pickups": sorted(n for n, collected in pickup.items() if not collected),
        "effect_cases": sorted(effect_cases),
        "qualifications": summaries,
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "scope": (
            "Registered superset; candidates and forced pickups do not prove natural reachability"
        ),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("runs/game-control-catalog-current"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-catalog", type=Path)
    parser.add_argument("--qualification", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new file")
    prototype_path = args.evidence / "active-prototypes.json"
    definitions_path = args.evidence / "definitions.json"
    prototypes = json.loads(prototype_path.read_text())
    definitions = json.loads(definitions_path.read_text())
    names = [p["name"] for p in prototypes]
    if len(set(names)) != len(names) or not all(p["ok"] for p in prototypes):
        raise ValueError("Invalid or duplicate prototype evidence")
    items = [p for p in prototypes if p["category"] == "item"]
    bard = next(p for p in prototypes if p["name"] == "Bard")
    handlers = []
    for archive in sorted({definitions[name]["archive"] for name in HANDLERS}):
        expected = {name for name in HANDLERS if definitions[name]["archive"] == archive}
        for name, entry in archive_entries(args.game_dir / archive):
            if name not in expected:
                continue
            payload = _wsp_entry_payload(entry)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != definitions[name]["sha256"]:
                raise ValueError(f"Installed handler differs from saved evidence: {name}")
            handlers.append(
                {
                    "path": name,
                    "archive": archive,
                    "sha256": digest,
                    "representation": "bytecode" if payload.startswith(b"\x1b") else "text",
                    "execution_order_verified": False,
                }
            )
    if len(handlers) != len(HANDLERS):
        raise ValueError("Not all selected handlers were found")
    rows = []
    for item in items:
        data = item["data"]
        rows.append(
            {
                "name": item["name"],
                "registered": True,
                "bard_loot_eligibility": "unresolved",
                "reason": "Saved probe omitted nested ban rules, unlock state and pool conditions",
                "pool_markers": sorted(k for k in data if k.startswith("itemPool")),
                "ban_markers": sorted(k for k in data if k.startswith("itemBan")),
                "capabilities": [
                    k for k, component in CAPABILITY_COMPONENTS.items() if component in data
                ],
            }
        )
    fields = [
        {
            "name": f.name,
            "component": f.component,
            "field": f.field,
            "prototype_examples": [
                p["name"]
                for p in items
                if isinstance(p["data"].get(f.component), dict)
                and f.field in p["data"][f.component]
            ][:3],
        }
        for f in EQUIPMENT_FIELDS
    ]
    report = {
        "schema_version": 1,
        "live_game_launched": False,
        "evidence_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (prototype_path, definitions_path)
        },
        "scope": "Previously captured configuration, not a refreshed enumeration",
        "registered_items": len(items),
        "items_with_pool_markers": sum(bool(row["pool_markers"]) for row in rows),
        "bard_ban_capture": bard["data"].get("inventoryBannedItems"),
        "bard_eligibility_complete": False,
        "capability_counts": {
            k: sum(k in r["capabilities"] for r in rows) for k in CAPABILITY_COMPONENTS
        },
        "handlers": handlers,
        "field_bindings": fields,
        "items": rows,
    }
    if args.runtime_catalog:
        refreshed = runtime_coverage(args.runtime_catalog, args.qualification)
        report = {
            "schema_version": 2,
            "live_game_launched": False,
            "scope": refreshed["scope"],
            "handlers": handlers,
            "field_bindings": fields,
            "runtime_coverage": refreshed,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.runtime_catalog:
        print(
            json.dumps(
                {
                    k: refreshed[k]
                    for k in (
                        "registered_items",
                        "positive_pool_candidates",
                        "pickup_cases",
                        "successful_ground_fixtures",
                        "non_walk_pickups",
                    )
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "registered_items",
                    "items_with_pool_markers",
                    "capability_counts",
                    "bard_eligibility_complete",
                    "live_game_launched",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
