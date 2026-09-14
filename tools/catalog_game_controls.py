"""Read installed definitions and dump active item prototypes in a private game host."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import struct
import sys
import time
from pathlib import Path

from probe_throw_semantics import GAME, ROOT, PrivateSupervisor

from autodancer.experiments.provenance import sha256_file
from autodancer.experiments.schema import atomic_json
from autodancer.live.qualify import _wsp_entry_payload
from autodancer.live.supervisor import SupervisorConfig

OUT = ROOT / "runs/game-control-catalog-current"
PROBE = r"""
        if not catalogDumped then
            catalogDumped = true
            local seen = {}
            for _, category in ipairs({"item", "playableCharacter", "spell"}) do
                for _, name in ipairs(Entities.getEntityTypesWithComponents({category})) do
                    if not seen[name] then
                        seen[name] = true
                        local proto = Entities.getEntityPrototype(name)
                        local data = {}
                        for _, component in ipairs(COMPONENTS) do
                            local ok, value = pcall(function() return proto[component] end)
                            if ok and value ~= nil then
                                local fields = {}
                                for _, field in ipairs({"name", "active", "damage", "ammo",
                                    "maximumAmmo", "ammoPerReload", "remainingTurns",
                                    "remainingKills", "quantity", "targetType", "slot",
                                    "duration", "distance", "combo", "charges"}) do
                                    local found, scalar = pcall(function() return value[field] end)
                                    local scalarType = type(scalar)
                                    if found and (scalarType == "string" or scalarType == "number"
                                        or scalarType == "boolean") then
                                        fields[field] = scalar
                                    end
                                end
                                data[component] = fields
                            end
                        end
                        print("AUTODANCER_CATALOG:" .. jsonEncode({name=name,
                            category=category, ok=true, data=data}))
                    end
                end
            end
        end
"""


def archive_entries(path: Path):
    data = path.read_bytes()
    if data[:4] != b"WSP1":
        raise ValueError(f"Unsupported archive: {path}")
    count = struct.unpack_from("<I", data, 4)[0]
    offset = 9
    names = []
    for _ in range(count):
        length = struct.unpack_from("<H", data, offset)[0]
        names.append(data[offset + 2 : offset + 2 + length].decode("ascii"))
        offset += 2 + length
    positions = [struct.unpack_from("<Q", data, offset + i * 8)[0] for i in range(count)]
    assert positions == sorted(positions) and positions[0] == offset + count * 8
    for i, name in enumerate(names):
        end = positions[i + 1] if i + 1 < count else len(data)
        yield name, data[positions[i] : end]


def parse_catalog(log):
    records = []
    for line in log.splitlines():
        if "AUTODANCER_CATALOG:{" not in line:
            continue
        text = line.split("AUTODANCER_CATALOG:", 1)[1]
        try:
            value = json.JSONDecoder().raw_decode(text)[0]
        except json.JSONDecodeError:
            # The logger changes outer quoting for names containing apostrophes.
            text = ast.literal_eval(line.split("[info] ", 1)[1])
            value = json.loads(text.split("AUTODANCER_CATALOG:", 1)[1])
        records.append(value)
    assert records and len({r["name"] for r in records}) == len(records)
    assert all(r["ok"] for r in records)
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    definitions = {}
    archives = [
        GAME / "NecroDancer.wsp",
        GAME / "versions/v4.2.1.wsp",
        *sorted((GAME / "dlc").glob("*.necromod")),
    ]
    for archive in archives:
        for name, entry in archive_entries(archive):
            if not name.endswith(".lua"):
                continue
            try:
                payload = _wsp_entry_payload(entry)
            except Exception:
                continue
            strings = [s.decode("ascii") for s in re.findall(rb"[ -~]{4,}", payload)]
            if name.startswith("mods/") or any(
                x in name for x in ("/item/", "/spell/", "/player/", "/character/")
            ):
                definitions[name] = {
                    "archive": archive.relative_to(GAME).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "strings": strings,
                }
    atomic_json(OUT / "definitions.json", definitions)
    host = OUT / "game-host"
    host.mkdir()
    for path in GAME.iterdir():
        if path.is_file() and path.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(path, host / path.name)
    for name in ("versions", "dlc"):
        shutil.copytree(GAME / name, host / name)
    mod = host / "probe-mods/AutoDancer"
    shutil.copytree(ROOT / "mods/AutoDancer", mod)
    config = json.loads((GAME / "config.json").read_text())
    config["wos"]["game"]["assets"]["external"]["path"] = GAME.parent.as_posix()
    config["wos"]["game"]["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(config))
    entry = mod / "scripts/AutoDancer.lua"
    source = entry.read_text()
    source = source.replace(
        "local function buildObservation()",
        "local catalogDumped = false\nlocal function buildObservation()",
    )
    marker = "        result.map_bounds = currentMapBounds()"
    assert source.count(marker) == 1
    components = {"friendlyName", "playableCharacter", "actionFilter", "item", "weapon", "spell"}
    for definition in definitions.values():
        for value in definition["strings"]:
            components.update(
                re.findall(
                    r"\b(?:weapon|item|spell|inventory|character)[A-Z][A-Za-z0-9_]+\b", value
                )
            )
    atomic_json(OUT / "component-candidates.json", sorted(components))
    lua_components = "{" + ",".join(json.dumps(c) for c in sorted(components)) + "}"
    entry.write_text(source.replace(marker, PROBE.replace("COMPONENTS", lua_components) + marker))
    hashes = {
        str(p): sha256_file(p)
        for p in [*archives, GAME / "config.json", ROOT / "mods/AutoDancer/scripts/AutoDancer.lua"]
    }
    with PrivateSupervisor(
        SupervisorConfig(
            game_dir=host,
            mod_dir=mod,
            num_instances=1,
            startup_timeout=30,
            steam_presence_worker=0,
            affinity_policy="none",
            profile_root=OUT / "profiles",
            diagnostic_root=OUT / "diagnostics",
        )
    ) as supervisor:
        worker = supervisor.environment(supervisor.worker_ids[0])
        try:
            worker.reset(seed=92008)
        finally:
            worker.close()
        time.sleep(0.2)
        handle = supervisor.workers[supervisor.worker_ids[0]]
        log = handle.log_path.read_text(encoding="utf-8", errors="replace")
        (OUT / "game.log").write_text(log, encoding="utf-8")
        records = parse_catalog(log)
        assert records, "No catalog records produced"
        atomic_json(OUT / "active-prototypes.json", records)
        atomic_json(
            OUT / "provenance.json",
            {
                "hashes": hashes,
                "original_files_unchanged": all(
                    sha256_file(Path(p)) == h for p, h in hashes.items()
                ),
                "worker_restarts": handle.restart_count,
                "prototypes": len(records),
                "blacklisted_gameplay_mods": ["Amplified", "DynChar", "Synchrony"],
                "disclosure": "Prototype enumeration and one reset; no gameplay actions",
            },
        )
        print(json.dumps({"prototypes": len(records), "errors": sum(not r["ok"] for r in records)}))


if __name__ == "__main__":
    if "--parse-existing" in sys.argv:
        records = parse_catalog((OUT / "game.log").read_text(encoding="utf-8"))
        atomic_json(OUT / "active-prototypes.json", records)
        print(json.dumps({"validated_prototypes": len(records)}))
    else:
        main()
