"""Startup-only authoritative Bard ban and registered-handler inspection.

Uses one isolated host and one reset; never sends gameplay actions or trains.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from catalog_game_controls import parse_catalog
from probe_throw_semantics import PrivateSupervisor

from autodancer.experiments.provenance import sha256_file
from autodancer.live.supervisor import SupervisorConfig

PROBE = r"""
        if not eligibilityDumped then
            eligibilityDumped = true
            local ItemBan = require "necro.game.item.ItemBan"
            local ActionItem = require "necro.game.item.ActionItem"
            local Events = require "system.events.Events"
            local bit = require "bit"
            local function clean(value, depth)
                if depth > 10 then error("Inspection nesting exceeds contract") end
                local kind = type(value)
                if kind == "string" or kind == "boolean" or kind == "number" then return value end
                if kind ~= "table" then return "<" .. kind .. ">" end
                local out = {}
                for key, item in pairs(value) do
                    if key ~= "func" and string.sub(tostring(key),1,1) ~= "_" then
                        out[key] = clean(item, depth+1)
                    end
                end
                return out
            end
            local function emit(name,data)
                print("AUTODANCER_CATALOG:" .. jsonEncode({name=name,ok=true,data=data}))
            end
            local function playerState()
                return clean({health=player.health.health,max_health=player.health.maxHealth,
                    x=player.position.x,y=player.position.y,
                    slots=player.inventory.itemSlots},0)
            end
            local before = jsonEncode(playerState())
            local flags = {}
            for _,name in ipairs({"PICKUP","EQUIP","ACTIVATE","GENERATE_ITEM_POOL",
                "GENERATE_CRATE","GENERATE_SHRINE_FIXED","GENERATE_SHRINE_POOL",
                "GENERATE_SHRINE_DROP","GENERATE_LEVEL","GENERATE_TRANSACTION",
                "CONVERT_SHRINE","CONVERT_SPELL","CONVERT_TRANSACTION"}) do
                flags[name] = ItemBan.Flag[name]
                assert(type(flags[name]) == "number", "Missing ban flag: " .. name)
            end
            emit("context",{bard_bans=clean(Entities.getEntityPrototype("Bard").inventoryBannedItems.components,0),
                flag_values=flags,level=CurrentLevel.getSequentialNumber(),player=playerState()})
            local names = Entities.getEntityTypesWithComponents({"item"})
            table.sort(names)
            for _,name in ipairs(names) do
                local proto = Entities.getEntityPrototype(name)
                local mask = ItemBan.getBanFlags(player,name)
                local blocked = {}
                for flag,value in pairs(flags) do blocked[flag] = bit.band(mask,value) ~= 0 end
                emit("item:" .. name,{ban_mask=mask,blocked=blocked,
                    slot=proto.itemSlot and proto.itemSlot.name or "",
                    rhythmless=Entities.typeHasComponent(name,"itemBanRhythmless")})
            end
            for _,name in ipairs({"holderGetActionItem","itemActivate",
                "itemBanCheck","itemBanCheckType"}) do
                local count = 0
                for handler,entry in pairs(Events.getRegisteredFunctions(name)) do
                    count = count + 1
                    emit("handler:"..name..":"..tostring(handler),clean({
                        key=entry.key,script=entry.script,modPriority=entry.modPriority},0))
                end
                emit("handler_count:"..name,{count=count})
            end
            local owners = {}
            for _,name in ipairs({"BOMB","ITEM_1","ITEM_2","THROW","SPELL_1","SPELL_2"}) do
                local item = ActionItem.getActionItem(player,Action.Special[name])
                owners[name] = item and item.name or "none"
            end
            emit("owners",owners)
            assert(before == jsonEncode(playerState()), "Inspection changed player state")
            emit("complete",{items=#names,player_unchanged=true,gameplay_actions=0})
        end
"""

CATALOG_PROBE = r"""
            local playerComponents = {}
            local allDefaults = require("necro.game.data.Components").getComponentDefaults()
            for name, fields in pairs(allDefaults) do
                if player[name] ~= nil then
                    playerComponents[#playerComponents+1] = name
                    local values = {}
                    for field in pairs(fields) do
                        local value = player[name][field]
                        local kind = type(value)
                        if kind == "number" or kind == "boolean" then values[field] = value
                        else values[field] = "<"..kind..">" end
                    end
                    emit("player_component:"..name,values)
                end
            end
            table.sort(playerComponents)
            emit("player_components",playerComponents)
            local Progression = require "necro.game.system.Progression"
            local Components = require "necro.game.data.Components"
            local defaults = Components.getComponentDefaults()
            local componentNames = {}
            for component in pairs(defaults) do
                if string.find(component, "item") or string.find(component, "Item")
                    or string.find(component, "weapon") or string.find(component, "spell") then
                    componentNames[#componentNames+1] = component
                end
            end
            table.sort(componentNames)
            emit("catalog_context",{progression_enabled=not not Progression.isEnabled(),
                component_names=componentNames})
            for _,name in ipairs(names) do
                local proto = Entities.getEntityPrototype(name)
                local present = {}
                for _,component in ipairs(componentNames) do
                    if proto[component] ~= nil then
                        present[#present+1] = component
                        local values = {}
                        for field in pairs(defaults[component]) do
                            values[field] = clean(proto[component][field],0)
                        end
                        emit("component:"..name..":"..component,values)
                    end
                end
                local unlockKey = proto.itemUnlockable and proto.itemUnlockable.key or name
                emit("catalog:"..name,{components=present,unlock_key=unlockKey,
                    physical=proto.gameObject ~= nil and proto.position ~= nil,
                    pool_unlocked=Progression.isUnlocked(Progression.UnlockType.ITEM_POOL,unlockKey),
                    pool_excluded=Progression.isUnlocked(Progression.UnlockType.ITEM_POOL_EXCLUSION,unlockKey)})
            end
"""


FLAGS = {
    "PICKUP",
    "EQUIP",
    "ACTIVATE",
    "GENERATE_ITEM_POOL",
    "GENERATE_CRATE",
    "GENERATE_SHRINE_FIXED",
    "GENERATE_SHRINE_POOL",
    "GENERATE_SHRINE_DROP",
    "GENERATE_LEVEL",
    "GENERATE_TRANSACTION",
    "CONVERT_SHRINE",
    "CONVERT_SPELL",
    "CONVERT_TRANSACTION",
}
EVENTS = {"holderGetActionItem", "itemActivate", "itemBanCheck", "itemBanCheckType"}
ACTIONS = {"BOMB", "ITEM_1", "ITEM_2", "THROW", "SPELL_1", "SPELL_2"}


def validate_records(rows):
    """Reject incomplete evidence; report bans without claiming loot reachability."""
    by_name = {}
    for row in rows:
        name = row["name"]
        if row.get("ok") is not True or name in by_name:
            raise ValueError("Failed or duplicate record: " + name)
        by_name[name] = row["data"]
    if not rows or rows[-1]["name"] != "complete":
        raise ValueError("Missing final completion record")
    complete = by_name["complete"]
    if complete.get("player_unchanged") is not True or complete.get("gameplay_actions") != 0:
        raise ValueError("Probe did not certify an unchanged startup inspection")
    flags = by_name["context"]["flag_values"]
    if set(flags) != FLAGS or any(
        type(v) is not int or v <= 0 or v & (v - 1) for v in flags.values()
    ):
        raise ValueError("Missing or invalid phase flags")
    if len(set(flags.values())) != len(flags):
        raise ValueError("Duplicate phase flag bits")
    items = {n[5:]: d for n, d in by_name.items() if n.startswith("item:")}
    if not items or len(items) != complete["items"]:
        raise ValueError("Incomplete item records")
    for name, item in items.items():
        mask = item["ban_mask"]
        if type(mask) is not int or mask < 0 or set(item["blocked"]) != FLAGS:
            raise ValueError("Invalid item ban evidence: " + name)
        for flag, value in flags.items():
            if item["blocked"][flag] is not bool(mask & value):
                raise ValueError("Ban mask disagrees with phase: " + name)
    handlers = {}
    for event in sorted(EVENTS):
        prefix = "handler:" + event + ":"
        entries = {n[len(prefix) :]: d for n, d in by_name.items() if n.startswith(prefix)}
        if len(entries) != by_name["handler_count:" + event]["count"]:
            raise ValueError("Incomplete handler records: " + event)
        for entry in entries.values():
            if not isinstance(entry.get("script"), str) or not isinstance(entry.get("key"), dict):
                raise ValueError("Missing handler identity")
        handlers[event] = entries
    if set(by_name["owners"]) != ACTIONS:
        raise ValueError("Incomplete action owners")
    expected = {"context", "complete", "owners"} | {"item:" + n for n in items}
    if "catalog_context" in by_name:
        expected.add("catalog_context")
        if "player_components" in by_name:
            expected.add("player_components")
            expected.update("player_component:" + n for n in by_name["player_components"])
        for name in items:
            key = "catalog:" + name
            expected.add(key)
            row = by_name[key]
            if type(row["pool_unlocked"]) is not bool or type(row["pool_excluded"]) is not bool:
                raise ValueError("Incomplete unlock state: " + name)
            if len(set(row["components"])) != len(row["components"]):
                raise ValueError("Duplicate component: " + name)
            expected.update("component:" + name + ":" + c for c in row["components"])
    for event, entries in handlers.items():
        expected.add("handler_count:" + event)
        expected.update("handler:" + event + ":" + n for n in entries)
    if set(by_name) != expected:
        raise ValueError("Unexpected records")
    return {
        "passed": True,
        "items": len(items),
        "owners": by_name["owners"],
        "blocked_by_phase": {
            flag: sorted(n for n, d in items.items() if d["blocked"][flag])
            for flag in sorted(FLAGS)
        },
        "handlers": handlers,
        "scope": (
            "Startup ban rules and registered handlers; "
            "not proof of loot reachability or action success"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path)
    parser.add_argument(
        "--existing-log", type=Path, help="Validate a flushed log without launching the game"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-detail", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Output must be a fresh directory")
    root.mkdir(parents=True)
    if args.existing_log:
        rows = parse_catalog(args.existing_log.read_text(encoding="utf-8", errors="strict"))
        report = validate_records(rows)
        report["log_sha256"] = sha256_file(args.existing_log)
        report["source_log"] = str(args.existing_log.resolve())
        report["validator_sha256"] = sha256_file(Path(__file__))
        report["game_launched_by_validation"] = False
        (root / "records.json").write_text(json.dumps(rows, indent=2) + "\n")
        (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "items": report["items"]}))
        return
    if not args.game_dir:
        parser.error("--game-dir is required unless --existing-log is supplied")
    game = args.game_dir.resolve()
    host = root / "game-host"
    host.mkdir()
    for p in game.iterdir():
        if p.is_file() and p.suffix.lower() in {".exe", ".dll", ".wsp"}:
            shutil.copy2(p, host / p.name)
    for name in ("versions", "dlc"):
        shutil.copytree(game / name, host / name)
    mod = host / "probe-mods/AutoDancer"
    shutil.copytree("mods/AutoDancer", mod)
    cfg = json.loads((game / "config.json").read_text())
    cfg["wos"]["game"]["assets"]["external"]["path"] = game.parent.as_posix()
    cfg["wos"]["game"]["mods"]["loadPaths"][0] = {
        "location": "WORKING_DIRECTORY",
        "name": "unpackaged",
        "package": False,
        "path": "probe-mods",
    }
    (host / "config.json").write_text(json.dumps(cfg))
    lua = mod / "scripts/AutoDancer.lua"
    source = lua.read_text()
    source = source.replace(
        "local function buildObservation()",
        "local eligibilityDumped = false\nlocal function buildObservation()",
    )
    marker = "        result.map_bounds = currentMapBounds()"
    assert source.count(marker) == 1
    probe = PROBE
    if args.catalog_detail:
        probe = probe.replace(
            '            emit("complete",', CATALOG_PROBE + '\n            emit("complete",'
        )
    lua.write_text(source.replace(marker, probe + marker))
    manifest = {
        "gameplay_actions": 0,
        "training": False,
        "passed": False,
        "scope": "one Bard reset, current isolated production content configuration",
        "hashes": {
            str(p): sha256_file(p)
            for p in [
                Path(__file__),
                *sorted((mod / "scripts").glob("*.lua")),
                game / "NecroDancer.wsp",
                game / "versions/v4.2.1.wsp",
                game / "lua51.dll",
                host / "config.json",
                *sorted((host / "dlc").glob("*.necromod")),
            ]
        },
    }
    try:
        with PrivateSupervisor(
            SupervisorConfig(
                game_dir=host,
                mod_dir=mod,
                num_instances=1,
                startup_timeout=45,
                reset_timeout=30,
                affinity_policy="none",
                steam_presence_worker=0,
                profile_root=root / "profiles",
                diagnostic_root=root / "diagnostics",
            )
        ) as supervisor:
            env = supervisor.environment(supervisor.worker_ids[0])
            try:
                env.reset(seed=230001)
            finally:
                env.close()
                handle = supervisor.workers[supervisor.worker_ids[0]]
                log_path = handle.log_path
        # Shutdown flushes the final log records. Never parse inside the supervisor.
        log = log_path.read_text(encoding="utf-8", errors="strict")
        (root / "game.log").write_text(log, encoding="utf-8")
        rows = parse_catalog(log)
        (root / "records.json").write_text(json.dumps(rows, indent=2) + "\n")
        manifest.update(validate_records(rows))
        manifest["log_sha256"] = sha256_file(log_path)
        print(
            json.dumps({"passed": True, "items": manifest["items"], "owners": manifest["owners"]})
        )

    finally:
        manifest["profile_hashes"] = {
            str(p): sha256_file(p) for p in sorted((root / "profiles").rglob("*.json"))
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
