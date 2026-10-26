"""Inspect registered world definitions during one private reset; no training."""

from pathlib import Path

import resolve_bard_runtime as runtime

from autodancer.experiments.provenance import sha256_file

WORLD_PROBE = r"""
            local defaults = require("necro.game.data.Components").getComponentDefaults()
            local worldNames = Entities.getEntityTypesWithComponents({"position"})
            table.sort(worldNames)
            for _,name in ipairs(worldNames) do
                local proto = Entities.getEntityPrototype(name)
                local components = {}
                for component,fields in pairs(defaults) do
                    if proto[component] ~= nil then
                        local values = {}
                        for field in pairs(fields) do
                            local value = proto[component][field]
                            local kind = type(value)
                            if kind == "number" or kind == "boolean" then values[field] = value
                            else values[field] = "<"..kind..">" end
                        end
                        components[component] = values
                    end
                end
                emit("world_entity:"..name,components)
            end
            local tiles = require("necro.game.tile.TileTypes").listTileNames()
            table.sort(tiles)
            emit("world_tiles",tiles)
            emit("world_complete",{entities=#worldNames,tiles=#tiles})
"""

validate_bard = runtime.validate_records


def validate_records(rows):
    report = validate_bard([r for r in rows if not r["name"].startswith("world_")])
    entities = [r for r in rows if r["name"].startswith("world_entity:")]
    by_name = {r["name"]: r["data"] for r in rows}
    tiles = by_name["world_tiles"]
    if len(by_name) != len(rows) or not all(r["ok"] for r in rows):
        raise ValueError("Duplicate or failed world definitions")
    if by_name["world_complete"] != {"entities": len(entities), "tiles": len(tiles)}:
        raise ValueError("Incomplete world catalog")
    if not entities or not tiles or len(set(tiles)) != len(tiles):
        raise ValueError("Empty or duplicate world catalog")
    report["world_entities"] = len(entities)
    report["world_tiles"] = len(tiles)
    report["world_inspector_sha256"] = sha256_file(Path(__file__))
    return report


if __name__ == "__main__":
    runtime.validate_records = validate_records
    runtime.PROBE = runtime.PROBE.replace(
        '            emit("complete",', WORLD_PROBE + '\n            emit("complete",'
    )
    runtime.main()
