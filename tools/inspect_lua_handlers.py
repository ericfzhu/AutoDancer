"""Inspect LuaJIT chunks without invoking the inspected functions or game modules."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

from catalog_game_controls import archive_entries

from autodancer.live.qualify import _wsp_entry_payload

# LuaJIT 2.1 opcode prefix; funcbc supplies operand modes from the loaded runtime.
OPCODES = (
    "ISLT ISGE ISLE ISGT ISEQV ISNEV ISEQS ISNES ISEQN ISNEN ISEQP ISNEP "
    "ISTC ISFC IST ISF ISTYPE ISNUM MOV NOT UNM LEN ADDVN SUBVN "
    "MULVN DIVVN MODVN ADDNV SUBNV MULNV DIVNV MODNV ADDVV SUBVV MULVV DIVVV "
    "MODVV POW CAT KSTR KCDATA KSHORT KNUM KPRI KNIL UGET USETV USETS "
    "USETN USETP UCLO FNEW TNEW TDUP GGET GSET TGETV TGETS TGETB TGETR "
    "TSETV TSETS TSETB TSETM TSETR CALLM CALL CALLMT CALLT ITERC ITERN VARG "
    "ISNEXT RETM RET RET0 RET1 FORI JFORI FORL IFORL JFORL ITERL IITERL "
    "JITERL LOOP ILOOP JLOOP JMP "
).split()

INSPECTOR = r"""
local util = require("jit.util")
local bit = require("bit")
assert(jit.version == "LuaJIT 2.1.1748749768", "Unqualified LuaJIT runtime")
local check = assert(loadstring("return 123"))
assert(bit.band(util.funcbc(check,1),255) == 41, "KSHORT opcode mismatch")
assert(bit.band(util.funcbc(check,2),255) == 76, "RET1 opcode mismatch")
local out = {"runtime " .. jit.version}
local opnames = {OP_NAMES}
local function emit(s) out[#out+1] = s end
local function scalar(v)
  if type(v) == "string" then return string.format("%q", v) end
  return tostring(v)
end
local function tab(t, path, depth)
  assert(depth < 64, "constant table nesting limit")
  local keys = {}
  for k in pairs(t) do keys[#keys+1] = k end
  table.sort(keys, function(a,b) return scalar(a) < scalar(b) end)
  for _,k in ipairs(keys) do
    local v = t[k]
    local p = path .. "[" .. scalar(k) .. "]"
    if type(v) == "table" then emit(p .. " = {}"); tab(v,p,depth+1)
    else emit(p .. " = " .. scalar(v)) end
  end
end
local function inspect(f, path)
  local info = util.funcinfo(f)
  emit("FUNCTION " .. path .. " line=" .. tostring(info.linedefined))
  for i=0,info.gcconsts-1 do
    local v = util.funck(f,-i-1)
    local p = path .. ".K" .. i
    if type(v) == "proto" then inspect(v,p)
    elseif type(v) == "table" then emit(p .. " = {}"); tab(v,p,0)
    else emit(p .. " = " .. scalar(v)) end
  end
  for i=0,info.nconsts-1 do emit(path .. ".N" .. i .. " = " .. scalar(util.funck(f,i))) end
  for i=0,info.upvalues-1 do emit(path .. ".UV" .. i .. " = " .. tostring(util.funcuvname(f,i))) end
  for pc=1,info.bytecodes-1 do
    local ins, mode = util.funcbc(f,pc)
    local op = bit.band(ins,255)
    local a = bit.band(bit.rshift(ins,8),255)
    local b = bit.rshift(ins,24)
    local c = bit.band(bit.rshift(ins,16),255)
    local d = bit.rshift(ins,16)
    emit(string.format("%s %04d %-8s A=%d B=%d C=%d D=%d mode=%d",
      path,pc,opnames[op+1] or ("OP"..op),a,b,c,d,mode))
  end
end
local chunk, problem = loadstring(CHUNK_BYTES, "@inspected")
if not chunk then return "INSPECT_ERROR:" .. tostring(problem) end
inspect(chunk,"root")
return table.concat(out,"\n")
"""


def inspect_chunk(payload: bytes, dll: Path) -> str:
    lib = ctypes.CDLL(str(dll.resolve()))
    lib.luaL_newstate.restype = ctypes.c_void_p
    for name, args, result in (
        ("luaL_openlibs", [ctypes.c_void_p], None),
        (
            "luaL_loadbuffer",
            [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p],
            ctypes.c_int,
        ),
        ("lua_pcall", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        (
            "lua_tolstring",
            [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)],
            ctypes.c_void_p,
        ),
        ("lua_close", [ctypes.c_void_p], None),
    ):
        func = getattr(lib, name)
        func.argtypes, func.restype = args, result
    state = lib.luaL_newstate()
    if not state:
        raise RuntimeError("Lua allocation failed")
    try:
        lib.luaL_openlibs(state)
        encoded = '"' + "".join(f"\\{b:03d}" for b in payload) + '"'
        script = (
            INSPECTOR.replace("OP_NAMES", ",".join(json.dumps(n) for n in OPCODES))
            .replace("CHUNK_BYTES", encoded)
            .encode()
        )
        error = lib.luaL_loadbuffer(state, script, len(script), b"@offline-inspector")
        if not error:
            error = lib.lua_pcall(state, 0, 1, 0)
        size = ctypes.c_size_t()
        address = lib.lua_tolstring(state, -1, ctypes.byref(size))
        text = ctypes.string_at(address, size.value).decode() if address else "No Lua result"
        if error or text.startswith("INSPECT_ERROR:"):
            raise ValueError(text)
        return text
    finally:
        lib.lua_close(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Exact archived Lua path; repeat to inspect additional modules",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be fresh")
    definitions = json.loads(Path("runs/game-control-catalog-current/definitions.json").read_text())
    selected = [
        "scripts/necro/game/item/" + n + ".lua"
        for n in ("Weapon", "ActionItem", "MovementItem", "ItemBan")
    ]
    selected += [
        "scripts/necro/game/data/item/ItemBans.lua",
        "scripts/necro/game/data/player/PlayableCharacters.lua",
    ]
    selected += [
        "scripts/necro/game/object/ObjectEvents.lua",
        "scripts/necro/game/item/Inventory.lua",
    ]
    if args.modules:
        selected = sorted(set(args.modules))
    args.output.mkdir(parents=True)
    dll = args.game_dir / "lua51.dll"
    manifest = {
        "game_launched": False,
        "modules_executed": False,
        "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dll_sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
        "modules": [],
    }
    effective = {}
    for archive in (args.game_dir / "NecroDancer.wsp", args.game_dir / "versions/v4.2.1.wsp"):
        for name, entry in archive_entries(archive):
            if name in selected:
                effective[name] = (archive, _wsp_entry_payload(entry))
    for name, (archive, payload) in sorted(effective.items()):
        digest = hashlib.sha256(payload).hexdigest()
        previous = definitions.get(name)
        if previous is not None and digest != previous["sha256"]:
            raise ValueError("Saved definition mismatch: " + name)
        stem = Path(name).stem
        if sum(Path(path).stem == stem for path in selected) > 1:
            stem = name.removeprefix("scripts/").removesuffix(".lua").replace("/", "__")
        output = args.output / (stem + ".txt")
        output.write_text(inspect_chunk(payload, dll), encoding="utf-8")
        manifest["modules"].append(
            {
                "path": name,
                "sha256": digest,
                "archive": str(archive.relative_to(args.game_dir)),
                "output": output.name,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "matched_saved_definition": previous is not None,
            }
        )
    if len(manifest["modules"]) != len(selected):
        raise ValueError("Missing selected modules")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Inspected", len(selected), "modules without executing them")


if __name__ == "__main__":
    main()
