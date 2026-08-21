"""Add the trusted AutoDancer native loader shim to a pinned WSP16 archive."""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SHA256 = "e081898fda1f85b4908224aa0c3cc5a4243b853ba0b7920dbf228d824c9e8836"
STANDARD_LIBRARY_PATH = "scripts/core/StandardLibrary.lua"
EXPORT_SLOT_PATH = "scripts/system/utils/serial/MessagePack.lua"
EXPORT_MODULE_PATH = "scripts/system/game/AutoDancerNative.lua"


@dataclass(frozen=True)
class Archive:
    header: bytes
    entries: dict[str, bytes]


def read_archive(path: Path) -> Archive:
    data = path.read_bytes()
    if not data.startswith(b"WSP16") or len(data) < 17:
        raise ValueError("not a WSP16 archive")
    offset = 9
    names: list[str] = []
    while offset + 2 <= len(data):
        length = struct.unpack_from("<H", data, offset)[0]
        if not 0 < length < 512 or offset + 2 + length > len(data):
            break
        encoded = data[offset + 2 : offset + 2 + length]
        if any(byte < 32 or byte > 126 for byte in encoded):
            break
        names.append(encoded.decode("ascii"))
        offset += 2 + length
    table_end = offset + 8 * len(names)
    if not names or table_end > len(data):
        raise ValueError("invalid WSP16 filename/offset table")
    positions = [
        struct.unpack_from("<Q", data, offset + 8 * index)[0]
        for index in range(len(names))
    ]
    if positions[0] != table_end or positions != sorted(positions):
        raise ValueError("invalid WSP16 entry offsets")
    entries: dict[str, bytes] = {}
    for index, name in enumerate(names):
        end = positions[index + 1] if index + 1 < len(positions) else len(data)
        entries[name] = data[positions[index] : end]
    return Archive(data[:9], entries)


def create_entry(name: str, payload: bytes) -> bytes:
    encoded_name = name.encode("ascii")
    compressed = zlib.compress(payload, level=9)
    return (
        struct.pack("<H", len(encoded_name))
        + encoded_name
        + b"\0"
        + b"\1"
        + struct.pack("<II", len(compressed) + 4, len(payload))
        + compressed
    )


def read_entry(entry: bytes) -> bytes:
    name_length = struct.unpack_from("<H", entry)[0]
    header = 2 + name_length + 1
    compressed = entry[header] == 1
    stored_size = struct.unpack_from("<I", entry, header + 1)[0]
    if stored_size < 4:
        raise ValueError("invalid WSP16 compressed size")
    compressed_size = stored_size - 4
    payload = entry[header + 9 : header + 9 + compressed_size]
    return zlib.decompress(payload) if compressed else payload


def write_archive(path: Path, archive: Archive) -> None:
    names = sorted(archive.entries)
    name_table = b"".join(struct.pack("<H", len(name)) + name.encode("ascii") for name in names)
    position = len(archive.header) + len(name_table) + 8 * len(names)
    positions: list[int] = []
    bodies: list[bytes] = []
    for name in names:
        positions.append(position)
        body = archive.entries[name]
        bodies.append(body)
        position += len(body)
    path.write_bytes(
        archive.header
        + name_table
        + b"".join(struct.pack("<Q", value) for value in positions)
        + b"".join(bodies)
    )


def _lua_bytes(value: bytes) -> str:
    return '"' + "".join(f"\\{byte:03d}" for byte in value) + '"'


def _compile(compiler: Path, lua_dll: Path, source: str, output: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="autodancer-wsp-") as temporary:
        source_path = Path(temporary) / "wrapper.lua"
        output_path = Path(temporary) / "wrapper.luac"
        source_path.write_text(source, encoding="utf-8")
        subprocess.run(
            [str(compiler), str(lua_dll), str(source_path), str(output_path)],
            check=True,
        )
        payload = output_path.read_bytes()
    output.write_bytes(payload)
    return payload


def patch_archive(
    source: Path,
    destination: Path,
    compiler: Path,
    lua_dll: Path,
) -> None:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SUPPORTED_SHA256:
        raise ValueError(
            f"unsupported NecroDancer.wsp SHA-256 {digest}; expected {SUPPORTED_SHA256}"
        )
    archive = read_archive(source)
    for path in (STANDARD_LIBRARY_PATH, EXPORT_SLOT_PATH):
        if path not in archive.entries:
            raise ValueError(f"supported bridge patch point is missing: {path}")
    entries = dict(archive.entries)
    standard_bytecode = read_entry(entries[STANDARD_LIBRARY_PATH])
    standard_source = "\n".join(
        (
            "local rawLoadstring, rawBridge, rawPackage = loadstring, bridge, package",
            f"local original = assert(rawLoadstring({_lua_bytes(standard_bytecode)}))",
            "local result = original()",
            "local nativeLoader, loadError = rawPackage.loadlib("
            '"autodancer_native.dll", "luaopen_autodancer_native")',
            "assert(nativeLoader, loadError)",
            "rawBridge.AutoDancerNative = nativeLoader()",
            "return result",
        )
    )
    export_source = "return bridge.AutoDancerNative"
    scratch = destination.parent / ".autodancer-wrapper.luac"
    try:
        standard_wrapper = _compile(compiler, lua_dll, standard_source, scratch)
        export_module = _compile(compiler, lua_dll, export_source, scratch)
    finally:
        scratch.unlink(missing_ok=True)
    entries[STANDARD_LIBRARY_PATH] = create_entry(STANDARD_LIBRARY_PATH, standard_wrapper)
    del entries[EXPORT_SLOT_PATH]
    entries[EXPORT_MODULE_PATH] = create_entry(EXPORT_MODULE_PATH, export_module)
    write_archive(destination, Archive(archive.header, entries))
    verified = read_archive(destination)
    if len(verified.entries) != len(archive.entries):
        raise ValueError("patched archive file count changed")
    for path in (STANDARD_LIBRARY_PATH, EXPORT_MODULE_PATH):
        if not read_entry(verified.entries[path]).startswith(b"\x1bLJ"):
            raise ValueError(f"patched archive verification failed for {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--compiler", type=Path, default=Path("native/build/compile_lua.exe")
    )
    parser.add_argument("--lua-dll", type=Path)
    arguments = parser.parse_args()
    lua_dll = arguments.lua_dll or arguments.source.parent / "lua51.dll"
    patch_archive(
        arguments.source,
        arguments.destination,
        arguments.compiler,
        lua_dll,
    )


if __name__ == "__main__":
    main()
