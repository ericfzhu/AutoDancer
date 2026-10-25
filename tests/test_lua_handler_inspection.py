"""Offline compiler/introspection checks; no game executable or module is run."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def inspector(monkeypatch):
    dll = Path(r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64\lua51.dll")
    if not dll.is_file():
        pytest.skip("Installed LuaJIT DLL is unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    module = importlib.import_module("inspect_lua_handlers")
    return lambda source: module.inspect_chunk(source, dll)


def test_inspection_does_not_execute_target(inspector):
    report = inspector(b'error("TARGET_MUST_NOT_EXECUTE")')
    assert '"TARGET_MUST_NOT_EXECUTE"' in report
    assert "CALL" in report


def test_nested_constants_and_child_functions_are_inspected(inspector):
    report = inspector(b"return function() return {nested={active=false, amount=0}} end")
    assert report.count("FUNCTION ") == 2
    assert '["active"] = false' in report
    assert '["amount"] = 0' in report


def test_invalid_chunks_fail_explicitly(inspector):
    # LuaJIT uses a caught Windows SEH exception for parser errors. Exercise
    # that path outside pytest's native faulthandler, which mislabels it fatal.
    code = """
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from inspect_lua_handlers import inspect_chunk
try:
    inspect_chunk(b'not valid lua !', Path(sys.argv[1]))
except ValueError as error:
    assert str(error).startswith('INSPECT_ERROR:')
else:
    raise AssertionError('Malformed chunk was accepted')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            r"X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64\lua51.dll",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def runtime_evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    module = importlib.import_module("resolve_bard_runtime")
    flags = {name: 1 << i for i, name in enumerate(sorted(module.FLAGS))}
    rows = [
        {"name": "context", "ok": True, "data": {"flag_values": flags}},
        {
            "name": "item:StartingWeapon",
            "ok": True,
            "data": {
                "ban_mask": flags["GENERATE_ITEM_POOL"],
                "blocked": {name: name == "GENERATE_ITEM_POOL" for name in flags},
            },
        },
        {"name": "owners", "ok": True, "data": dict.fromkeys(module.ACTIONS, "none")},
    ]
    rows.extend(
        {"name": "handler_count:" + event, "ok": True, "data": {"count": 0}}
        for event in module.EVENTS
    )
    rows.append(
        {
            "name": "complete",
            "ok": True,
            "data": {
                "items": 1,
                "player_unchanged": True,
                "gameplay_actions": 0,
            },
        }
    )
    return module.validate_records, rows


def test_runtime_bans_preserve_acquisition_phase(runtime_evidence):
    validate, rows = runtime_evidence
    report = validate(rows)
    assert report["blocked_by_phase"]["GENERATE_ITEM_POOL"] == ["StartingWeapon"]
    assert report["blocked_by_phase"]["EQUIP"] == []
    assert report["blocked_by_phase"]["PICKUP"] == []


@pytest.mark.parametrize(
    "fault", ["truncated", "duplicate", "item_count", "handler_count", "mask", "flags", "actions"]
)
def test_runtime_evidence_rejects_incomplete_or_inconsistent_records(runtime_evidence, fault):
    validate, rows = runtime_evidence
    if fault == "truncated":
        rows.pop()
    elif fault == "duplicate":
        rows.insert(1, rows[0])
    elif fault == "item_count":
        rows[-1]["data"]["items"] = 2
    elif fault == "handler_count":
        rows[3]["data"]["count"] = 1
    elif fault == "mask":
        rows[1]["data"]["blocked"]["EQUIP"] = True
    elif fault == "flags":
        rows[0]["data"]["flag_values"].pop("EQUIP")
    else:
        rows[-1]["data"]["gameplay_actions"] = 1
    with pytest.raises(ValueError):
        validate(rows)
