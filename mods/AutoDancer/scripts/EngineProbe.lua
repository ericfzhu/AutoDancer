-- Phase-one benchmark for driving the real engine without Windows key injection.
--
-- F6: direct client Input.add(), normal game-loop pacing
-- F7: ClientActionBuffer.addAction() + Turn.process(), accelerated turn stepping
-- F5: stop the active probe
--
-- Probe records are printed after AutoDancer's normal turn telemetry so the
-- Python collector can pair each acknowledgement with the immediately preceding
-- full observation.

local EngineProbe = {}

local Action = require "necro.game.system.Action"
local ClientActionBuffer = require "necro.client.ClientActionBuffer"
local CurrentLevel = require "necro.game.level.CurrentLevel"
local CustomActions = require "necro.game.data.CustomActions"
local GameInput = require "necro.client.Input"
local Player = require "necro.game.character.Player"
local Timer = require "system.utils.Timer"
local Turn = require "necro.cycles.Turn"

local PROBE_MARKER = "AUTODANCER_PROBE:"
local PROBE_SCHEMA_VERSION = 1
local DEFAULT_COMMANDS = 256
local PROCESS_BATCH_SIZE = 16
local COMMAND_TIMEOUT_SECONDS = 2.0

-- AutoDancer logical actions: up=0, right=1, down=2, left=3.
local LOGICAL_TO_ENGINE = {
    [0] = Action.Direction.UP,
    [1] = Action.Direction.RIGHT,
    [2] = Action.Direction.DOWN,
    [3] = Action.Direction.LEFT,
}

-- A deterministic oscillating script. It is intentionally movement-only so the
-- same sequence can be used in both normal and accelerated benchmark modes.
local ACTION_SCRIPT = {1, 3, 0, 2}

local probeCounter = 0
local state = {
    active = false,
    awaiting = false,
    mode = "idle",
    status = "idle",
    probeID = "",
    playerID = 0,
    targetCommands = DEFAULT_COMMANDS,
    commandID = 0,
    completedCommands = 0,
    startedAt = 0,
    pending = nil,
}

local function jsonEscape(value)
    local escaped = string.gsub(value, "[\\\"\b\f\n\r\t]", {
        ["\\"] = "\\\\",
        ["\""] = "\\\"",
        ["\b"] = "\\b",
        ["\f"] = "\\f",
        ["\n"] = "\\n",
        ["\r"] = "\\r",
        ["\t"] = "\\t",
    })
    return "\"" .. escaped .. "\""
end

local function isArray(value)
    local count = 0
    for key, _ in pairs(value) do
        if type(key) ~= "number" then
            return false
        end
        count = count + 1
    end
    return count == #value
end

local function jsonEncode(value)
    local valueType = type(value)
    if valueType == "nil" then
        return "null"
    elseif valueType == "boolean" then
        return value and "true" or "false"
    elseif valueType == "number" then
        return tostring(value)
    elseif valueType == "string" then
        return jsonEscape(value)
    elseif valueType ~= "table" then
        error("Unsupported JSON type: " .. valueType)
    end

    local parts = {}
    if isArray(value) then
        for index = 1, #value do
            parts[#parts + 1] = jsonEncode(value[index])
        end
        return "[" .. table.concat(parts, ",") .. "]"
    end
    for key, item in pairs(value) do
        parts[#parts + 1] = jsonEscape(tostring(key)) .. ":" .. jsonEncode(item)
    end
    table.sort(parts)
    return "{" .. table.concat(parts, ",") .. "}"
end

local function emit(kind, values)
    local now = Timer.getGlobalTime()
    local record = {
        schema_version = PROBE_SCHEMA_VERSION,
        kind = kind,
        probe_id = state.probeID,
        mode = state.mode,
        status = state.status,
        target_commands = state.targetCommands,
        completed_commands = state.completedCommands,
        timestamp = now,
        probe_elapsed_seconds = state.startedAt > 0 and now - state.startedAt or 0,
    }
    for key, value in pairs(values or {}) do
        record[key] = value
    end
    print(PROBE_MARKER .. jsonEncode(record))
end

local function gameReady(playerID)
    return not CurrentLevel.isLoading()
        and not CurrentLevel.isLobby()
        and not GameInput.isBlocked()
        and not GameInput.isPlayerInputBlocked()
        and Player.getPlayerEntity(playerID) ~= nil
end

local function finish(status, reason, details)
    if state.status == "idle" then
        return
    end
    state.active = false
    state.awaiting = false
    state.status = status
    local values = details or {}
    values.reason = reason
    emit(status == "error" and "error" or "finish", values)
    state.pending = nil
end

local function start(mode, playerID, targetCommands)
    if state.active then
        finish("stopped", "restarted_by_hotkey")
    end
    if not gameReady(playerID) then
        state.mode = mode
        state.status = "error"
        state.probeID = "unstarted"
        state.targetCommands = targetCommands or DEFAULT_COMMANDS
        state.completedCommands = 0
        state.startedAt = Timer.getGlobalTime()
        emit("error", {reason = "game_not_ready"})
        state.status = "idle"
        return false
    end

    probeCounter = probeCounter + 1
    local now = Timer.getGlobalTime()
    state.active = true
    state.awaiting = false
    state.mode = mode
    state.status = "running"
    state.probeID = string.format("%.6f:%d", now, probeCounter)
    state.playerID = playerID
    state.targetCommands = targetCommands or DEFAULT_COMMANDS
    state.commandID = 0
    state.completedCommands = 0
    state.startedAt = now
    state.pending = nil
    emit("start", {
        action_script = ACTION_SCRIPT,
        process_batch_size = mode == "process" and PROCESS_BATCH_SIZE or 1,
    })
    return true
end

local function nextLogicalAction()
    local index = state.commandID % #ACTION_SCRIPT + 1
    return ACTION_SCRIPT[index]
end

local function issueCommand()
    local logicalAction = nextLogicalAction()
    local engineAction = LOGICAL_TO_ENGINE[logicalAction]
    local now = Timer.getGlobalTime()
    local turnBefore = Turn.getCurrentTurnID()

    state.commandID = state.commandID + 1
    state.awaiting = true
    state.pending = {
        command_id = state.commandID,
        requested_action = logicalAction,
        engine_action = engineAction,
        observed_action = nil,
        turn_before = turnBefore,
        issued_at = now,
        buffer_has_action = false,
    }
    local args = {
        AutoDancer_probe_id = state.probeID,
        AutoDancer_command_id = state.commandID,
    }

    if state.mode == "input" then
        GameInput.add(engineAction, state.playerID, args)
        return
    end

    ClientActionBuffer.addAction(state.playerID, turnBefore, engineAction, 0, args)
    state.pending.buffer_has_action = ClientActionBuffer.hasAction(state.playerID, turnBefore)

    -- This is the same direct stepping pattern used by an existing Synchrony
    -- tool: insert the action at the current turn, then process the engine.
    Turn.process()
    if state.awaiting then
        finish("error", "turn_process_returned_without_turn", {
            command_id = state.commandID,
            turn_before = turnBefore,
        })
    end
end

-- Observe the action after client-side input redirection/transformation.
event.clientAddInput.add("captureAutoDancerProbeInput", {
    order = "entity",
    sequence = 1000,
}, function(ev)
    local pending = state.pending
    if not (state.active and state.awaiting and pending) then
        return
    end
    local args = ev.args
    if ev.playerID == state.playerID
        and type(args) == "table"
        and args.AutoDancer_probe_id == state.probeID
        and args.AutoDancer_command_id == pending.command_id then
        pending.observed_action = ev.action
    end
end)

-- AutoDancer's main exporter emits at turnID sequence -1. Sequence 0 here is
-- deliberately later, making each probe acknowledgement follow its matching
-- full AUTODANCER_JSON observation in the log.
event.turn.add("acknowledgeAutoDancerProbeTurn", {
    order = "turnID",
    sequence = 0,
}, function()
    local pending = state.pending
    if not (state.active and state.awaiting and pending) then
        return
    end

    local now = Timer.getGlobalTime()
    local turnAfter = Turn.getCurrentTurnID()
    state.completedCommands = state.completedCommands + 1
    state.awaiting = false
    state.pending = nil

    emit("turn", {
        command_id = pending.command_id,
        requested_action = pending.requested_action,
        engine_action = pending.engine_action,
        observed_action = pending.observed_action,
        turn_before = pending.turn_before,
        turn_after = turnAfter,
        turn_delta = turnAfter - pending.turn_before,
        issued_at = pending.issued_at,
        completed_at = now,
        command_elapsed_seconds = now - pending.issued_at,
        buffer_has_action = pending.buffer_has_action,
    })

    if state.completedCommands >= state.targetCommands then
        finish("completed", "target_reached")
    end
end)

event.tick.add("driveAutoDancerEngineProbe", "input", function()
    if not state.active then
        return
    end
    if state.awaiting then
        local pending = state.pending
        if pending and Timer.getGlobalTime() - pending.issued_at > COMMAND_TIMEOUT_SECONDS then
            finish("error", "command_timeout", {command_id = pending.command_id})
        end
        return
    end
    if not gameReady(state.playerID) then
        finish("stopped", "game_became_unavailable")
        return
    end

    local iterations = state.mode == "process" and PROCESS_BATCH_SIZE or 1
    for _ = 1, iterations do
        if not state.active or state.awaiting then
            break
        end
        local ok, errorMessage = pcall(issueCommand)
        if not ok then
            finish("error", "command_exception", {message = tostring(errorMessage)})
            break
        end
    end
end)

local function canStart(playerID)
    return gameReady(playerID)
end

CustomActions.registerHotkey {
    id = "engineProbeStop",
    name = "Stop AutoDancer engine probe",
    keyBinding = "F5",
    perPlayerBinding = true,
    callback = function()
        if state.active then
            finish("stopped", "stopped_by_hotkey")
        end
    end,
}

CustomActions.registerHotkey {
    id = "engineProbeInput",
    name = "Start AutoDancer normal-input probe",
    keyBinding = "F6",
    perPlayerBinding = true,
    enableIf = canStart,
    callback = function(playerID)
        start("input", playerID, DEFAULT_COMMANDS)
    end,
}

CustomActions.registerHotkey {
    id = "engineProbeProcess",
    name = "Start AutoDancer direct-turn probe",
    keyBinding = "F7",
    perPlayerBinding = true,
    enableIf = canStart,
    callback = function(playerID)
        start("process", playerID, DEFAULT_COMMANDS)
    end,
}

EngineProbe.start = start
EngineProbe.stop = finish
EngineProbe.state = state

return EngineProbe
