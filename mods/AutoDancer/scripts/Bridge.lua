-- Single-instance Python -> SYNCHRONY action bridge.
--
-- Python updates bridge-command.txt in the installed AutoDancer mod directory.
-- The command is addressed through the mod's logical asset path, so the engine's
-- unpacked-mod reloader sees external updates. This module injects exactly one engine
-- action, and exposes the completed command to AutoDancer.lua so the resulting
-- observation carries an unambiguous command acknowledgement.

local Action = require "necro.game.system.Action"
local CurrentLevel = require "necro.game.level.CurrentLevel"
local CharacterSelector = require "necro.client.CharacterSelector"
local FileIO = require "system.game.FileIO"
local GameSession = require "necro.client.GameSession"
local GameInput = require "necro.client.Input"
local Player = require "necro.game.character.Player"
local SinglePlayer = require "necro.client.SinglePlayer"
local StateControl = require "necro.client.StateControl"

local Bridge = {}

local COMMAND_RESOURCE = "mods/AutoDancer/bridge-command.txt"
local MAX_COMMAND_BYTES = 512

local LOGICAL_TO_ENGINE = {
    [0] = Action.Direction.UP,
    [1] = Action.Direction.RIGHT,
    [2] = Action.Direction.DOWN,
    [3] = Action.Direction.LEFT,
    [4] = Action.Special.IDLE,
    [5] = Action.Special.BOMB,
    [6] = Action.Special.ITEM_1,
    [7] = Action.Special.ITEM_2,
    [8] = Action.Special.THROW,
    [9] = Action.Special.SPELL_1,
    [10] = Action.Special.SPELL_2,
}

local lastPayload = nil
local pending = nil
local completed = nil

local function playable()
    return not CurrentLevel.isLoading()
        and not CurrentLevel.isLobby()
        and Player.getPlayerEntity(1) ~= nil
end

local function readPayload()
    local ok, payload = pcall(FileIO.readFileToString, COMMAND_RESOURCE, MAX_COMMAND_BYTES)
    if not ok or type(payload) ~= "string" or payload == "" then
        return nil
    end
    return payload
end

print("AUTODANCER_BRIDGE_READY:" .. COMMAND_RESOURCE)

local function parse(payload)
    local kind, sessionID, commandText, actionText = string.match(
        payload,
        "^([A-Z]+)%s+([%w%-_]+)%s+(%d+)%s*([%-]?%d*)%s*$"
    )
    if not kind then
        return nil
    end
    return kind, sessionID, tonumber(commandText), tonumber(actionText)
end

local function acceptAction(sessionID, commandID, logicalAction)
    local engineAction = LOGICAL_TO_ENGINE[logicalAction]
    if not engineAction or pending or not playable() then
        return false
    end
    pending = {
        session_id = sessionID,
        command_id = commandID,
        requested_action = logicalAction,
        engine_action = engineAction,
        observed_action = nil,
    }
    GameInput.add(engineAction, 1, {
        AutoDancer_session_id = sessionID,
        AutoDancer_command_id = commandID,
    })
    return true
end

local function startAllZonesBard()
    SinglePlayer.setActive(true)
    CharacterSelector.setPreferredCharacter(1, "Bard")
    CharacterSelector.setSelectedCharacter(1, "Bard")
    GameSession.start({mode = GameSession.Mode.AllZones}, 0)
end

event.clientAddInput.add("captureAutoDancerBridgeInput", {
    order = "entity",
    sequence = 1000,
}, function(ev)
    if not pending or ev.playerID ~= 1 or type(ev.args) ~= "table" then
        return
    end
    if ev.args.AutoDancer_session_id == pending.session_id
        and ev.args.AutoDancer_command_id == pending.command_id then
        pending.observed_action = ev.action
    end
end)

-- This must run immediately before AutoDancer's telemetry event.
event.turn.add("completeAutoDancerBridgeCommand", {
    order = "turnID",
    sequence = -2,
}, function()
    if pending then
        completed = pending
        pending = nil
    end
end)

event.tick.add("pollAutoDancerBridgeCommand", "input", function()
    local payload = readPayload()
    if not payload or payload == lastPayload then
        return
    end
    local kind, sessionID, commandID, logicalAction = parse(payload)
    local accepted = false
    if kind == "ACTION" and logicalAction ~= nil then
        accepted = acceptAction(sessionID, commandID, logicalAction)
    elseif kind == "START" and CurrentLevel.isLobby() then
        pending = nil
        completed = nil
        startAllZonesBard()
        accepted = true
    elseif kind == "RESTART" and StateControl.isRestartAllowed() then
        pending = nil
        completed = nil
        StateControl.restart(0)
        accepted = true
    elseif not kind then
        accepted = true
    end
    if accepted then
        lastPayload = payload
    end
end)

function Bridge.consumeCompletedCommand()
    local result = completed
    completed = nil
    return result
end

return Bridge
