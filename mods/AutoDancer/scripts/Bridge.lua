-- Arbitrary-N Python -> SYNCHRONY bridge.

local Action = require "necro.game.system.Action"
local CharacterSelector = require "necro.client.CharacterSelector"
local CurrentLevel = require "necro.game.level.CurrentLevel"
local FileIO = require "system.game.FileIO"
local GameInput = require "necro.client.Input"
local GameSession = require "necro.client.GameSession"
local MultiInstance = require "necro.client.MultiInstance"
local Player = require "necro.game.character.Player"
local SinglePlayer = require "necro.client.SinglePlayer"

local Bridge = {}

local SCHEMA_VERSION = 4
local GAME_VERSION = "v4.2.1-b5713"
local STEAM_BUILD = "22938426"
local MAX_COMMAND_BYTES = 512
local assignmentResource = "mods/AutoDancer/bridge-assignment.txt"
local assignmentOK, assignment = pcall(FileIO.readFileToString, assignmentResource, 64)
local assignedID = assignmentOK and type(assignment) == "string"
    and string.match(assignment, "^%s*([%w%-_]+)%s*$") or nil
local sessionUID = MultiInstance.getSessionUID()
local instanceID = assignedID or sessionUID or "coordinator"
local isWorker = instanceID ~= "coordinator"
instanceID = tostring(instanceID or "worker-unknown")
instanceID = string.gsub(instanceID, "[^%w%-_]", "_")
local role = isWorker and "worker" or "coordinator"
local commandModuleID = string.gsub(instanceID, "%-", "_")
local commandModuleName = "AutoDancer.scripts.BridgeCommand_" .. commandModuleID
local Command = require(commandModuleName)
local commandResource = "scripts/BridgeCommand_" .. commandModuleID .. ".lua"

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
local spawnedInstances = {}

local function readPayload()
    local payload = Command.payload
    if type(payload) ~= "string" or payload == "" or #payload > MAX_COMMAND_BYTES then
        return nil
    end
    return payload
end

local function printReady()
    print("AUTODANCER_READY:{\"schema_version\":" .. tostring(SCHEMA_VERSION)
        .. ",\"instance_id\":\"" .. instanceID
        .. "\",\"role\":\"" .. role
        .. "\",\"game_version\":\"" .. GAME_VERSION
        .. "\",\"steam_build\":\"" .. STEAM_BUILD
        .. "\",\"command_resource\":\"" .. commandResource .. "\"}")
end

local function printCoordinatorResult(kind, sessionID, commandID, workerID, ok)
    print("AUTODANCER_COORDINATOR_JSON:{\"schema_version\":" .. tostring(SCHEMA_VERSION)
        .. ",\"kind\":\"" .. kind
        .. "\",\"session_id\":\"" .. sessionID
        .. "\",\"command_id\":" .. tostring(commandID)
        .. ",\"worker_id\":\"" .. tostring(workerID)
        .. "\",\"ok\":" .. tostring(ok == true) .. "}")
end

printReady()

local function playable()
    return not CurrentLevel.isLoading()
        and not CurrentLevel.isLobby()
        and Player.getPlayerEntity(1) ~= nil
end

local function acceptAction(sessionID, commandID, logicalAction)
    local engineAction = LOGICAL_TO_ENGINE[logicalAction]
    if not engineAction or pending or not playable() then
        return false
    end
    pending = {
        kind = "ACTION",
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

local function startAllZonesBard(seed)
    SinglePlayer.setActive(true)
    CharacterSelector.setPreferredCharacter(1, "Bard")
    CharacterSelector.setSelectedCharacter(1, "Bard")
    GameSession.start({mode = GameSession.Mode.AllZones, seed = seed}, 0)
end

local function acceptReset(sessionID, commandID, seed)
    if CurrentLevel.isLoading() or not seed then
        return false
    end
    pending = nil
    completed = {
        kind = "RESET",
        session_id = sessionID,
        command_id = commandID,
        seed = seed,
    }
    startAllZonesBard(seed)
    return true
end

local function spawnWorker(sessionID, commandID, workerID)
    if spawnedInstances[workerID] and spawnedInstances[workerID].isOpen() then
        printCoordinatorResult("SPAWN", sessionID, commandID, workerID, true)
        return true
    end
    local ok, instance = pcall(MultiInstance.create, {
        independent = true,
        external = true,
        uid = workerID,
        configName = "AutoDancer-" .. workerID .. ".lua",
        windowTitle = "AutoDancer " .. workerID,
    })
    if ok and instance then
        spawnedInstances[workerID] = instance
    end
    printCoordinatorResult("SPAWN", sessionID, commandID, workerID, ok and not not instance)
    return ok and not not instance
end

local function closeWorker(sessionID, commandID, workerID)
    local instance = spawnedInstances[workerID]
    local ok = instance ~= nil
    if instance then
        ok = pcall(instance.close)
        spawnedInstances[workerID] = nil
    end
    printCoordinatorResult("CLOSE", sessionID, commandID, workerID, ok)
    return true
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

    local kind, sessionID, commandText, argument = string.match(
        payload,
        "^([A-Z_]+)%s+([%w%-_]+)%s+(%d+)%s+([%w%-_]*)%s*$"
    )
    if not kind then
        lastPayload = payload
        return
    end

    local commandID = tonumber(commandText)
    local accepted = false
    if isWorker then
        if kind == "ACTION" then
            accepted = acceptAction(sessionID, commandID, tonumber(argument))
        elseif kind == "RESET" then
            accepted = acceptReset(sessionID, commandID, tonumber(argument))
        end
    else
        if kind == "SPAWN" then
            accepted = spawnWorker(sessionID, commandID, argument)
        elseif kind == "CLOSE" then
            accepted = closeWorker(sessionID, commandID, argument)
        end
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

function Bridge.getInstanceID()
    return instanceID
end

function Bridge.getRole()
    return role
end

return Bridge
