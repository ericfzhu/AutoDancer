-- Diagnosable Python -> SYNCHRONY controller bridge.

local Action = require "necro.game.system.Action"
local CharacterSelector = require "necro.client.CharacterSelector"
local ClientActionBuffer = require "necro.client.ClientActionBuffer"
local CurrentLevel = require "necro.game.level.CurrentLevel"
local Cutscene = require "necro.client.Cutscene"
local GameClient = require "necro.client.GameClient"
local GameInput = require "necro.client.Input"
local GameSession = require "necro.client.GameSession"
local Netplay = require "necro.network.Netplay"
local NetRNG = require "necro.client.NetRNG"
local Player = require "necro.game.character.Player"
local PlayerList = require "necro.client.PlayerList"
local Resources = require "necro.client.Resources"
local SinglePlayer = require "necro.client.SinglePlayer"
local Native = require "system.game.AutoDancerNative"

local Bridge = {}

local SCHEMA_VERSION = 10
local GAME_VERSION = "v4.2.1-b5713"
local STEAM_BUILD = "22938426"
local MAX_COMMAND_BYTES = 512
local STARTUP_STABLE_TICKS = 360
local HEARTBEAT_TICKS = 60
local ACTION_WATCHDOG_TICKS = 600
local RESET_WATCHDOG_TICKS = 2700
local CURRICULUM_PROFILES = {
    ["normal"] = true,
    ["player20"] = true,
    ["player10"] = true,
    ["player8"] = true,
    ["player6"] = true,
}

local instanceID = tostring(Native.getInstanceID() or "worker-unknown")
instanceID = string.gsub(instanceID, "[^%w%-_]", "_")
local launchID = tostring(Native.getLaunchID() or "launch-unknown")
launchID = string.gsub(launchID, "[^%w%-_]", "_")
local supervisorSession = tostring(Native.getSupervisorSession() or "session-unknown")
supervisorSession = string.gsub(supervisorSession, "[^%w%-_]", "_")
local qualificationMode = Native.isQualification() == true

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

local queuedCommand = nil
local pending = nil
local completed = nil
local startupStableTicks = 0
local bridgeReady = false
local tickCount = 0
local lastHeartbeatTick = 0
local lastDeferredReason = nil

local function jsonEscape(value)
    local escaped = string.gsub(tostring(value or ""), "[\\\"\b\f\n\r\t]", {
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

local function boolean(value)
    return value and "true" or "false"
end

local function levelIdentity()
    return tostring(CurrentLevel.getSeed())
        .. ":" .. tostring(CurrentLevel.getUniqueID())
        .. ":" .. tostring(CurrentLevel.getSequentialNumber())
end

local function engineState()
    return "{\"tick\":" .. tostring(tickCount)
        .. ",\"loading\":" .. boolean(CurrentLevel.isLoading())
        .. ",\"lobby\":" .. boolean(CurrentLevel.isLobby())
        .. ",\"cutscene\":" .. boolean(Cutscene.isActive())
        .. ",\"player_available\":" .. boolean(Player.getPlayerEntity(1) ~= nil)
        .. ",\"single_player\":" .. boolean(SinglePlayer.isActive())
        .. ",\"logged_in\":" .. boolean(GameClient.isLoggedIn())
        .. ",\"resources_ready\":" .. boolean(Resources.isResourceListReady())
        .. ",\"transfers_done\":" .. boolean(Resources.allTransfersDone())
        .. ",\"level_identity\":" .. jsonEscape(levelIdentity()) .. "}"
end

local function sendFrame(payload)
    local ok = Native.send(payload)
    if not ok then
        print("AUTODANCER_FATAL:{\"launch_id\":" .. jsonEscape(launchID)
            .. ",\"reason\":\"telemetry_pipe_write_failed\"}")
    end
    return ok
end

local function baseFrame(messageType)
    return "{\"message_type\":" .. jsonEscape(messageType)
        .. ",\"schema_version\":" .. tostring(SCHEMA_VERSION)
        .. ",\"instance_id\":" .. jsonEscape(instanceID)
        .. ",\"role\":\"worker\""
        .. ",\"session_id\":" .. jsonEscape(supervisorSession)
        .. ",\"launch_id\":" .. jsonEscape(launchID)
end

local function sendHello()
    local payload = baseFrame("hello")
        .. ",\"game_version\":" .. jsonEscape(GAME_VERSION)
        .. ",\"steam_build\":" .. jsonEscape(STEAM_BUILD)
        .. ",\"engine_state\":" .. engineState() .. "}"
    assert(sendFrame(payload), "AutoDancer HELLO pipe write failed")
    print("AUTODANCER_READY:" .. payload)
end

local function sendCommandStatus(command, phase, reason)
    if not command then
        return
    end
    local payload = baseFrame("command_status")
        .. ",\"command_kind\":" .. jsonEscape(command.kind)
        .. ",\"command_id\":" .. tostring(command.command_id)
        .. ",\"command_session_id\":" .. jsonEscape(command.session_id)
        .. ",\"phase\":" .. jsonEscape(phase)
        .. ",\"reason\":" .. jsonEscape(reason or "")
        .. ",\"requested_action\":" .. tostring(command.requested_action or -1)
        .. ",\"curriculum_profile\":" .. jsonEscape(command.curriculum_profile or "")
        .. ",\"engine_state\":" .. engineState() .. "}"
    assert(sendFrame(payload), "AutoDancer command-status pipe write failed")
end

local function parsePayload(payload)
    local kind, sessionID, commandText, argument = string.match(
        payload,
        "^([A-Z_]+)%s+([%w%-_]+)%s+(%d+)%s+([%w%-_]*)%s*$"
    )
    if not kind then
        return nil
    end
    return {
        kind = kind,
        session_id = sessionID,
        command_id = tonumber(commandText),
        argument = argument,
        requested_action = kind == "ACTION" and tonumber(argument) or nil,
        received_tick = tickCount,
    }
end

local function readCommand()
    if queuedCommand then
        return queuedCommand
    end
    local payload = Native.poll()
    if type(payload) ~= "string" or payload == "" or #payload > MAX_COMMAND_BYTES then
        return nil
    end
    queuedCommand = parsePayload(payload)
    if queuedCommand then
        lastDeferredReason = nil
        sendCommandStatus(queuedCommand, "received")
    else
        print("AUTODANCER_FATAL:{\"launch_id\":" .. jsonEscape(launchID)
            .. ",\"reason\":\"malformed_command\"}")
    end
    return queuedCommand
end

local function playableReason()
    if CurrentLevel.isLoading() then
        return "level_loading"
    elseif CurrentLevel.isLobby() then
        return "lobby"
    elseif Player.getPlayerEntity(1) == nil then
        return "missing_player"
    elseif pending then
        return "command_pending"
    end
    return nil
end

local function resetReason(seed)
    if not bridgeReady then
        return "bridge_not_ready"
    elseif not seed then
        return "invalid_seed"
    elseif CurrentLevel.isLoading() then
        return "level_loading"
    end
    Cutscene.skipStartupCutscenes()
    if Cutscene.isActive() then
        Cutscene.skip(true)
        return "cutscene"
    elseif not SinglePlayer.isActive() then
        SinglePlayer.init({gameState = Netplay.GameState.UNINITIALIZED})
        return "initializing_single_player"
    elseif not GameClient.isLoggedIn() then
        return "client_not_logged_in"
    elseif not Resources.isResourceListReady() then
        return "resource_list_not_ready"
    elseif not Resources.allTransfersDone() then
        return "resource_transfer_pending"
    end
    return nil
end

local function startAllZonesBard(seed)
    local playerID = PlayerList.getLocalPlayerID() or PlayerList.getHostPlayerID() or 1
    CharacterSelector.setPreferredCharacter(1, "Bard")
    CharacterSelector.setSelectedCharacter(playerID, "Bard")
    PlayerList.setAttribute(Netplay.PlayerAttribute.READY, true)
    PlayerList.setAttribute(Netplay.PlayerAttribute.CHARACTER, "Bard")
    NetRNG.setSeed(seed)
    GameSession.start({
        modeID = GameSession.Mode.AllZonesSeeded,
        seed = seed,
        preserveSeed = true,
        initialCharacters = {[playerID] = "Bard"},
        primaryPlayerID = playerID,
    }, 0)
end

local function acceptCommand(command)
    if command.kind == "ACTION" then
        local engineAction = LOGICAL_TO_ENGINE[command.requested_action]
        local reason = not engineAction and "invalid_action" or playableReason()
        if reason then
            return false, reason
        end
        pending = {
            kind = "ACTION",
            session_id = command.session_id,
            command_id = command.command_id,
            requested_action = command.requested_action,
            engine_action = engineAction,
            observed_action = nil,
            accepted_tick = tickCount,
        }
        local inserted = GameInput.add(engineAction, 1, {
            AutoDancer_session_id = command.session_id,
            AutoDancer_command_id = command.command_id,
        })
        if not inserted then
            pending = nil
            return false, "input_buffer_unavailable"
        end
        pending.observed_action = engineAction
        sendCommandStatus(pending, "accepted")
        sendCommandStatus(pending, "input_observed")
        return true
    elseif command.kind == "RESET" then
        local seed = tonumber(command.argument)
        local reason = resetReason(seed)
        if reason then
            return false, reason
        end
        pending = nil
        completed = {
            kind = "RESET",
            session_id = command.session_id,
            command_id = command.command_id,
            seed = seed,
        }
        sendCommandStatus(command, "accepted")
        sendCommandStatus(command, "reset_started")
        startAllZonesBard(seed)
        return true
    elseif command.kind == "GOTO" then
        local targetText, curriculumProfile = string.match(
            command.argument,
            "^(%d+)_([%w%-]+)$"
        )
        local targetLevel = tonumber(targetText or command.argument)
        local reason = not qualificationMode and "qualification_disabled"
            or (not targetLevel or targetLevel < 1) and "invalid_level"
            or (curriculumProfile and not CURRICULUM_PROFILES[curriculumProfile])
                and "invalid_curriculum_profile"
            or targetLevel ~= CurrentLevel.getSequentialNumber() + 1
                and "nonsequential_qualification_level"
            or playableReason()
        if reason then
            return false, reason
        end
        completed = {
            kind = "GOTO",
            session_id = command.session_id,
            command_id = command.command_id,
            target_level = targetLevel,
            curriculum_profile = curriculumProfile,
        }
        sendCommandStatus(command, "accepted")
        sendCommandStatus(command, "reset_started")
        GameSession.nextLevel(0)
        return true
    end
    return false, "unsupported_command"
end

local function clearQueued()
    queuedCommand = nil
    lastDeferredReason = nil
end

event.clientAddInput.add("assignAutoDancerBridgeTurnID", {
    order = "turnID",
    sequence = -1000,
}, function(ev)
    if ev.playerID ~= 1 or type(ev.args) ~= "table" then
        return
    end
    if ev.args.AutoDancer_session_id == supervisorSession
        and ev.args.AutoDancer_command_id then
        ev.turnID = ClientActionBuffer.findAvailableTurnID(ev.playerID)
    end
end)

event.turn.add("completeAutoDancerBridgeCommand", {
    order = "turnID",
    sequence = -2,
}, function()
    if pending then
        sendCommandStatus(pending, "turn_completed")
        completed = pending
        pending = nil
    end
end)

event.tick.add("pollAutoDancerBridgeCommand", "input", function()
    tickCount = tickCount + 1
    if not bridgeReady then
        if CurrentLevel.isLoading() then
            startupStableTicks = 0
            return
        end
        startupStableTicks = startupStableTicks + 1
        if startupStableTicks < STARTUP_STABLE_TICKS then
            return
        end
        bridgeReady = true
        sendHello()
    end

    local command = readCommand()
    if command then
        local accepted, reason = acceptCommand(command)
        if accepted then
            clearQueued()
        elseif reason ~= lastDeferredReason then
            lastDeferredReason = reason
            sendCommandStatus(command, "deferred", reason)
        end
        if queuedCommand
            and tickCount - queuedCommand.received_tick >= RESET_WATCHDOG_TICKS then
            sendCommandStatus(queuedCommand, "command_error", reason or "accept_timeout")
            clearQueued()
        end
    end

    local outstanding = pending or queuedCommand
    if outstanding and tickCount - lastHeartbeatTick >= HEARTBEAT_TICKS then
        lastHeartbeatTick = tickCount
        sendCommandStatus(outstanding, "heartbeat", lastDeferredReason)
    end
    if pending and tickCount - pending.accepted_tick >= ACTION_WATCHDOG_TICKS then
        sendCommandStatus(pending, "command_error", "accepted_action_no_turn")
        pending = nil
    end
end)

function Bridge.consumeCompletedCommand(allowReset)
    if completed and completed.kind == "RESET" and not allowReset then
        return nil
    end
    if completed and completed.kind == "GOTO"
        and CurrentLevel.getSequentialNumber() ~= completed.target_level then
        return nil
    end
    local result = completed
    completed = nil
    return result
end

function Bridge.markTelemetrySent(command)
    sendCommandStatus(command, "telemetry_sent")
end

function Bridge.getInstanceID()
    return instanceID
end

function Bridge.getLaunchID()
    return launchID
end

function Bridge.getSupervisorSession()
    return supervisorSession
end

function Bridge.getRole()
    return "worker"
end

return Bridge
