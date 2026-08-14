-- AutoDancer live telemetry for an unpackaged local SYNCHRONY mod.
--
-- This script only uses the supported game API and print(). The game writes the
-- printed record to its normal debug log. Python tails that log. Set the two
-- build values before collecting a conformance trace.

local CurrentLevel = require "necro.game.level.CurrentLevel"
local Player = require "necro.game.character.Player"

local GRID_SIZE = 21
local GRID_CHANNELS = 7
local PLAYER_FEATURES = 16
local INVENTORY_SLOTS = 8
local INVENTORY_FEATURES = 3
local ACTION_COUNT = 11
local LOG_MARKER = "AUTODANCER_JSON:"

-- Replace these values with the values shown by the installed game and Steam.
local GAME_VERSION = "v4.2.1-b5713"
local STEAM_BUILD = "22938426"

local sequence = 0
local lastRunIdentity = ""

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

local function zeroVector(length)
    local result = {}
    for index = 1, length do
        result[index] = 0
    end
    return result
end

local function zeroMatrix(rows, columns)
    local result = {}
    for row = 1, rows do
        result[row] = zeroVector(columns)
    end
    return result
end

local function zeroGrid()
    local result = {}
    for row = 1, GRID_SIZE do
        result[row] = zeroMatrix(GRID_SIZE, GRID_CHANNELS)
    end
    return result
end

local function buildObservation(ev)
    local player = Player.getPlayerEntity(1)
    local grid = zeroGrid()
    local playerValues = zeroVector(PLAYER_FEATURES)
    local inventory = zeroMatrix(INVENTORY_SLOTS, INVENTORY_FEATURES)
    local mask = zeroVector(ACTION_COUNT)

    -- Movement and wait are always representable. Special actions stay masked
    -- until their inventory mappings have a matching conformance trace.
    for index = 1, 5 do
        mask[index] = 1
    end

    if player and player.position then
        local health = 0
        local maxHealth = 0
        if player.health then
            health = player.health.health or 0
            maxHealth = player.health.maxHealth or health
        end
        local centre = math.floor(GRID_SIZE / 2) + 1
        grid[centre][centre][1] = 1 -- floor
        grid[centre][centre][2] = 1 -- player
        grid[centre][centre][3] = health
        grid[centre][centre][6] = 2 -- visible now

        playerValues[1] = health
        playerValues[2] = maxHealth
        playerValues[5] = player.position.x
        playerValues[6] = player.position.y
        playerValues[7] = CurrentLevel.getZone()
        playerValues[8] = CurrentLevel.getFloor()
        playerValues[9] = sequence
    end

    return {
        grid = grid,
        player = playerValues,
        inventory = inventory,
        action_mask = mask,
    }
end

local function emitTurn(ev)
    if CurrentLevel.isLoading() or CurrentLevel.isLobby() then
        return
    end
    local runIdentity = tostring(CurrentLevel.getSeed())
        .. ":" .. tostring(CurrentLevel.getUniqueID())
        .. ":" .. tostring(CurrentLevel.getSequentialNumber())
    local kind = "turn"
    if CurrentLevel.getSequentialNumber() == 1 and runIdentity ~= lastRunIdentity then
        sequence = 0
        kind = "reset"
    end
    lastRunIdentity = runIdentity

    local record = {
        schema_version = 1,
        sequence = sequence,
        kind = kind,
        game = {
            version = GAME_VERSION,
            steam_build = STEAM_BUILD,
        },
        seed = CurrentLevel.getSeed(),
        character = Player.getCharacterType(1),
        zone = CurrentLevel.getZone(),
        floor = CurrentLevel.getFloor(),
        observation = buildObservation(ev),
        events = {},
        terminated = false,
        truncated = false,
        metrics = {
            turns = sequence,
        },
    }
    print(LOG_MARKER .. jsonEncode(record))
    sequence = sequence + 1
end

-- turnID is the last stable order key in the supported turn event.
event.turn.add("emitAutoDancerTelemetry", {order = "turnID", sequence = -1}, emitTurn)
