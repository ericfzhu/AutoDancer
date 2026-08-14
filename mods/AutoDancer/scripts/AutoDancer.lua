-- AutoDancer live telemetry for an unpackaged local SYNCHRONY mod.
--
-- This script only uses the supported game API and print(). The game writes the
-- printed record to its normal debug log. Python tails that log. Set the two
-- build values before collecting a conformance trace.

local CurrentLevel = require "necro.game.level.CurrentLevel"
local Map = require "necro.game.object.Map"
local Player = require "necro.game.character.Player"
local Tile = require "necro.game.tile.Tile"
local Vision = require "necro.game.vision.Vision"
local Entities = require "system.game.Entities"

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

local function hasComponent(entity, component)
    return Entities.typeHasComponent(entity.name, component) == true
end

local function actorKind(entity)
    if not (hasComponent(entity, "character") and hasComponent(entity, "health"))
        or hasComponent(entity, "playableCharacter") then
        return 0
    end
    local name = string.lower(entity.name or "")
    if name == "slime" or string.find(name, "greenslime", 1, true) then
        return 2
    elseif name == "slime2" or string.find(name, "blueslime", 1, true) then
        return 3
    elseif string.find(name, "skeleton", 1, true) then
        return 4
    elseif string.find(name, "bat", 1, true) then
        return 5
    elseif string.find(name, "armadillo", 1, true) then
        return 6
    elseif string.find(name, "warlock", 1, true) then
        return 7
    elseif string.find(name, "blademaster", 1, true) then
        return 8
    elseif string.find(name, "zombie", 1, true) then
        return 10
    elseif hasComponent(entity, "boss") then
        return 9
    end
    return 0
end

local function encodeVisibleEntities(x, y, cell)
    for _, entityID in ipairs(Map.getAll(x, y)) do
        local entity = Entities.getEntityByID(entityID)
        if entity then
            local name = string.lower(entity.name or "")
            local actor = actorKind(entity)
            if actor ~= 0 then
                cell[2] = actor
                cell[3] = entity.health.health or 0
            end
            if hasComponent(entity, "trap") and string.find(name, "spike", 1, true) then
                cell[5] = 1
            end
            if string.find(name, "stairs", 1, true) then
                cell[1] = 3
            end
        end
    end
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
        for row = 1, GRID_SIZE do
            for column = 1, GRID_SIZE do
                local x = player.position.x + column - centre
                local y = player.position.y + row - centre
                local visible = Vision.isVisible(x, y)
                local revealed = visible or Vision.isRevealed(x, y)
                if revealed and Tile.exists(x, y) then
                    grid[row][column][1] = Tile.isSolid(x, y) and 2 or 1
                    grid[row][column][6] = visible and 2 or 1
                    if visible then
                        encodeVisibleEntities(x, y, grid[row][column])
                    end
                end
            end
        end

        grid[centre][centre][2] = 1 -- player
        grid[centre][centre][3] = health

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

local function buildEntityDebug()
    local player = Player.getPlayerEntity(1)
    local result = {}
    if not (player and player.position) then
        return result
    end
    local centre = math.floor(GRID_SIZE / 2)
    for y = player.position.y - centre, player.position.y + centre do
        for x = player.position.x - centre, player.position.x + centre do
            if Vision.isVisible(x, y) then
                for _, entityID in ipairs(Map.getAll(x, y)) do
                    local entity = Entities.getEntityByID(entityID)
                    local components = {}
                    for key, _ in pairs(entity and entity._components or {}) do
                        components[#components + 1] = tostring(key)
                    end
                    table.sort(components)
                    result[#result + 1] = {
                        id = entity and entity.id or entityID,
                        name = entity and entity.name or "",
                        x = x,
                        y = y,
                        components = components,
                        character = entity and hasComponent(entity, "character") or false,
                        playable = entity and hasComponent(entity, "playableCharacter") or false,
                        trap = entity and hasComponent(entity, "trap") or false,
                        item = entity and hasComponent(entity, "item") or false,
                        currency = entity and hasComponent(entity, "currency") or false,
                        health = entity and hasComponent(entity, "health")
                            and entity.health.health or 0,
                        max_health = entity and hasComponent(entity, "health")
                            and entity.health.maxHealth or 0,
                    }
                end
            end
        end
    end
    return result
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
        debug_entities = buildEntityDebug(),
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
