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
local pendingEvents = {}
local playerDead = false

local function isLocalPlayer(entity)
    local player = Player.getPlayerEntity(1)
    return player ~= nil and entity ~= nil and entity.id == player.id
end

local function queueEvent(kind, amount, entity)
    pendingEvents[#pendingEvents + 1] = {
        kind = kind,
        amount = amount or 0,
        entity_id = entity and entity.id or 0,
    }
end

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
    elseif name == "slime3" then
        return 11
    elseif string.find(name, "skeleton2", 1, true) then
        return 13
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
    elseif string.find(name, "monkey", 1, true) then
        return 12
    elseif string.find(name, "dragon", 1, true) then
        return 14
    elseif hasComponent(entity, "boss") then
        return 9
    end
    return 0
end

local function itemKind(entity)
    local name = string.lower(entity and entity.name or "")
    if string.find(name, "resourcecoin", 1, true) then
        return 1
    elseif string.find(name, "food", 1, true) then
        return 2
    elseif string.find(name, "bomb", 1, true) then
        return 3
    elseif string.find(name, "resourcediamond", 1, true) then
        return 4
    elseif string.find(name, "weapondagger", 1, true) then
        return 5
    elseif string.find(name, "weaponbroadsword", 1, true) then
        return 8
    elseif string.find(name, "shovel", 1, true) then
        return 6
    elseif entity and hasComponent(entity, "item") then
        return 7
    end
    return 0
end

local function encodeInventoryItem(inventory, row, entityID)
    local item = entityID and Entities.getEntityByID(entityID) or nil
    if not item then
        return
    end
    inventory[row][1] = itemKind(item)
    inventory[row][2] = hasComponent(item, "itemStack") and item.itemStack.quantity or 1
    inventory[row][3] = hasComponent(item, "weapon") and item.weapon.damage or 0
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
            if hasComponent(entity, "item") then
                cell[4] = itemKind(entity)
            end
            if hasComponent(entity, "trap") then
                if string.find(name, "spike", 1, true) then
                    cell[5] = 1
                elseif string.find(name, "bouncetrapright", 1, true) then
                    cell[5] = 2
                elseif string.find(name, "bouncetrapup", 1, true) then
                    cell[5] = 3
                elseif string.find(name, "bouncetrapleft", 1, true) then
                    cell[5] = 4
                elseif string.find(name, "bouncetrapdown", 1, true) then
                    cell[5] = 5
                elseif string.find(name, "tempodowntrap", 1, true) then
                    cell[5] = 6
                elseif string.find(name, "trapdoor", 1, true) then
                    cell[5] = 7
                end
            end
            if string.find(name, "stairs", 1, true) then
                cell[1] = 3
            end
            if string.find(name, "bomblit", 1, true) then
                cell[7] = 1
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

    -- Cardinal movement is always representable. Bard has no native wait input.
    -- Special actions stay masked
    -- until their inventory mappings have a matching conformance trace.
    for index = 1, 4 do
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
                    local tileInfo = Tile.getInfo(x, y) or {}
                    if tileInfo.name == "Stairs" or (tileInfo.descent or 0) > 0 then
                        grid[row][column][1] = 3
                    else
                        grid[row][column][1] = Tile.isSolid(x, y) and 2 or 1
                    end
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
        playerValues[3] = hasComponent(player, "goldCounter")
            and player.goldCounter.amount or 0
        playerValues[5] = player.position.x
        playerValues[6] = player.position.y
        playerValues[7] = CurrentLevel.getZone()
        playerValues[8] = CurrentLevel.getFloor()
        playerValues[9] = sequence

        local slots = player.inventory and player.inventory.itemSlots or {}
        encodeInventoryItem(inventory, 1, slots.weapon and slots.weapon[1])
        encodeInventoryItem(inventory, 2, slots.action and slots.action[1])
        encodeInventoryItem(inventory, 3, slots.action and slots.action[2])
        encodeInventoryItem(inventory, 4, slots.shovel and slots.shovel[1])
        encodeInventoryItem(inventory, 5, slots.spell and slots.spell[1])
        encodeInventoryItem(inventory, 6, slots.spell and slots.spell[2])
        encodeInventoryItem(inventory, 7, slots.bomb and slots.bomb[1])
        encodeInventoryItem(inventory, 8, slots.misc and slots.misc[1])

        playerValues[10] = inventory[7][2]
        playerValues[11] = inventory[1][3]
        mask[6] = inventory[7][1] ~= 0 and 1 or 0
        mask[7] = inventory[2][1] ~= 0 and 1 or 0
        mask[8] = inventory[3][1] ~= 0 and 1 or 0
        mask[9] = inventory[1][1] ~= 0 and 1 or 0
        mask[10] = inventory[5][1] ~= 0 and 1 or 0
        mask[11] = inventory[6][1] ~= 0 and 1 or 0
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
                    result[#result + 1] = {
                        id = entity and entity.id or entityID,
                        name = entity and entity.name or "",
                        x = x,
                        y = y,
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
    local kind = sequence == 0 and "reset" or "turn"
    if lastRunIdentity == nil
        or (CurrentLevel.getSequentialNumber() == 1 and runIdentity ~= lastRunIdentity) then
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
        events = pendingEvents,
        terminated = playerDead,
        truncated = false,
        metrics = {
            turns = sequence,
        },
    }
    print(LOG_MARKER .. jsonEncode(record))
    pendingEvents = {}
    playerDead = false
    sequence = sequence + 1
end

-- Capture only events involving the local player. Running after damage application
-- ensures suppression and armor have already finalized the event's damage value.
event.objectDealDamage.add("captureAutoDancerDamage", {
    order = "statistics",
    sequence = 100,
}, function(ev)
    if ev.suppressed or (ev.damage or 0) <= 0 then
        return
    end
    if isLocalPlayer(ev.entity) and actorKind(ev.victim) ~= 0 then
        queueEvent("enemy_damage", ev.damage, ev.victim)
    elseif isLocalPlayer(ev.victim) then
        queueEvent("player_damage", ev.damage, ev.entity)
    end
end)

event.objectKill.add("captureAutoDancerKill", {
    order = "statistics",
    sequence = 100,
}, function(ev)
    if isLocalPlayer(ev.entity) and actorKind(ev.victim) ~= 0 then
        queueEvent("enemy_kill", 1, ev.victim)
    elseif isLocalPlayer(ev.entity) then
        local name = string.lower(ev.victim and ev.victim.name or "")
        if string.find(name, "chest", 1, true) or string.find(name, "crate", 1, true) then
            queueEvent("container_opened", 1, ev.victim)
        end
    end
end)

event.objectDeath.add("captureAutoDancerPlayerDeath", {
    order = "statistics",
    sequence = 100,
}, function(ev)
    if isLocalPlayer(ev.entity) then
        playerDead = true
        queueEvent("failure", 1, ev.entity)
    end
end)

event.objectCurrency.add("captureAutoDancerCurrencyPickup", {
    order = "statistics",
    sequence = 100,
}, function(ev)
    if isLocalPlayer(ev.entity) and ev.item and (ev.difference or 0) > 0 then
        queueEvent("item_collected", ev.difference, ev.item)
    end
end)

-- turnID is the last stable order key in the supported turn event.
event.turn.add("emitAutoDancerTelemetry", {order = "turnID", sequence = -1}, emitTurn)
