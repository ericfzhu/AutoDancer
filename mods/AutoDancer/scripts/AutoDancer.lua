-- AutoDancer live telemetry for an unpackaged local SYNCHRONY mod.
--
-- Python sends actions through Bridge.lua's native named pipe. The resulting turn
-- record is written to the normal debug log with the matching command ID.

local Bridge = require "AutoDancer.scripts.Bridge"

local CurrentLevel = require "necro.game.level.CurrentLevel"
local Map = require "necro.game.object.Map"
local Player = require "necro.game.character.Player"
local Tile = require "necro.game.tile.Tile"
local Vision = require "necro.game.vision.Vision"
local Entities = require "system.game.Entities"

local GRID_SIZE = 21
local GRID_CHANNELS = 11
local PLAYER_FEATURES = 16
local INVENTORY_SLOTS = 8
local INVENTORY_FEATURES = 4
local ACTION_COUNT = 11
local LOG_MARKER = "AUTODANCER_JSON:"
local SCHEMA_VERSION = 5

-- Replace these values with the values shown by the installed game and Steam.
local GAME_VERSION = "v4.2.1-b5713"
local STEAM_BUILD = "22938426"

local sequence = 0
local runCounter = 0
local activeRunID = ""
local lastLevelIdentity = ""
local pendingEvents = {}
local playerDead = false
local terminalEmitted = false
local lastObservation = nil
local lastContext = nil

local function isLocalPlayer(entity)
    local player = Player.getPlayerEntity(1)
    return player ~= nil and entity ~= nil and entity.id == player.id
end

local function queueEvent(kind, amount, entity, data)
    local value = {
        kind = kind,
        amount = amount or 0,
        entity_id = entity and entity.id or 0,
    }
    if data then
        value.data = data
    end
    pendingEvents[#pendingEvents + 1] = value
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

local function clone(value)
    if type(value) ~= "table" then
        return value
    end
    local result = {}
    for key, item in pairs(value) do
        result[key] = clone(item)
    end
    return result
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

local function emptyObservation()
    return {
        grid = zeroGrid(),
        player = zeroVector(PLAYER_FEATURES),
        inventory = zeroMatrix(INVENTORY_SLOTS, INVENTORY_FEATURES),
        action_mask = zeroVector(ACTION_COUNT),
    }
end

local function hasComponent(entity, component)
    return entity ~= nil and Entities.typeHasComponent(entity.name, component) == true
end

-- Deterministic across Lua processes and runs. Zero means "no type".
local function typeID(name)
    local hash = 0
    name = string.lower(name or "")
    for index = 1, #name do
        hash = (hash * 31 + string.byte(name, index)) % 4095
    end
    return #name > 0 and hash + 1 or 0
end

local function actorKind(entity)
    if not entity
        or not (hasComponent(entity, "character") and hasComponent(entity, "health"))
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
    return 15
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
    inventory[row][2] = typeID(item.name)
    inventory[row][3] = hasComponent(item, "itemStack") and item.itemStack.quantity or 1
    inventory[row][4] = hasComponent(item, "weapon") and item.weapon.damage or 0
end

local function encodeVisibleEntities(x, y, cell)
    for _, entityID in ipairs(Map.getAll(x, y)) do
        local entity = Entities.getEntityByID(entityID)
        if entity then
            local name = string.lower(entity.name or "")
            local actor = actorKind(entity)
            if actor ~= 0 then
                cell[3] = actor
                cell[4] = typeID(entity.name)
                cell[5] = entity.health.health or 0
                cell[6] = entity.health.maxHealth or cell[5]
            end
            if hasComponent(entity, "item") then
                cell[7] = itemKind(entity)
                cell[8] = typeID(entity.name)
            end
            if hasComponent(entity, "trap") then
                if string.find(name, "spike", 1, true) then
                    cell[9] = 1
                elseif string.find(name, "bouncetrapright", 1, true) then
                    cell[9] = 2
                elseif string.find(name, "bouncetrapup", 1, true) then
                    cell[9] = 3
                elseif string.find(name, "bouncetrapleft", 1, true) then
                    cell[9] = 4
                elseif string.find(name, "bouncetrapdown", 1, true) then
                    cell[9] = 5
                elseif string.find(name, "tempodowntrap", 1, true) then
                    cell[9] = 6
                elseif string.find(name, "trapdoor", 1, true) then
                    cell[9] = 7
                end
            end
            if string.find(name, "stairs", 1, true) then
                cell[1] = 3
                cell[2] = typeID(entity.name)
            end
            if string.find(name, "bomblit", 1, true) then
                cell[11] = 1
            end
        end
    end
end

local function buildObservation()
    local player = Player.getPlayerEntity(1)
    local result = emptyObservation()
    local grid = result.grid
    local playerValues = result.player
    local inventory = result.inventory
    local mask = result.action_mask

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
        local visibleEnemies = 0
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
                    grid[row][column][2] = typeID(tileInfo.name)
                    grid[row][column][10] = visible and 2 or 1
                    if visible then
                        encodeVisibleEntities(x, y, grid[row][column])
                        if grid[row][column][3] > 1 then
                            visibleEnemies = visibleEnemies + 1
                        end
                    end
                end
            end
        end

        grid[centre][centre][3] = 1
        grid[centre][centre][4] = typeID(player.name)
        grid[centre][centre][5] = health
        grid[centre][centre][6] = maxHealth

        playerValues[1] = health
        playerValues[2] = maxHealth
        playerValues[3] = hasComponent(player, "goldCounter")
            and player.goldCounter.amount or 0
        playerValues[5] = player.position.x
        playerValues[6] = player.position.y
        playerValues[7] = CurrentLevel.getZone()
        playerValues[8] = CurrentLevel.getFloor()
        playerValues[9] = sequence
        playerValues[12] = visibleEnemies
        playerValues[13] = grid[centre][centre][1] == 3 and 1 or 0

        local slots = player.inventory and player.inventory.itemSlots or {}
        encodeInventoryItem(inventory, 1, slots.weapon and slots.weapon[1])
        encodeInventoryItem(inventory, 2, slots.action and slots.action[1])
        encodeInventoryItem(inventory, 3, slots.action and slots.action[2])
        encodeInventoryItem(inventory, 4, slots.shovel and slots.shovel[1])
        encodeInventoryItem(inventory, 5, slots.spell and slots.spell[1])
        encodeInventoryItem(inventory, 6, slots.spell and slots.spell[2])
        encodeInventoryItem(inventory, 7, slots.bomb and slots.bomb[1])
        encodeInventoryItem(inventory, 8, slots.misc and slots.misc[1])

        playerValues[10] = inventory[7][3]
        playerValues[11] = inventory[1][4]
        mask[6] = inventory[7][1] ~= 0 and 1 or 0
        mask[7] = inventory[2][1] ~= 0 and 1 or 0
        mask[8] = inventory[3][1] ~= 0 and 1 or 0
        mask[9] = inventory[1][1] ~= 0 and 1 or 0
        mask[10] = inventory[5][1] ~= 0 and 1 or 0
        mask[11] = inventory[6][1] ~= 0 and 1 or 0
    end

    return result
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

local function currentContext()
    return {
        seed = CurrentLevel.getSeed(),
        zone = CurrentLevel.getZone(),
        floor = CurrentLevel.getFloor(),
        character = Player.getCharacterType(1),
    }
end

local function beginRunIfNeeded()
    local levelIdentity = tostring(CurrentLevel.getSeed())
        .. ":" .. tostring(CurrentLevel.getUniqueID())
        .. ":" .. tostring(CurrentLevel.getSequentialNumber())
    local newRun = activeRunID == ""
        or (CurrentLevel.getSequentialNumber() == 1 and levelIdentity ~= lastLevelIdentity)
    if newRun then
        sequence = 0
        runCounter = runCounter + 1
        activeRunID = tostring(CurrentLevel.getSeed())
            .. ":" .. tostring(runCounter)
            .. ":" .. tostring(CurrentLevel.getUniqueID())
        pendingEvents = {}
        playerDead = false
        terminalEmitted = false
    end
    lastLevelIdentity = levelIdentity
    return newRun
end

local function observationForStatus(observation, status)
    local result = clone(observation or emptyObservation())
    result.player[15] = status == "won" and 1 or 0
    result.player[16] = status == "dead" and 1 or 0
    return result
end

local function emitRecord(kind, status, observation, context, debugEntities, bridgeCommand)
    if kind == "terminal" and terminalEmitted then
        return
    end
    local terminated = status == "won" or status == "dead"
    local truncated = status == "aborted"
    local record = {
        schema_version = SCHEMA_VERSION,
        instance_id = Bridge.getInstanceID(),
        role = Bridge.getRole(),
        run_id = activeRunID,
        sequence = sequence,
        kind = kind,
        game = {
            version = GAME_VERSION,
            steam_build = STEAM_BUILD,
        },
        seed = context.seed,
        character = context.character,
        zone = context.zone,
        floor = context.floor,
        bridge = bridgeCommand,
        observation = observationForStatus(observation, status),
        debug_entities = debugEntities or {},
        events = pendingEvents,
        episode_status = status,
        terminated = terminated,
        truncated = truncated,
        metrics = {
            turns = sequence,
            completed = status == "won" and 1 or 0,
            deaths = status == "dead" and 1 or 0,
        },
    }
    print(LOG_MARKER .. jsonEncode(record))
    lastObservation = clone(record.observation)
    lastContext = clone(context)
    pendingEvents = {}
    playerDead = false
    if kind == "terminal" then
        terminalEmitted = true
    end
    sequence = sequence + 1
end

local function emitTurn()
    if CurrentLevel.isLoading() or CurrentLevel.isLobby() then
        return
    end
    local bridgeCommand = Bridge.consumeCompletedCommand()
    if activeRunID ~= "" and not bridgeCommand then
        return
    end
    local newRun = beginRunIfNeeded()
    local observation = buildObservation()
    local context = currentContext()
    local kind = newRun and "reset" or "turn"
    local status = "running"
    if playerDead and not newRun then
        kind = "terminal"
        status = "dead"
    end
    emitRecord(
        kind,
        status,
        observation,
        context,
        buildEntityDebug(),
        bridgeCommand
    )
end

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
        queueEvent("failure", 1, ev.entity, { reason = "death" })
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

-- Victory occurs after the last normal turn, so it needs an explicit terminal
-- record. Non-victory run completion is explicit too, preventing Python from
-- waiting forever for another turn.
event.runComplete.add("emitAutoDancerRunComplete", {
    order = "menu",
    sequence = -1,
}, function(ev)
    if terminalEmitted then
        return
    end
    if activeRunID == "" then
        if not CurrentLevel.isLobby() then
            beginRunIfNeeded()
        else
            runCounter = runCounter + 1
            activeRunID = "terminal:" .. tostring(runCounter)
            sequence = 0
            terminalEmitted = false
        end
    end
    local victory = ev.summary and ev.summary.victory == true
    local status
    if victory then
        status = "won"
        queueEvent("success", 1, nil, { task_complete = true })
    elseif playerDead then
        status = "dead"
    else
        status = "aborted"
        queueEvent("failure", 1, nil, { reason = "run_aborted" })
    end
    local observation = lastObservation
    local context = lastContext
    if not observation and not CurrentLevel.isLobby() and not CurrentLevel.isLoading() then
        observation = buildObservation()
        context = currentContext()
    end
    observation = observation or emptyObservation()
    context = context or {
        seed = 0,
        zone = 0,
        floor = 0,
        character = "Bard",
    }
    emitRecord("terminal", status, observation, context, {})
end)

-- turnID is the last stable order key in the supported turn event.
event.turn.add("emitAutoDancerTelemetry", {order = "turnID", sequence = -1}, emitTurn)

-- Emit once when a playable level first appears, and once after each level
-- transition. This carries RESET acknowledgements without requiring an action.
event.tick.add("emitAutoDancerInitialObservation", {
    order = "input",
    sequence = -100,
}, function()
    local player = Player.getPlayerEntity(1)
    if player and not CurrentLevel.isLoading() and not CurrentLevel.isLobby() then
        local levelIdentity = tostring(CurrentLevel.getSeed())
            .. ":" .. tostring(CurrentLevel.getUniqueID())
            .. ":" .. tostring(CurrentLevel.getSequentialNumber())
        if activeRunID == "" or levelIdentity ~= lastLevelIdentity then
            emitTurn()
        end
    end
end)
