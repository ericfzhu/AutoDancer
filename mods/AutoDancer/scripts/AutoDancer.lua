-- AutoDancer live telemetry for an unpackaged local SYNCHRONY mod.
--
-- Python sends actions and receives matching transition records through the
-- process-local native named pipe.

local Bridge = require "AutoDancer.scripts.Bridge"

local Music = require "necro.audio.Music"
local AnimationTimer = require "necro.render.AnimationTimer"
local CurrentLevel = require "necro.game.level.CurrentLevel"
local Action = require "necro.game.system.Action"
local Map = require "necro.game.object.Map"
local Player = require "necro.game.character.Player"
local Tile = require "necro.game.tile.Tile"
local Vision = require "necro.game.vision.Vision"
local Entities = require "system.game.Entities"
local Native = require "system.game.AutoDancerNative"

local GRID_SIZE = 21
local GRID_CHANNELS = 29
local MAP_SIZE = 65
local PLAYER_FEATURES = 21
local INVENTORY_SLOTS = 13
local INVENTORY_FEATURES = 8
local ACTION_COUNT = 11
local SCHEMA_VERSION = 10

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
local mapOriginX = nil
local mapOriginY = nil
local mapLevelIdentity = ""

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

local function currentLevelIdentity()
    return tostring(CurrentLevel.getSeed())
        .. ":" .. tostring(CurrentLevel.getUniqueID())
        .. ":" .. tostring(CurrentLevel.getSequentialNumber())
end

local function ensureMapOrigin(player)
    local identity = currentLevelIdentity()
    if identity ~= mapLevelIdentity then
        mapLevelIdentity = identity
        mapOriginX = player.position.x
        mapOriginY = player.position.y
    end
end

local function playerHasMap(player)
    local slots = player.inventory and player.inventory.itemSlots or {}
    for _, slot in pairs(slots) do
        if type(slot) == "table" then
            for _, entityID in ipairs(slot) do
                local item = Entities.getEntityByID(entityID)
                if item and string.find(string.lower(item.name or ""), "map", 1, true) then
                    return true
                end
            end
        end
    end
    return false
end

local function buildRevealedMap(player)
    -- A periodic snapshot recovers map reveals that occur outside the local grid.
    -- Map-item reveals are emitted immediately. The cadence keeps normal turns cheap.
    if sequence % 32 ~= 0 and not playerHasMap(player) then
        return nil
    end
    ensureMapOrigin(player)
    local result = zeroMatrix(MAP_SIZE, MAP_SIZE)
    local half = math.floor(MAP_SIZE / 2)
    for row = 1, MAP_SIZE do
        for column = 1, MAP_SIZE do
            local x = mapOriginX + column - half - 1
            local y = mapOriginY + row - half - 1
            if Vision.isRevealed(x, y) and Tile.exists(x, y) then
                local tileInfo = Tile.getInfo(x, y) or {}
                if tileInfo.name == "Stairs" or (tileInfo.descent or 0) > 0 then
                    result[row][column] = 3
                else
                    result[row][column] = Tile.isSolid(x, y) and 2 or 1
                end
            end
        end
    end
    return result
end

local function currentMapBounds()
    local x, y, width, height = Tile.getLevelBounds()
    return {
        x = x,
        y = y,
        width = width,
        height = height,
    }
end

local function shopMusicVolumeBasisPoints()
    local player = Player.getPlayerEntity(1)
    if not player or not player.position then
        return 0
    end
    local volume = 0
    for entity in Entities.entitiesWithComponents({ "shopkeeper", "musicLayerAddVolume" }) do
        local layer = entity.musicLayerAddVolume
        if layer.active and layer.effective and entity.position then
            local dx = entity.position.x - player.position.x
            local dy = entity.position.y - player.position.y
            local distance = math.sqrt(dx * dx + dy * dy)
            local innerRadius = layer.innerRadius or 0
            local outerRadius = math.max(innerRadius, layer.outerRadius or innerRadius)
            local innerVolume = layer.innerVolume or 0
            local outerVolume = layer.outerVolume or 0
            local layerVolume
            if distance <= innerRadius or outerRadius == innerRadius then
                layerVolume = innerVolume
            elseif distance >= outerRadius then
                layerVolume = outerVolume
            else
                local factor = (distance - innerRadius) / (outerRadius - innerRadius)
                layerVolume = innerVolume + (outerVolume - innerVolume) * factor
            end
            volume = math.max(volume, layerVolume)
        end
    end
    return math.max(0, math.min(32767, math.floor(volume * 10000 + 0.5)))
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

local function logicalDirection(direction)
    if direction == Action.Direction.UP then
        return 1
    elseif direction == Action.Direction.RIGHT then
        return 2
    elseif direction == Action.Direction.DOWN then
        return 3
    elseif direction == Action.Direction.LEFT then
        return 4
    end
    return 0
end

local function remainingStatusTurns(component)
    if not component then
        return 0
    elseif component.permanent then
        return 32767
    end
    return math.max(0, math.min(32767, component.remainingTurns or 0))
end

local function addFlag(value, flag)
    if math.floor(value / flag) % 2 == 0 then
        return value + flag
    end
    return value
end

local objectPriority = { [1] = 5, [2] = 5, [3] = 1, [4] = 4, [5] = 2 }

local function setVisibleObject(cell, kind, entity)
    local current = cell[20]
    if current == 0 or (objectPriority[kind] or 0) > (objectPriority[current] or 0) then
        cell[20] = kind
        cell[21] = typeID(entity.name)
    end
end

local function animationDeciseconds(entity, component, animationName)
    if not hasComponent(entity, component)
        or not AnimationTimer.isPlayingInTurn(entity.id, animationName) then
        return 0
    end
    local elapsed = AnimationTimer.getTime(entity.id, animationName) or 0
    return math.max(0, math.min(32767, math.floor(elapsed * 10 + 0.5)))
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
    -- itemHUDCooldown is only a HUD-opacity marker. The mutable turn counter
    -- belongs to spellCooldownTime in the pinned game schema.
    inventory[row][5] = hasComponent(item, "spellCooldownTime")
        and math.max(0, item.spellCooldownTime.remainingTurns or 0) or 0
    inventory[row][6] = hasComponent(item, "spellCooldownKills")
        and math.max(0, item.spellCooldownKills.remainingKills or 0) or 0
    inventory[row][7] = inventory[row][5] == 0 and inventory[row][6] == 0 and 1 or 0
    inventory[row][8] = hasComponent(item, "itemToggleable")
        and item.itemToggleable.active and 1 or 0
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
            if hasComponent(entity, "facingDirection") then
                cell[12] = logicalDirection(entity.facingDirection.direction)
            end
            if hasComponent(entity, "beatDelay") then
                cell[13] = math.max(0, entity.beatDelay.counter or 0)
                cell[14] = math.max(0, entity.beatDelay.interval or 0)
            end
            if hasComponent(entity, "freezable") then
                cell[15] = remainingStatusTurns(entity.freezable)
            end
            if hasComponent(entity, "confusable") then
                cell[16] = remainingStatusTurns(entity.confusable)
            end
            if hasComponent(entity, "charge") and entity.charge.active then
                cell[17] = 1
                cell[18] = logicalDirection(entity.charge.direction)
            end
            if hasComponent(entity, "shieldDirection") then
                cell[19] = logicalDirection(entity.shieldDirection.direction)
            elseif hasComponent(entity, "shieldDirectionFollowFacingDirection")
                and hasComponent(entity, "facingDirection") then
                cell[19] = logicalDirection(entity.facingDirection.direction)
            end
            if hasComponent(entity, "chestLike") then
                setVisibleObject(cell, 1, entity)
            elseif hasComponent(entity, "shrine") then
                setVisibleObject(cell, 2, entity)
            elseif hasComponent(entity, "shopkeeper") then
                setVisibleObject(cell, 4, entity)
            elseif hasComponent(entity, "interactable") then
                setVisibleObject(cell, 5, entity)
            elseif hasComponent(entity, "priceTag") then
                setVisibleObject(cell, 3, entity)
            end
            if hasComponent(entity, "interactable") and entity.interactable.active ~= false then
                cell[22] = addFlag(cell[22], 1)
            end
            if hasComponent(entity, "interactableConsumeKey") then
                cell[22] = addFlag(cell[22], 2)
            end
            if hasComponent(entity, "shrine") and entity.shrine.active then
                cell[22] = addFlag(cell[22], 4)
            end
            -- `sale` is a marker component on the supported game build; unlike
            -- `interactable`, it does not define an `active` field.
            if hasComponent(entity, "sale") then
                cell[22] = addFlag(cell[22], 8)
            end
            if hasComponent(entity, "priceTagShopliftable") then
                cell[22] = addFlag(cell[22], 16)
            end
            if hasComponent(entity, "priceTagCostCurrency") then
                cell[23] = typeID(tostring(entity.priceTagCostCurrency.currency or "gold"))
                cell[24] = math.max(0, math.min(32767,
                    math.floor((entity.priceTagCostCurrency.cost or 0) + 0.5)))
            end
            if hasComponent(entity, "priceTagCostHealth") then
                cell[25] = math.max(0, math.min(32767,
                    math.floor((entity.priceTagCostHealth.costMultiplier or 0) * 10000 + 0.5)))
            end
            cell[26] = math.max(cell[26],
                animationDeciseconds(entity, "trapActivationAnimation", "trapActivationAnimation"))
            cell[27] = math.max(cell[27],
                animationDeciseconds(entity, "trapFailAnimation", "trapFailAnimation"))
            cell[28] = math.max(cell[28],
                animationDeciseconds(entity, "tellAnimation", "tellAnimation"))
            if hasComponent(entity, "explosive") then
                cell[29] = 1
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
    -- Bard can always consume a beat without moving. This is logical action 4
    -- (Lua array index 5) and must remain available independently of inventory.
    mask[5] = 1

    if player and player.position then
        ensureMapOrigin(player)
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
        playerValues[4] = hasComponent(player, "grooveChain")
            and player.grooveChain.multiplier or 0
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
        encodeInventoryItem(inventory, 9, slots.body and slots.body[1])
        encodeInventoryItem(inventory, 10, slots.head and slots.head[1])
        encodeInventoryItem(inventory, 11, slots.feet and slots.feet[1])
        encodeInventoryItem(inventory, 12, slots.torch and slots.torch[1])
        encodeInventoryItem(inventory, 13, slots.ring and slots.ring[1])

        playerValues[10] = inventory[7][3]
        playerValues[11] = inventory[1][4]
        mask[6] = inventory[7][1] ~= 0 and 1 or 0
        mask[7] = inventory[2][1] ~= 0 and inventory[2][7] or 0
        mask[8] = inventory[3][1] ~= 0 and inventory[3][7] or 0
        mask[9] = inventory[1][1] ~= 0 and 1 or 0
        mask[10] = inventory[5][1] ~= 0 and inventory[5][7] or 0
        mask[11] = inventory[6][1] ~= 0 and inventory[6][7] or 0
        local musicTime = math.max(Music.getMusicTime() or 0, 0)
        local musicLength = math.max(Music.getMusicLength() or 0, 0)
        playerValues[17] = math.floor(musicTime * 10 + 0.5)
        playerValues[18] = math.floor(musicLength * 10 + 0.5)
        playerValues[19] = math.floor(math.max(musicLength - musicTime, 0) * 10 + 0.5)
        playerValues[20] = Music.isSongEndReached() and 1 or 0
        playerValues[21] = shopMusicVolumeBasisPoints()
        result.map_bounds = currentMapBounds()
        result.revealed_map = buildRevealedMap(player)
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
    local levelIdentity = currentLevelIdentity()
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

local function emitRecord(kind, status, observation, context, bridgeCommand)
    if kind == "terminal" and terminalEmitted then
        return
    end
    local terminated = status == "won" or status == "dead"
    local truncated = status == "aborted"
    local record = {
        message_type = "transition",
        schema_version = SCHEMA_VERSION,
        instance_id = Bridge.getInstanceID(),
        role = Bridge.getRole(),
        session_id = Bridge.getSupervisorSession(),
        launch_id = Bridge.getLaunchID(),
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
    assert(Native.send(jsonEncode(record)), "AutoDancer telemetry pipe write failed")
    if bridgeCommand then
        Bridge.markTelemetrySent(bridgeCommand)
    end
    lastObservation = record.observation
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
    emitRecord("terminal", status, observation, context)
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
