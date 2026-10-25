-- Engine-backed diagnostic state. Owner selection is NOT action availability.
local Action = require "necro.game.system.Action"
local ActionItem = require "necro.game.item.ActionItem"
local ItemBan = require "necro.game.item.ItemBan"
local SpellItem = require "necro.game.item.SpellItem"
local Entities = require "system.game.Entities"
local ObjectEvents = require "necro.game.object.ObjectEvents"
local Map = require "necro.game.object.Map"
local Vision = require "necro.game.vision.Vision"
local bit = require "bit"

local equipment = {}
local fields = {
    itemActivable = {"active"}, itemToggleable = {"active"},
    weaponReloadable = {"ammo", "maximumAmmo", "ammoPerReload"},
    itemStack = {"quantity"}, weapon = {"damage"},
    weaponThrowable = {}, itemHolster = {},
    spellCooldownTime = {"remainingTurns"}, spellCooldownKills = {"remainingKills"},
    itemComboable = {"combo"}, itemDamageOnActivation = {"damage"},
    spellBloodMagic = {"damage"}, itemBag = {"actionSlotCapacity"},
    itemCastOnUse = {}, itemConvertible = {},
    item = {"equipped"}, storage = {},
    itemConditionalDamageIncrease = {"remainingTurns"},
    itemArmorGrooveChain = {"damageReduction"},
    itemConditionalInvincibility = {"active"},
    itemConditionalInvincibilityOnKill = {"active"},
    itemConditionalDigStrengthIncrease = {"active"},
    itemConditionalDigStrengthOnActivation = {"pending"},
    itemDashOnKill = {"active"}, itemDashOnDig = {"active"},
    itemIncreaseMaxHealth = {"healthGained"}, itemHealOnFirstPickup = {"done"},
    itemDamageOnPickup = {"active"}, weaponParryOnAttack = {"active"},
    itemHolsterAutoSwap = {"justSwapped"}, itemGrantBonusItem = {"granted"},
    health = {"health", "maxHealth"}, cursedHealth = {"health"},
    itemNoReturn = {"active"}, itemStackDupable = {"active"},
    itemOverlaySpriteHUD = {"active"},
    Coldsteel_itemTunnelAttack = {"active"},
    Coldsteel_weaponComboTracker = {"active", "combo"},
    Coldsteel_weaponComboable = {"pendingIncrease"},
    Goldman_itemCollectibleDuringPogo = {"active"},
}
fields.itemComboable = {"combo", "reset"}
fields.itemHolster = {"sticky"}
local specials = {"BOMB", "ITEM_1", "ITEM_2", "THROW", "SPELL_1", "SPELL_2"}
local playerFields = {
    regenerationKillCounter={"killCount", "killThreshold", "totalIncrement"},
    cursedHealth={"health"}, grooveChain={"killCount", "multiplier"},
    invincibility={"remainingTurns", "permanent"}, barrier={"remainingTurns", "permanent"},
    gigantism={"remainingTurns", "permanent"}, dwarfism={"remainingTurns", "permanent"},
    freezable={"remainingTurns", "permanent"}, confusable={"remainingTurns", "permanent"},
    riskDamage={"active", "pending"}, charmable={"active"},
}

function equipment.capture(player)
    local result = {version = 3, current = player ~= nil,
        coverage = {}, items = {}, slots = {}, controls = {}, containers = {}, owned = {},
        player_effects = {}, display = {items={}, controls={}}, world = {}}
    for component in pairs(fields) do
        result.coverage[#result.coverage + 1] = component
    end
    table.sort(result.coverage)
    if not player then return result end
    -- Match the existing visible grid, retaining every overlapping ground item.
    -- Exact names are registry-encoded by Python; no entity IDs become features.
    for dy=-10,10 do for dx=-10,10 do
        local x,y = player.position.x+dx,player.position.y+dy
        if Vision.isVisible(x,y) then
            for _,id in ipairs(Map.getAll(x,y)) do
                local entity = Entities.getEntityByID(id)
                if entity and entity.gameObject and entity.gameObject.tangible then
                    if entity.item and entity.item.holder == 0 then
                        result.world[#result.world+1] = {kind="item",name=entity.name,x=dx,y=dy}
                    elseif entity.charmable and entity.charmable.active then
                        result.world[#result.world+1] = {kind="charmed",x=dx,y=dy}
                    end
                end
            end
        end
    end end
    -- No-return squares are owned marker items, so the ordinary tangible
    -- ground-item filter intentionally excludes them. Their floor marks are
    -- nevertheless visible and affect subsequent movement.
    for marker in Entities.entitiesWithComponents({"itemNoReturn", "position"}) do
        local dx,dy = marker.position.x-player.position.x,marker.position.y-player.position.y
        if marker.itemNoReturn.active and math.abs(dx) <= 10 and math.abs(dy) <= 10
            and Vision.isVisible(marker.position.x,marker.position.y) then
            result.world[#result.world+1] = {kind="no_return",name=marker.name,x=dx,y=dy}
        end
    end
    for name, keys in pairs(playerFields) do
        if player[name] then
            local values = {}
            for _,key in ipairs(keys) do values[key] = player[name][key] end
            result.player_effects[name] = values
        end
    end

    local captured = {}
    local prototypeHandles = {}
    -- Storage may hold a prototype name without a live entity. These local join
    -- handles are outside the engine's entity-ID range and never reach tensors.
    local syntheticID = 4503599627370496
    local addItem
    addItem = function(item, handle)
        local id = handle or item and item.id
        assert(item and id and item.name, "Missing equipment entity")
        assert(handle or id < 4503599627370496, "Equipment entity-ID range changed")
        if captured[id] then return end
        captured[id] = true
        if handle then
            prototypeHandles[item.name] = prototypeHandles[item.name] or {}
            table.insert(prototypeHandles[item.name], handle)
        end
        local components = {}
        for name, keys in pairs(fields) do
            local component = item[name]
            if component ~= nil then
                local values = {}
                for _, key in ipairs(keys) do values[key] = component[key] end
                components[name] = values
            end
        end
        local bans = ItemBan.getBanFlags(player, item.name)
        local record = {
            id = id, name = item.name, components = components,
            bans = {
                pickup = bit.band(bans, ItemBan.Flag.PICKUP) ~= 0,
                equip = bit.band(bans, ItemBan.Flag.EQUIP) ~= 0,
                activate = bit.band(bans, ItemBan.Flag.ACTIVATE) ~= 0,
            },
        }
        if item.itemCastOnUse then
            record.spell_usable = SpellItem.isUsable(item)
        end
        if item.itemConvertible then
            record.conversion_target = item.itemConvertible.targetType
        end
        result.items[#result.items + 1] = record
        if item.itemHolster then
            local content = item.itemHolster.content
            assert(type(content) == "number", "Unknown holster contents")
            result.containers[#result.containers + 1] = {
                item_id = id, content_id = content, kind = "holster", index = 1,
            }
            if content ~= 0 then addItem(Entities.getEntityByID(content)) end
        end
        if item.storage then
            local contents = item.storage.items
            if #contents == 0 then
                result.containers[#result.containers+1] = {
                    item_id=id, content_id=0, kind="storage", index=1,
                }
            end
            for index, value in ipairs(contents) do
                local childID
                if type(value) == "string" then
                    syntheticID = syntheticID + 1
                    childID = syntheticID
                    addItem(Entities.getEntityPrototype(value), childID)
                else
                    assert(type(value) == "number" and value > 0, "Invalid stored item")
                    childID = value
                    addItem(Entities.getEntityByID(value))
                end
                result.containers[#result.containers+1] = {
                    item_id=id, content_id=childID, kind="storage", index=index,
                }
            end
        end
    end
    -- inventory.items includes owned objects that need not occupy a HUD slot.
    for _, id in ipairs(player.inventory and player.inventory.items or {}) do
        addItem(Entities.getEntityByID(id))
        result.owned[#result.owned + 1] = id
    end
    for name, contents in pairs(player.inventory and player.inventory.itemSlots or {}) do
        for index, id in ipairs(contents) do
            addItem(Entities.getEntityByID(id))
            result.slots[#result.slots + 1] = {name = name, index = index, item_id = id}
        end
    end
    for _, name in ipairs(specials) do
        -- Do not set ev.visual: visual fallback/labels are not gameplay ownership.
        local item = ActionItem.getActionItem(player, Action.Special[name])
        if item then addItem(item) end
        result.controls[#result.controls + 1] = {action = name, owner_id = item and item.id or 0}
    end
    -- Query the same equipment selectors as InventoryHUD. Rendering positions,
    -- animation clocks and debug IDs are not policy information.
    local shown = {}
    local show
    show = function(item, slot, action, slotInfo, handle)
        if not item then return end
        local id = handle or item.id
        if not id or id == 0 then
            for _,storedID in ipairs(prototypeHandles[item.name] or {}) do
                show(item,slot,action,slotInfo,storedID)
            end
            return
        end
        if not captured[id] then return end
        local entry = shown[id]
        if not entry then
            entry = {id=id, fields={}}
            shown[id] = entry
        end
        local visible = entry.fields
        if slot then visible.equipped = true end
        if slot and item.item.equipped then
            if item.weaponReloadable and item.weaponAmmoCounterHUD then visible.ammo = true end
            if item.itemComboable and item.itemComboableCounterHUD
                and item.itemComboable.combo >= item.itemComboableCounterHUD.visibilityThreshold then
                visible.combo = true
            end
        end
        if item.weaponReloadable and item.weaponReloadableAmmoSprite
            and (item.item.equipped or item.weaponReloadableAmmoSprite.whileDropped) then
            -- Crossbow ammunition is encoded by distinct sprite frames. A
            -- non-injective sprite mapping (e.g. loaded/unloaded) cannot reveal
            -- an exact counter without the separate HUD quantity label.
            local frames, unique = {}, true
            for ammo=0,item.weaponReloadable.maximumAmmo do
                local frame = item.weaponReloadableAmmoSprite.frameXOffsets[ammo]
                if frame == nil or frames[frame] then unique = false break end
                frames[frame] = true
            end
            if unique then visible.ammo = true end
        end
        if item.itemStack and item.itemStackQuantityLabelHUD
            and (not slotInfo or slotInfo.showQuantity ~= false)
            and item.itemStack.quantity >= item.itemStackQuantityLabelHUD.minimumQuantity then
            visible.quantity = true
        end
        if action and item.itemActivable and item.itemActivablePromptHUD then
            visible.activation_active = true
        end
        if action and item.itemToggleable then visible.toggle_active = true end
        if item.itemHUDCooldown and not (action and item.itemActivable and item.itemActivable.active) then
            visible.cooldown_turns = item.spellCooldownTime ~= nil
            visible.cooldown_kills = item.spellCooldownKills ~= nil
        end
    end
    local hud = {slots={},shared=false}
    ObjectEvents.fire("getHUDEquipment", player, hud)
    for name, contents in pairs(hud.slots) do
        for _,item in ipairs(contents) do show(item, name, nil) end
    end
    for _,name in ipairs(specials) do
        local slot = ActionItem.getVisualSlot(player, Action.Special[name])
        local item = slot.item
        -- If the engine substitutes a different icon, do not disclose the hidden
        -- underlying identity through the player projection.
        if slot.visualItem and item and slot.visualItem.name ~= item.name then item = nil end
        show(item, nil, name, slot)
        result.display.controls[#result.display.controls+1] = {
            action=name, owner_id=item and captured[item.id] and item.id or 0,
        }
    end
    for _,entry in pairs(shown) do
        local names = {}
        for name,visible in pairs(entry.fields) do if visible then names[#names+1] = name end end
        table.sort(names)
        entry.fields = names
        result.display.items[#result.display.items+1] = entry
    end
    table.sort(result.display.items, function(a,b) return a.id < b.id end)
    table.sort(result.items, function(a, b) return a.id < b.id end)
    table.sort(result.owned)
    table.sort(result.containers, function(a, b)
        if a.item_id ~= b.item_id then return a.item_id < b.item_id end
        if a.kind ~= b.kind then return a.kind < b.kind end
        return a.index < b.index
    end)
    table.sort(result.slots, function(a, b)
        return a.name < b.name or (a.name == b.name and a.index < b.index)
    end)
    return result
end

return equipment
