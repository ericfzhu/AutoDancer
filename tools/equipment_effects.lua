-- Qualification-only fixtures, embedded in a PRIVATE mod copy.
local effectState
local function applyEquipmentEffectFixture(player, result, sequence)
    local spec = effectFixtures[tonumber(CurrentLevel.getSeed())]
    if not spec or not player or CurrentLevel.getSequentialNumber() ~= 1 then return end
    local Object = require "necro.game.object.Object"
    local Inventory = require "necro.game.item.Inventory"
    local Tile = require "necro.game.tile.Tile"
    local px, py = player.position.x, player.position.y
    if not effectState or effectState.level ~= CurrentLevel.getUniqueID() then
        effectState = {level=CurrentLevel.getUniqueID(), x=px, y=py, enemies={},world={}}
        local remove = {}
        for entity in Entities.entitiesWithComponents({"position"}) do
            if entity.id ~= player.id and entity.gameObject and entity.gameObject.tangible
                and (not entity.item or entity.item.holder == 0)
                and math.abs(entity.position.x-px) <= 5
                and math.abs(entity.position.y-py) <= 5 then
                remove[#remove+1] = entity
            end
        end
        for _, entity in ipairs(remove) do Object.delete(entity) end
        local floor = Tile.get(px, py)
        for dx=-5,5 do for dy=-5,5 do
            if Tile.isInBounds(px+dx,py+dy) then Tile.set(px+dx,py+dy,floor) end
        end end
        player.health.maxHealth = spec.maxHealth or 20
        player.health.health = spec.health or player.health.maxHealth
        for _,name in ipairs(spec.items or {}) do Inventory.replace(name,player) end
        for _,name in ipairs(spec.grant or {}) do Inventory.grant(name,player) end
        for component,fields in pairs(spec.playerFields or {}) do
            assert(player[component], "Missing fixture player component")
            for field,value in pairs(fields) do
                assert(player[component][field] ~= nil, "Unknown fixture player field")
                player[component][field] = value
            end
        end
        for slot,components in pairs(spec.itemFields or {}) do
            local item = assert(Inventory.getItemInSlot(player,slot))
            for component,fields in pairs(components) do
                assert(item[component], "Missing fixture item component")
                for field,value in pairs(fields) do
                    assert(item[component][field] ~= nil, "Unknown fixture item field")
                    item[component][field] = value
                end
            end
        end
        if spec.ammo then
            local weapon = Inventory.getItemInSlot(player,"weapon")
            weapon.weaponReloadable.ammo = spec.ammo
        end
        if spec.wall then Tile.setType(px+1,py,spec.wall) end
        if spec.pickup then effectState.pickup = Object.spawn(spec.pickup,px+1,py).id end
        if spec.stash or spec.stashPrototypes then
            local ItemStash = require "necro.game.item.ItemStash"
            local ItemStorage = require "necro.game.item.ItemStorage"
            local stash = ItemStash.getOrCreate(player)
            for _,name in ipairs(spec.stash or {}) do
                assert(ItemStorage.store(Object.spawn(name,px,py),stash), "Stash fixture failed")
            end
            for _,name in ipairs(spec.stashPrototypes or {}) do
                assert(ItemStorage.store(name,stash), "Prototype stash fixture failed")
            end
        end
        local AI = require "necro.game.enemy.ai.AI"
        local slimeNames = {}
        for _,name in ipairs(Entities.getEntityTypesWithComponents({"enemy","ai"})) do
            if string.find(string.lower(name),"slime")
                and Entities.getEntityPrototype(name).ai.id == AI.Type.IDLE then
                slimeNames[#slimeNames+1] = name
            end
        end
        table.sort(slimeNames)
        assert(#slimeNames > 0,"No stationary slime fixture")
        for _,offset in ipairs(spec.enemies or {}) do
            local enemy = Object.spawn(slimeNames[1],px+offset[1],py+offset[2])
            enemy.health.maxHealth = spec.enemyHealth or 20
            enemy.health.health = enemy.health.maxHealth
            effectState.enemies[#effectState.enemies+1] = enemy.id
        end
        for _,tile in ipairs(spec.worldTiles or {}) do
            Tile.setType(px+tile.x,py+tile.y,tile.name)
        end
        for _,entry in ipairs(spec.worldEntities or {}) do
            local entity = Object.spawn(entry.name,px+entry.x,py+entry.y)
            for component,fields in pairs(entry.fields or {}) do
                assert(entity[component],"Missing world fixture component: "..component)
                for field,value in pairs(fields) do
                    assert(entity[component][field] ~= nil,"Unknown world fixture field: "..field)
                    entity[component][field] = value
                end
            end
            effectState.world[#effectState.world+1] = {id=entity.id,name=entry.name}
        end
    end
    if sequence == 1 and not effectState.injected then
        effectState.injected = true
        if spec.incomingDamage then
            local Damage = require "necro.game.system.Damage"
            Damage.inflict({victim=player, damage=spec.incomingDamage, type=Damage.Type.PHYSICAL})
        end
        if spec.replace then Inventory.replace(spec.replace,player) end
        if spec.healAfterDamage then player.health.health = player.health.health + spec.healAfterDamage end
        for index,components in pairs(spec.worldChange or {}) do
            local entity = assert(Entities.getEntityByID(effectState.world[index].id))
            for component,fields in pairs(components) do
                for field,value in pairs(fields) do entity[component][field] = value end
            end
        end
    end
    local evidence = {enemies={},world={},wall=Tile.getInfo(effectState.x+1,effectState.y).name}
    for _,entry in ipairs(effectState.world) do
        local entity = Entities.getEntityByID(entry.id)
        if entity then
            evidence.world[#evidence.world+1] = {id=entry.id,name=entity.name,
                x=entity.position.x-player.position.x,y=entity.position.y-player.position.y,
                visible=not not Vision.isVisible(entity.position.x,entity.position.y)}
        end
    end
    for _,id in ipairs(effectState.enemies) do
        local enemy = Entities.getEntityByID(id)
        evidence.enemies[#evidence.enemies+1] = {
            id=id, name=enemy and enemy.name or "deleted",
            health=enemy and enemy.health and enemy.health.health or 0,
            x=enemy and enemy.position and enemy.position.x or 0,
            y=enemy and enemy.position and enemy.position.y or 0,
            frozen=enemy and enemy.freezable and enemy.freezable.remainingTurns or 0,
            charmed=not not (enemy and enemy.charmable and enemy.charmable.active),
        }
    end
    if effectState.pickup then
        local item = Entities.getEntityByID(effectState.pickup)
        evidence.pickup = {exists=item~=nil, name=item and item.name or "deleted",
            holder=item and item.item and item.item.holder or 0,
            tangible=item and item.gameObject and item.gameObject.tangible or false}
    end
    if spec.hud then
        local ok, slots = pcall(function()
            local ev = {slots={},shared=false}
            require("necro.game.object.ObjectEvents").fire("getHUDEquipment",player,ev)
            local out = {}
            for slot,items in pairs(ev.slots) do
                for _,item in ipairs(items) do
                    out[#out+1] = {name=item.name,slot=slot,id=item.id}
                end
            end
            return out
        end)
        evidence.hud = {ok=ok,slots=slots}
    end
    result.equipment_fixture = evidence
end
