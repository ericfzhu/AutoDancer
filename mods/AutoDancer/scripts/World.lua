-- Separate world records at the settled input boundary. No simulation or action advice.
local Entities = require "system.game.Entities"
local Map = require "necro.game.object.Map"
local Tile = require "necro.game.tile.Tile"
local Vision = require "necro.game.vision.Vision"
local world = {}
-- Kept in lockstep with world_registry.json; validated by offline and live checks.
local fields = {
    health={"health","maxHealth"},
    facingDirection={"direction"},
    beatDelay={"counter","interval"},
    beatCounter={"counter"},
    ai={"active","id"},
    aiPattern={"index"},
    aggro={"active"},
    provokable={"active"},
    charge={"active","direction"},
    chargeLimitedDuration={"counter","maxDuration"},
    shield={"active","damageReduction","maximumDamageTaken"},
    shieldDirection={"direction"},
    freezable={"remainingTurns","permanent"},
    confusable={"remainingTurns","permanent"},
    barrier={"remainingTurns","permanent"},
    invincibility={"remainingTurns","permanent"},
    stun={"counter"},
    charmable={"active"},
    slide={"direction"},
    interactable={"active"},
    shrine={"active"},
    priceTag={"active"},
    priceTagCostCurrency={"cost"},
    priceTagCostHealth={"cost"},
    priceTagShopliftable={"allowFurtherPurchases"},
    explosive={"damage"},
    autoCastCooldown={"remainingTurns","cooldown"},
    castWithDelay={"remainingTurns"},
    necrodancer={"cooldown","interval","lastAction"},
    nd2Phase2={"cooldown","interval"},
    boss={"defeated","type"},
    stairLocker={"level"},
    preBossFreeze={"active"},
    grabbable={"isGrabbed"},
    castler={"active","done"},
    tentacleMaster={"active"},
    kingCongaTeleport={"hadThrone"},
    castTrapWalls={"counter","interval"},
    itemNoReturn={"active"},
    gameObject={"active","tangible"},
    aiOverride={"active","id"},
    aiBasicSeek={"vertical"},
    ambusher={"active","pending"},
    amplifyFirstMove={"active"},
    apparitionSpawnProximityCheck={"active"},
    clonking={"active","direction","distance"},
    deadRingerBell={"cooldown","remainingTurns","rung","targeted"},
    doppelganger={"counter","interval"},
    gigantism={"remainingTurns","permanent"},
    goolemCastOnHit={"active"},
    innateAttack={"damage","knockback"},
    innateAttackInhibitTemporarily={"remainingTurns"},
    luteHead={"phase","fireballCooldown","flee"},
    parry={"cooldown"},
    parryCounterAttack={"active","direction","distance"},
    spawnCap={"counter"},
    spawnInvincibility={"active"},
    spawnPrep={"active"},
    stasis={"active"},
    supercharge={"active"},
    targetable={"active"},
    trapCast={"direction"},
    captiveAudience={"active"},
    dismountOnHit={"active"},
    provokeOnAddsDeath={"active"},
    provokableFearOtherTeams={"active"},
    shieldDodgeOnHit={"direction"},
}
local markers = {"item","trap","enemy","playableCharacter","chestLike","shrine","shopkeeper","interactable","priceTag","explosive","interactableConsumeKey","shieldDirectionFollowFacingDirection"}

function world.capture(player)
    local result = {version=1,current=player~=nil,coverage={},entities={},tiles={}}
    for component,keys in pairs(fields) do
        for _,key in ipairs(keys) do result.coverage[#result.coverage+1] = component.."."..key end
    end
    table.sort(result.coverage)
    if not player then return result end
    local seen = {}
    for dy=-10,10 do for dx=-10,10 do
        local x,y = player.position.x+dx,player.position.y+dy
        if Vision.isVisible(x,y) then
            if Tile.exists(x,y) then
                result.tiles[#result.tiles+1] = {x=dx,y=dy,name=Tile.getInfo(x,y).name}
            end
            for _,id in ipairs(Map.getAll(x,y)) do
                local entity = Entities.getEntityByID(id)
                -- Preserve intangible hazards and markers. Held inventory is represented
                -- by Equipment; map locations of held items are not world observations.
                if entity and not seen[id] and entity.gameObject
                    and not entity.gameObject.despawned
                    and not (entity.item and entity.item.holder ~= 0 and not entity.itemNoReturn) then
                    seen[id] = true
                    local row = {id=id,name=entity.name,x=dx,y=dy,components={},markers={},
                        displayed=not not ((entity.sprite and entity.sprite.visible)
                            or (entity.priceTagLabel and entity.priceTag and entity.priceTag.active)),
                        relations={}}
                    for component,keys in pairs(fields) do
                        if entity[component] ~= nil then
                            local values = {}
                            for _,key in ipairs(keys) do values[key] = entity[component][key] end
                            row.components[component] = values
                        end
                    end
                    for _,component in ipairs(markers) do
                        if entity[component] ~= nil then row.markers[#row.markers+1] = component end
                    end
                    if entity.priceTagCostCurrency then
                        row.currency = entity.priceTagCostCurrency.currency
                    end
                    if entity.grab and entity.grab.target ~= 0 then row.relations.grab = entity.grab.target end
                    if entity.grabbable and entity.grabbable.grabber ~= 0 then
                        row.relations.grabber = entity.grabbable.grabber
                    end
                    result.entities[#result.entities+1] = row
                end
            end
        end
    end end
    return result
end
return world
