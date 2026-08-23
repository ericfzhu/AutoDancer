$ErrorActionPreference = "Stop"

$pilotRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pilotUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
$pilotGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$pilotMod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$pilotCheckpoint = Join-Path $pilotRepo "runs\reward-v2-250k\final.pt"
$pilotRoot = Join-Path $pilotRepo "runs\reward-v3-directional"
$pilotSeeds = @(31001, 31002, 31003)

if (-not (Test-Path -LiteralPath $pilotUv -PathType Leaf)) {
    throw "uv is missing at $pilotUv"
}
if (-not (Test-Path -LiteralPath $pilotCheckpoint -PathType Leaf)) {
    throw "V2 initialization checkpoint is missing at $pilotCheckpoint"
}

New-Item -ItemType Directory -Path $pilotRoot -Force | Out-Null
Set-Location -LiteralPath $pilotRepo

foreach ($pilotSeed in $pilotSeeds) {
    $pilotRun = Join-Path $pilotRoot ("seed-{0}" -f $pilotSeed)
    if (Test-Path -LiteralPath $pilotRun) {
        throw "Refusing to overwrite existing pilot run $pilotRun"
    }
    New-Item -ItemType Directory -Path $pilotRun | Out-Null
    $pilotLog = Join-Path $pilotRun "console.log"
    & $pilotUv run autodancer-train `
        --game-dir $pilotGame `
        --mod-dir $pilotMod `
        --num-instances 8 `
        --total-steps 51200 `
        --run-dir $pilotRun `
        --device auto `
        --seed $pilotSeed `
        --initialize-from $pilotCheckpoint `
        --evaluation-interval 0 `
        --dashboard 8765 *> $pilotLog
    if ($LASTEXITCODE -ne 0) {
        throw "Pilot seed $pilotSeed failed with exit code $LASTEXITCODE; see $pilotLog"
    }
}

@{
    completed_at = (Get-Date).ToString("o")
    seeds = $pilotSeeds
    steps_per_seed = 51200
    initialization_checkpoint = $pilotCheckpoint
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pilotRoot "pilot-complete.json")
