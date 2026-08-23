$ErrorActionPreference = "Stop"

$evalRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$evalPython = Join-Path $evalRepo ".venv\Scripts\python.exe"
$evalGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$evalMod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$evalRoot = Join-Path $evalRepo "runs\reward-v3-heldout-evaluation"
$evalSeeds = 41001..41030
$evalSeedList = $evalSeeds -join ","
$evalCheckpoints = [ordered]@{
    "v2-final" = Join-Path $evalRepo "runs\reward-v2-250k\final.pt"
    "v3-seed-31001" = Join-Path $evalRepo "runs\reward-v3-directional\seed-31001\final.pt"
    "v3-seed-31002" = Join-Path $evalRepo "runs\reward-v3-directional\seed-31002\final.pt"
    "v3-seed-31003" = Join-Path $evalRepo "runs\reward-v3-directional\seed-31003\final.pt"
}

if (-not (Test-Path -LiteralPath $evalPython -PathType Leaf)) {
    throw "Project Python is missing at $evalPython"
}
foreach ($evalCheckpoint in $evalCheckpoints.Values) {
    if (-not (Test-Path -LiteralPath $evalCheckpoint -PathType Leaf)) {
        throw "Evaluation checkpoint is missing at $evalCheckpoint"
    }
}
if (Test-Path -LiteralPath $evalRoot) {
    throw "Refusing to overwrite existing evaluation at $evalRoot"
}

New-Item -ItemType Directory -Path $evalRoot | Out-Null
Set-Location -LiteralPath $evalRepo
$env:PYTHONPATH = Join-Path $evalRepo "src"

foreach ($evalEntry in $evalCheckpoints.GetEnumerator()) {
    $evalName = $evalEntry.Key
    $evalOutput = Join-Path $evalRoot ("{0}.json" -f $evalName)
    $evalLog = Join-Path $evalRoot ("{0}.log" -f $evalName)
    & $evalPython -m autodancer.training.baseline `
        --game-dir $evalGame `
        --mod-dir $evalMod `
        --checkpoint $evalEntry.Value `
        --output $evalOutput `
        --num-instances 8 `
        --seeds $evalSeedList `
        --max-steps 3000 `
        --device auto `
        --dashboard 8765 `
        --trained-only *> $evalLog
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation $evalName failed with exit code $LASTEXITCODE; see $evalLog"
    }
}

$evalSummaries = foreach ($evalEntry in $evalCheckpoints.GetEnumerator()) {
    $evalReport = Get-Content -LiteralPath (Join-Path $evalRoot ("{0}.json" -f $evalEntry.Key)) -Raw |
        ConvertFrom-Json
    [ordered]@{
        name = $evalEntry.Key
        checkpoint = $evalEntry.Value
        trained = $evalReport.trained
    }
}

[ordered]@{
    completed_at = (Get-Date).ToString("o")
    seeds = $evalSeeds
    max_steps_per_episode = 3000
    results = $evalSummaries
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evalRoot "comparison.json")
