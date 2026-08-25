$ErrorActionPreference = "Stop"

$retryRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$retryPython = Join-Path $retryRepo ".venv\Scripts\python.exe"
$retryGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$retryMod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$retrySource = Join-Path $retryRepo "runs\reward-v2-250k\final.pt"
$retryReward = Join-Path $retryRepo "configs\reward-v2.json"
$retryRoot = Join-Path $retryRepo "runs\architecture8-controls"
$retryAttempt = "control-retry-1"
$retryTraining = Join-Path $retryRoot "training\$retryAttempt"
$retryCurve = Join-Path $retryRoot "curve-evaluation\$retryAttempt"
$retryBroad = Join-Path $retryRoot "broad-evaluation\$retryAttempt"
$retryComparison = Join-Path $retryRoot "comparison-$retryAttempt.json"
$retryTrainingSeed = 36001
$retryCurveSeeds = (45001..45016) -join ","
$retryBroadSeeds = (46001..46030) -join ","
$retryFinalSteps = 30720

foreach ($required in @($retryPython, $retrySource, $retryReward)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file is missing: $required"
    }
}
foreach ($required in @($retryGame, $retryMod)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) {
        throw "Required directory is missing: $required"
    }
}
$cudaAvailable = & $retryPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cudaAvailable.Trim() -ne "True") {
    throw "The Architecture-8 control retry requires CUDA-enabled PyTorch"
}

New-Item -ItemType Directory -Path $retryTraining, $retryCurve, $retryBroad -Force |
    Out-Null
Set-Location -LiteralPath $retryRepo
$env:PYTHONPATH = Join-Path $retryRepo "src"

function Invoke-RetryTraining {
    param([string]$Arm, [string]$ActionContract)
    $run = Join-Path $retryTraining $Arm
    $final = Join-Path $run "final.pt"
    if (Test-Path -LiteralPath $final -PathType Leaf) { return }
    New-Item -ItemType Directory -Path $run -Force | Out-Null
    $log = Join-Path $run "console.log"
    $arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $retryGame,
        "--mod-dir", $retryMod,
        "--num-instances", "8",
        "--total-steps", "$retryFinalSteps",
        "--run-dir", $run,
        "--architecture", "2",
        "--device", "cuda",
        "--seed", "$retryTrainingSeed",
        "--reward-config", $retryReward,
        "--action-contract", $ActionContract,
        "--checkpoint-interval", "10240",
        "--evaluation-interval", "0",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765"
    )
    $latest = Join-Path $run "latest.pt"
    if (Test-Path -LiteralPath $latest -PathType Leaf) {
        $arguments += @("--resume", $latest)
    } else {
        $arguments += @("--fine-tune-from", $retrySource)
    }
    & $retryPython @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Training $Arm failed; see $log" }
}

function Get-RetryRestarts {
    param([string]$Arm)
    $metrics = Join-Path $retryTraining "$Arm\metrics.jsonl"
    $value = & $retryPython -c (
        "import json; print(max(int(json.loads(x).get('worker_restarts',0)) for x in open(r'{0}', encoding='utf-8') if x.strip()))" -f $metrics
    )
    if ($LASTEXITCODE -ne 0) { throw "Could not read retry health for $Arm" }
    return [int]$value
}

function Invoke-RetryEvaluation {
    param(
        [string]$Name,
        [string]$Checkpoint,
        [string]$ActionContract,
        [string]$Seeds,
        [int]$MaxSteps,
        [string]$Directory
    )
    $output = Join-Path $Directory "$Name.json"
    if (Test-Path -LiteralPath $output -PathType Leaf) { return }
    $log = Join-Path $Directory "$Name.log"
    & $retryPython -m autodancer.training.baseline `
        --game-dir $retryGame `
        --mod-dir $retryMod `
        --checkpoint $Checkpoint `
        --output $output `
        --num-instances 8 `
        --seeds $Seeds `
        --max-steps $MaxSteps `
        --reward-config $retryReward `
        --action-contract $ActionContract `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation $Name failed; see $log" }
}

Invoke-RetryTraining "a2-legacy" "legacy-no-wait"
Invoke-RetryTraining "a2-fixed" "current"

$legacyRestarts = Get-RetryRestarts "a2-legacy"
$fixedRestarts = Get-RetryRestarts "a2-fixed"
if ($legacyRestarts -ne 0 -or $fixedRestarts -ne 0) {
    @{
        completed_at = (Get-Date).ToString("o")
        decision = "retry_controls_unhealthy"
        a2_legacy_restarts = $legacyRestarts
        a2_fixed_restarts = $fixedRestarts
    } | ConvertTo-Json | Set-Content -LiteralPath (
        Join-Path $retryRoot "pipeline-$retryAttempt-complete.json"
    )
    exit 0
}

foreach ($arm in @("a2-legacy", "a2-fixed")) {
    $contract = if ($arm -eq "a2-legacy") { "legacy-no-wait" } else { "current" }
    foreach ($step in @(0, 10240, 20480, 30720)) {
        $checkpoint = if ($step -eq 0) {
            $retrySource
        } else {
            Join-Path $retryTraining ("$arm\checkpoint-{0:D8}.pt" -f $step)
        }
        Invoke-RetryEvaluation "$arm-step-$('{0:D8}' -f $step)" `
            $checkpoint $contract $retryCurveSeeds 1500 $retryCurve
    }
}

& $retryPython -m autodancer.training.architecture8_compare `
    --root $retryRoot `
    --control-attempt $retryAttempt `
    --output $retryComparison
if ($LASTEXITCODE -ne 0) { throw "Control-retry comparison failed" }
$comparison = Get-Content -LiteralPath $retryComparison -Raw | ConvertFrom-Json
if (-not $comparison.curve_gate_passed) {
    @{
        completed_at = (Get-Date).ToString("o")
        decision = $comparison.decision
    } | ConvertTo-Json | Set-Content -LiteralPath (
        Join-Path $retryRoot "pipeline-$retryAttempt-complete.json"
    )
    exit 0
}

foreach ($arm in @("a2-legacy", "a2-fixed", "a8")) {
    $contract = if ($arm -eq "a2-legacy") { "legacy-no-wait" } else { "current" }
    $checkpoint = if ($arm -eq "a8") {
        Join-Path $retryRoot "training\a8\final.pt"
    } else {
        Join-Path $retryTraining "$arm\final.pt"
    }
    Invoke-RetryEvaluation $arm $checkpoint $contract $retryBroadSeeds 3000 $retryBroad
}

& $retryPython -m autodancer.training.architecture8_compare `
    --root $retryRoot `
    --control-attempt $retryAttempt `
    --output $retryComparison
if ($LASTEXITCODE -ne 0) { throw "Final control-retry comparison failed" }
$comparison = Get-Content -LiteralPath $retryComparison -Raw | ConvertFrom-Json
@{
    completed_at = (Get-Date).ToString("o")
    decision = $comparison.decision
    control_attempt = $retryAttempt
} | ConvertTo-Json | Set-Content -LiteralPath (
    Join-Path $retryRoot "pipeline-$retryAttempt-complete.json"
)
