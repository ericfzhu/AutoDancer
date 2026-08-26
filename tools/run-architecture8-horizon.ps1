$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$hzRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$hzPython = Join-Path $hzRepo ".venv\Scripts\python.exe"
$hzGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$hzMod = Join-Path $hzRepo "mods\AutoDancer"
$hzReward = Join-Path $hzRepo "configs\reward-v2.json"
$hzQualification = Join-Path $hzRepo "runs\controller-qualification\qualification.json"
$hzPriorRoot = Join-Path $hzRepo "runs\architecture8-qualified-replication\training"
$hzSources = @{
    "a2-continuation" = Join-Path $hzPriorRoot "a2-finetune\final.pt"
    "a8-continuation" = Join-Path $hzPriorRoot "a8-candidate\final.pt"
}
$hzExpectedHashes = @{
    "a2-continuation" = "e29cd49f6eabf57c3b2e9a6c46661a174579e380e03c0d306c63d788fcc7744d"
    "a8-continuation" = "8c2e311cef39dd0c59c9a6c4cd4a7f4a21c88df89d13e26f862c8f821512b6e9"
}
$hzExpectedQualificationHash = "126533ab66c31b699709c6e32bfaf9099fbeb4e96b651c68f1f6d602b8572888"
$hzRoot = Join-Path $hzRepo "runs\architecture8-horizon"
$hzTraining = Join-Path $hzRoot "training"
$hzCurve = Join-Path $hzRoot "curve-evaluation"
$hzLineage = Join-Path $hzRoot "artifact-lineage"
$hzStatus = Join-Path $hzRoot "pipeline-status.json"
$hzSeed = 51001
$hzEvaluationSeeds = (56001..56016) -join ","
$hzSteps = @(30720, 61440, 122880, 250880)
$hzFinalStep = 250880

function Write-HorizonStatus {
    param([string]$Status, [string]$Decision = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0008"
        status = $Status
        decision = $Decision
        updated_at = (Get-Date).ToString("o")
        training_seed = $hzSeed
        evaluation_seeds = 56001..56016
        curve_steps = $hzSteps
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $hzStatus
}

function Test-CompletedLineage {
    param([string]$Directory)
    $manifest = Join-Path $Directory "lineage.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { return $false }
    $status = & $hzPython -c (
        "import json; print(json.load(open(r'{0}', encoding='utf-8')).get('status',''))" -f $manifest
    )
    return $LASTEXITCODE -eq 0 -and $status.Trim() -eq "completed"
}

function Register-HorizonArtifact {
    param(
        [string]$Arm,
        [string]$Trial,
        [string]$Stage,
        [string[]]$Artifacts,
        [string]$Notes
    )
    $lineageDirectory = Join-Path $hzLineage $Trial
    if (Test-CompletedLineage $lineageDirectory) { return }
    $arguments = @(
        "-m", "autodancer.experiments.cli", "attach",
        "--experiment-id", "EXP-0008",
        "--arm", $Arm,
        "--trial", $Trial,
        "--stage", $Stage,
        "--run-dir", $lineageDirectory,
        "--game-dir", $hzGame,
        "--mod-dir", $hzMod,
        "--qualification-report", $hzQualification,
        "--device", "cuda",
        "--notes", $Notes
    )
    foreach ($artifact in $Artifacts) { $arguments += @("--artifact", $artifact) }
    & $hzPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "Could not attach artifact lineage for $Trial" }
}

function Invoke-HorizonTraining {
    param([string]$Arm, [int]$Architecture, [int]$FreezeBaseUpdates)
    $run = Join-Path $hzTraining $Arm
    $final = Join-Path $run "final.pt"
    if ((Test-Path -LiteralPath $final -PathType Leaf) -and (Test-CompletedLineage $run)) {
        $step = & $hzPython -c (
            "import torch; print(torch.load(r'{0}', map_location='cpu', weights_only=False).get('global_step',0))" -f $final
        )
        if ($LASTEXITCODE -eq 0 -and [int]$step -eq $hzFinalStep) { return }
    }
    New-Item -ItemType Directory -Path $run -Force | Out-Null
    $resume = Join-Path $run "latest.pt"
    if (-not (Test-Path -LiteralPath $resume -PathType Leaf)) { $resume = $hzSources[$Arm] }
    $log = Join-Path $run "console.log"
    $arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $hzGame,
        "--mod-dir", $hzMod,
        "--num-instances", "8",
        "--total-steps", "$hzFinalStep",
        "--run-dir", $run,
        "--architecture", "$Architecture",
        "--device", "cuda",
        "--seed", "$hzSeed",
        "--reward-config", $hzReward,
        "--reward-lineage-version", "V2",
        "--action-contract", "current",
        "--freeze-base-updates", "$FreezeBaseUpdates",
        "--checkpoint-interval", "10240",
        "--evaluation-interval", "0",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765",
        "--experiment-id", "EXP-0008",
        "--experiment-arm", $Arm,
        "--trial-id", "seed-$hzSeed",
        "--controller-qualification", $hzQualification,
        "--resume", $resume
    )
    Add-Content -LiteralPath $log -Value "`n=== resume $Arm at $((Get-Date).ToString('o')) ==="
    & $hzPython @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Training $Arm failed; see $log" }
}

function Invoke-HorizonEvaluation {
    param([string]$Arm, [int]$Step, [string]$Checkpoint, [string]$Directory)
    $output = Join-Path $Directory "report.json"
    if ((Test-Path -LiteralPath $output -PathType Leaf) -and (Test-CompletedLineage $Directory)) {
        return
    }
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $log = Join-Path $Directory "console.log"
    & $hzPython -m autodancer.training.baseline `
        --game-dir $hzGame `
        --mod-dir $hzMod `
        --checkpoint $Checkpoint `
        --output $output `
        --num-instances 8 `
        --seeds $hzEvaluationSeeds `
        --max-steps 3000 `
        --reward-config $hzReward `
        --reward-lineage-version V2 `
        --action-contract current `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only `
        --experiment-id EXP-0008 `
        --experiment-arm $Arm `
        --trial-id "curve-$Arm-step-$Step" `
        --controller-qualification $hzQualification *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation $Arm step $Step failed; see $log" }
}

New-Item -ItemType Directory -Path $hzRoot, $hzTraining, $hzCurve, $hzLineage -Force | Out-Null
Write-HorizonStatus "preflight"
trap {
    Write-HorizonStatus "failed" $_.Exception.Message
    throw
}

foreach ($required in @($hzPython, $hzReward, $hzQualification) + $hzSources.Values) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing file: $required" }
}
foreach ($required in @($hzGame, $hzMod)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) { throw "Missing directory: $required" }
}
foreach ($arm in $hzSources.Keys) {
    $actual = (Get-FileHash -LiteralPath $hzSources[$arm] -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $hzExpectedHashes[$arm]) { throw "EXP-0008 source hash mismatch for $arm" }
}
$qualificationHash = (Get-FileHash -LiteralPath $hzQualification -Algorithm SHA256).Hash.ToLowerInvariant()
if ($qualificationHash -ne $hzExpectedQualificationHash) { throw "EXP-0008 qualification hash mismatch" }
$cuda = & $hzPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0008 requires CUDA" }

Set-Location -LiteralPath $hzRepo
& $hzPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }

Write-HorizonStatus "training-a2"
Invoke-HorizonTraining "a2-continuation" 2 0
Write-HorizonStatus "training-a8"
Invoke-HorizonTraining "a8-continuation" 8 10

$representation = Join-Path $hzRoot "representation-final.json"
if (-not (Test-Path -LiteralPath $representation -PathType Leaf)) {
    & $hzPython -m autodancer.training.representation `
        (Join-Path $hzTraining "a8-continuation\final.pt") `
        --output $representation `
        --require-material-new-groups
    if ($LASTEXITCODE -ne 0) { throw "A8 lost material rich-observation influence" }
}
Register-HorizonArtifact "a8-continuation" "representation-final" "diagnostic" `
    @($representation) "A8 rich-observation influence at 250,880 transitions."

Write-HorizonStatus "evaluation"
$frozenSource = Join-Path $hzRepo "runs\reward-v2-250k\final.pt"
Invoke-HorizonEvaluation "a2-frozen" 30720 $frozenSource (Join-Path $hzCurve "a2-frozen")
foreach ($arm in @("a2-continuation", "a8-continuation")) {
    foreach ($step in $hzSteps) {
        if ($step -eq 30720) {
            $checkpoint = $hzSources[$arm]
        } elseif ($step -eq $hzFinalStep) {
            $checkpoint = Join-Path $hzTraining "$arm\final.pt"
        } else {
            $checkpoint = Join-Path $hzTraining ("$arm\checkpoint-{0:D8}.pt" -f $step)
        }
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Missing checkpoint for $arm at ${step}: $checkpoint"
        }
        Invoke-HorizonEvaluation $arm $step $checkpoint `
            (Join-Path $hzCurve ("$arm\step-{0:D8}" -f $step))
    }
}

Write-HorizonStatus "comparison"
$comparison = Join-Path $hzRoot "comparison.json"
& $hzPython -m autodancer.training.architecture8_horizon_compare --root $hzRoot --output $comparison
if ($LASTEXITCODE -ne 0) { throw "EXP-0008 comparison failed" }
$result = Get-Content -LiteralPath $comparison -Raw | ConvertFrom-Json
Register-HorizonArtifact "aggregate" "horizon-comparison" "comparison" @($comparison) `
    "EXP-0008 predeclared long-horizon comparison."
Write-HorizonStatus "completed" $result.decision
