$ErrorActionPreference = "Stop"

$repRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repPython = Join-Path $repRepo ".venv\Scripts\python.exe"
$repGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$repMod = Join-Path $repRepo "mods\AutoDancer"
$repSource = Join-Path $repRepo "runs\reward-v2-250k\final.pt"
$repReward = Join-Path $repRepo "configs\reward-v2.json"
$repQualification = Join-Path $repRepo "runs\controller-qualification\qualification.json"
$repRoot = Join-Path $repRepo "runs\architecture8-qualified-replication"
$repTraining = Join-Path $repRoot "training"
$repCurve = Join-Path $repRoot "curve-evaluation"
$repBroad = Join-Path $repRoot "broad-evaluation"
$repLineage = Join-Path $repRoot "artifact-lineage"
$repStatus = Join-Path $repRoot "pipeline-status.json"
$repTrainingSeed = 51001
$repCurveSeeds = (52001..52016) -join ","
$repBroadSeeds = (53001..53030) -join ","
$repFinalSteps = 30720
$repWarmupSteps = 10240
$repExpectedQualificationHash = "126533ab66c31b699709c6e32bfaf9099fbeb4e96b651c68f1f6d602b8572888"

function Write-ReplicationStatus {
    param([string]$Status, [string]$Decision = "")
    $payload = @{
        schema_version = 1
        experiment_id = "EXP-0007"
        status = $Status
        decision = $Decision
        updated_at = (Get-Date).ToString("o")
        training_seed = $repTrainingSeed
        curve_evaluation_seeds = 52001..52016
        broad_evaluation_seeds = 53001..53030
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $repStatus
}

New-Item -ItemType Directory -Path $repRoot, $repTraining, $repCurve, $repBroad, $repLineage `
    -Force | Out-Null
Write-ReplicationStatus "starting"

trap {
    Write-ReplicationStatus "failed" $_.Exception.Message
    throw
}

foreach ($repRequired in @($repPython, $repSource, $repReward, $repQualification)) {
    if (-not (Test-Path -LiteralPath $repRequired -PathType Leaf)) {
        throw "Required file is missing: $repRequired"
    }
}
foreach ($repRequiredDirectory in @($repGame, $repMod)) {
    if (-not (Test-Path -LiteralPath $repRequiredDirectory -PathType Container)) {
        throw "Required directory is missing: $repRequiredDirectory"
    }
}
$repQualificationHash = (Get-FileHash -LiteralPath $repQualification -Algorithm SHA256).Hash
if ($repQualificationHash.ToLowerInvariant() -ne $repExpectedQualificationHash) {
    throw "Controller qualification hash does not match the predeclared EXP-0007 contract"
}
$repQualificationPassed = & $repPython -c (
    "import json; print(json.load(open(r'{0}', encoding='utf-8')).get('passed') is True)" -f `
        $repQualification
)
if ($LASTEXITCODE -ne 0 -or $repQualificationPassed.Trim() -ne "True") {
    throw "EXP-0007 requires a passed controller qualification"
}
$repCudaAvailable = & $repPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $repCudaAvailable.Trim() -ne "True") {
    throw "EXP-0007 requires CUDA-enabled PyTorch"
}
Set-Location -LiteralPath $repRepo
& $repPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }

function Test-CompletedLineage {
    param([string]$Directory)
    $manifest = Join-Path $Directory "lineage.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { return $false }
    $status = & $repPython -c (
        "import json; print(json.load(open(r'{0}', encoding='utf-8')).get('status',''))" -f `
            $manifest
    )
    return $LASTEXITCODE -eq 0 -and $status.Trim() -eq "completed"
}

function Register-ReplicationArtifact {
    param(
        [string]$Arm,
        [string]$Trial,
        [string]$Stage,
        [string[]]$Artifacts,
        [string]$Notes
    )
    $lineageDirectory = Join-Path $repLineage $Trial
    if (Test-CompletedLineage $lineageDirectory) { return }
    $arguments = @(
        "-m", "autodancer.experiments.cli", "attach",
        "--experiment-id", "EXP-0007",
        "--arm", $Arm,
        "--trial", $Trial,
        "--stage", $Stage,
        "--run-dir", $lineageDirectory,
        "--game-dir", $repGame,
        "--mod-dir", $repMod,
        "--qualification-report", $repQualification,
        "--device", "cuda",
        "--notes", $Notes
    )
    foreach ($artifact in $Artifacts) { $arguments += @("--artifact", $artifact) }
    & $repPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "Could not attach artifact lineage for $Trial" }
}

function Invoke-ReplicationTraining {
    param(
        [string]$Arm,
        [int]$Architecture,
        [int]$TotalSteps,
        [int]$FreezeBaseUpdates = 0
    )
    $run = Join-Path $repTraining $Arm
    $final = Join-Path $run "final.pt"
    if (Test-Path -LiteralPath $final -PathType Leaf) {
        $step = & $repPython -c (
            "import torch; print(torch.load(r'{0}', map_location='cpu', weights_only=False).get('global_step',0))" -f `
                $final
        )
        if ($LASTEXITCODE -eq 0 -and [int]$step -ge $TotalSteps -and `
            (Test-CompletedLineage $run)) { return }
    }
    New-Item -ItemType Directory -Path $run -Force | Out-Null
    $log = Join-Path $run "console.log"
    $arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $repGame,
        "--mod-dir", $repMod,
        "--num-instances", "8",
        "--total-steps", "$TotalSteps",
        "--run-dir", $run,
        "--architecture", "$Architecture",
        "--device", "cuda",
        "--seed", "$repTrainingSeed",
        "--reward-config", $repReward,
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
        "--experiment-id", "EXP-0007",
        "--experiment-arm", $Arm,
        "--trial-id", "seed-$repTrainingSeed",
        "--controller-qualification", $repQualification
    )
    $latest = Join-Path $run "latest.pt"
    if (Test-Path -LiteralPath $latest -PathType Leaf) {
        $arguments += @("--resume", $latest)
    } else {
        $arguments += @("--fine-tune-from", $repSource)
    }
    Add-Content -LiteralPath $log -Value (
        "`n=== launch $Arm to $TotalSteps at $((Get-Date).ToString('o')) ==="
    )
    & $repPython @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Training $Arm failed; see $log" }
}

function Invoke-ReplicationEvaluation {
    param(
        [string]$Arm,
        [string]$Trial,
        [string]$Checkpoint,
        [string]$Seeds,
        [int]$MaxSteps,
        [string]$Directory
    )
    $output = Join-Path $Directory "report.json"
    if ((Test-Path -LiteralPath $output -PathType Leaf) -and `
        (Test-CompletedLineage $Directory)) { return }
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    $log = Join-Path $Directory "console.log"
    & $repPython -m autodancer.training.baseline `
        --game-dir $repGame `
        --mod-dir $repMod `
        --checkpoint $Checkpoint `
        --output $output `
        --num-instances 8 `
        --seeds $Seeds `
        --max-steps $MaxSteps `
        --reward-config $repReward `
        --reward-lineage-version V2 `
        --action-contract current `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only `
        --experiment-id EXP-0007 `
        --experiment-arm $Arm `
        --trial-id $Trial `
        --controller-qualification $repQualification *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation $Trial failed; see $log" }
}

Write-ReplicationStatus "parity"
$repParity = Join-Path $repRoot "parity.json"
if (-not (Test-Path -LiteralPath $repParity -PathType Leaf)) {
    & $repPython -m autodancer.training.architecture8_parity `
        --checkpoint $repSource `
        --output $repParity
    if ($LASTEXITCODE -ne 0) { throw "Architecture-8 parity/gradient preflight failed" }
}
Register-ReplicationArtifact "a8-candidate" "parity" "diagnostic" @($repParity) `
    "Exact A2-to-A8 parity and gradient-opening preflight."

Write-ReplicationStatus "training"
Invoke-ReplicationTraining "a2-finetune" 2 $repFinalSteps
Invoke-ReplicationTraining "a8-candidate" 8 $repWarmupSteps 10

$repWarmupCheckpoint = Join-Path $repTraining "a8-candidate\checkpoint-00010240.pt"
$repWarmupRepresentation = Join-Path $repRoot "representation-warmup.json"
if (-not (Test-Path -LiteralPath $repWarmupRepresentation -PathType Leaf)) {
    & $repPython -m autodancer.training.representation `
        $repWarmupCheckpoint `
        --output $repWarmupRepresentation `
        --require-material-new-groups
    if ($LASTEXITCODE -ne 0) {
        throw "A8 failed the predeclared warmup representation gate"
    }
}
Register-ReplicationArtifact "a8-candidate" "representation-warmup" "diagnostic" `
    @($repWarmupRepresentation) "A8 material-feature diagnostic at 10,240 transitions."

Invoke-ReplicationTraining "a8-candidate" 8 $repFinalSteps 10
$repFinalCheckpoint = Join-Path $repTraining "a8-candidate\final.pt"
$repFinalRepresentation = Join-Path $repRoot "representation-final.json"
if (-not (Test-Path -LiteralPath $repFinalRepresentation -PathType Leaf)) {
    & $repPython -m autodancer.training.representation `
        $repFinalCheckpoint `
        --output $repFinalRepresentation `
        --require-material-new-groups
    if ($LASTEXITCODE -ne 0) { throw "A8 failed the predeclared final representation gate" }
}
Register-ReplicationArtifact "a8-candidate" "representation-final" "diagnostic" `
    @($repFinalRepresentation) "A8 material-feature diagnostic at 30,720 transitions."

Write-ReplicationStatus "curve-evaluation"
Invoke-ReplicationEvaluation "a2-frozen" "curve-frozen" $repSource $repCurveSeeds 1500 `
    (Join-Path $repCurve "a2-frozen")
foreach ($arm in @("a2-finetune", "a8-candidate")) {
    foreach ($step in @(10240, 20480, 30720)) {
        $checkpoint = Join-Path $repTraining ("$arm\checkpoint-{0:D8}.pt" -f $step)
        $directory = Join-Path $repCurve ("$arm\step-{0:D8}" -f $step)
        Invoke-ReplicationEvaluation $arm "curve-$arm-step-$step" $checkpoint `
            $repCurveSeeds 1500 $directory
    }
}

$repComparison = Join-Path $repRoot "comparison.json"
& $repPython -m autodancer.training.architecture8_replication_compare `
    --root $repRoot `
    --output $repComparison
if ($LASTEXITCODE -ne 0) { throw "EXP-0007 curve comparison failed" }
$repCurveComparison = Get-Content -LiteralPath $repComparison -Raw | ConvertFrom-Json
if (-not $repCurveComparison.curve_gate_passed) {
    Register-ReplicationArtifact "aggregate" "curve-comparison" "comparison" `
        @($repComparison) "EXP-0007 stopped at the predeclared curve gate."
    Write-ReplicationStatus "completed" $repCurveComparison.decision
    exit 0
}

Write-ReplicationStatus "broad-evaluation"
$repBroadCheckpoints = @{
    "a2-frozen" = $repSource
    "a2-finetune" = Join-Path $repTraining "a2-finetune\final.pt"
    "a8-candidate" = $repFinalCheckpoint
}
foreach ($arm in @("a2-frozen", "a2-finetune", "a8-candidate")) {
    Invoke-ReplicationEvaluation $arm "broad-$arm" $repBroadCheckpoints[$arm] `
        $repBroadSeeds 3000 (Join-Path $repBroad $arm)
}

& $repPython -m autodancer.training.architecture8_replication_compare `
    --root $repRoot `
    --output $repComparison
if ($LASTEXITCODE -ne 0) { throw "EXP-0007 broad comparison failed" }
$repFinalComparison = Get-Content -LiteralPath $repComparison -Raw | ConvertFrom-Json
Register-ReplicationArtifact "aggregate" "broad-comparison" "comparison" `
    @($repComparison) "EXP-0007 final qualified A8 replication comparison."
Write-ReplicationStatus "completed" $repFinalComparison.decision
