param(
    [int]$QualificationPid = 0
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$expRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$expPython = Join-Path $expRepo ".venv\Scripts\python.exe"
$expGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$expMod = Join-Path $expRepo "mods\AutoDancer"
$expReward = Join-Path $expRepo "configs\reward-v2.json"
$expQualification = Join-Path $expRepo "runs\controller-qualification-current\qualification.json"
$expCheckpoint = Join-Path $expRepo "runs\reward-v2-250k\final.pt"
$expRoot = Join-Path $expRepo "runs\corrected-a2-replication"
$expTraining = Join-Path $expRoot "training\seed-39001"
$expEvaluation = Join-Path $expRoot "evaluation"
$expStatus = Join-Path $expRoot "pipeline-status.json"
$expSeeds = (60001..60024) -join ","
$expExpectedSourceHash = "9fa39f242995555a8b9b3c9556253d0c7ab19945e9cb80292bf312388d6eda5d"
$expTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-92001"; Mode = "stochastic"; PolicySeed = 92001 },
    @{ Name = "stochastic-92002"; Mode = "stochastic"; PolicySeed = 92002 }
)
$expCheckpoints = @(
    @{ Step = 61440; Path = Join-Path $expTraining "checkpoint-00061440.pt" },
    @{ Step = 122880; Path = Join-Path $expTraining "checkpoint-00122880.pt" },
    @{ Step = 250880; Path = Join-Path $expTraining "final.pt" }
)

function Write-ExperimentStatus {
    param([string]$Status, [int]$Step = 0, [string]$Trial = "", [string]$Error = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0010"
        status = $Status
        checkpoint_step = $Step
        trial = $Trial
        error = $Error
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $expStatus
}

function Test-ValidEvaluation {
    param([string]$Path, [string]$Mode, [int]$PolicySeed)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return (
            $report.controller_valid -eq $true -and
            [int]$report.worker_restarts -eq 0 -and
            @($report.infrastructure_events).Count -eq 0 -and
            $report.policy_mode -eq $Mode -and
            [int]$report.policy_seed -eq $PolicySeed -and
            (@($report.seeds) -join ",") -eq $expSeeds
        )
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $expRoot, $expTraining, $expEvaluation -Force | Out-Null
trap {
    Write-ExperimentStatus "failed" 0 "" $_.Exception.Message
    throw
}

Write-ExperimentStatus "waiting-for-controller-qualification"
if ($QualificationPid -gt 0) {
    while (Get-Process -Id $QualificationPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}
if (-not (Test-Path -LiteralPath $expQualification -PathType Leaf)) {
    throw "Fresh controller qualification report is missing"
}
$qualification = Get-Content -LiteralPath $expQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) {
    throw "Fresh controller qualification did not pass: $($qualification.failure)"
}

Set-Location -LiteralPath $expRepo
& $expPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }
$sourceHash = (Get-FileHash -LiteralPath $expCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceHash -ne $expExpectedSourceHash) { throw "Source checkpoint hash mismatch" }
$cuda = & $expPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0010 requires CUDA" }

$finalCheckpoint = Join-Path $expTraining "final.pt"
if (-not (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf)) {
    Write-ExperimentStatus "training"
    $trainingLog = Join-Path $expTraining "console.log"
    & $expPython -m autodancer.training.train `
        --game-dir $expGame `
        --mod-dir $expMod `
        --num-instances 8 `
        --total-steps 250880 `
        --run-dir $expTraining `
        --initialize-from $expCheckpoint `
        --architecture 2 `
        --seed 39001 `
        --rollout-length 128 `
        --sequence-length 32 `
        --checkpoint-interval 10240 `
        --evaluation-interval 0 `
        --reward-config $expReward `
        --reward-lineage-version V2 `
        --action-contract current `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --experiment-id EXP-0010 `
        --experiment-arm corrected-a2 `
        --trial-id seed-39001 `
        --controller-qualification $expQualification *>> $trainingLog
    if ($LASTEXITCODE -ne 0) { throw "EXP-0010 training failed; see $trainingLog" }
}

foreach ($checkpoint in $expCheckpoints) {
    if (-not (Test-Path -LiteralPath $checkpoint.Path -PathType Leaf)) {
        throw "Missing declared checkpoint $($checkpoint.Path)"
    }
    foreach ($trial in $expTrials) {
        $trialName = "step-$($checkpoint.Step)-$($trial.Name)"
        $directory = Join-Path $expEvaluation $trialName
        $output = Join-Path $directory "report.json"
        if (Test-ValidEvaluation $output $trial.Mode $trial.PolicySeed) { continue }
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        Write-ExperimentStatus "evaluating" $checkpoint.Step $trial.Name
        $log = Join-Path $directory "console.log"
        & $expPython -m autodancer.training.baseline `
            --game-dir $expGame `
            --mod-dir $expMod `
            --checkpoint $checkpoint.Path `
            --output $output `
            --num-instances 8 `
            --seeds $expSeeds `
            --max-steps 5000 `
            --policy-mode $trial.Mode `
            --policy-seed $trial.PolicySeed `
            --reward-config $expReward `
            --reward-lineage-version V2 `
            --action-contract current `
            --device cuda `
            --startup-timeout 60 `
            --turn-timeout 30 `
            --reset-timeout 60 `
            --affinity none `
            --dashboard 8765 `
            --trained-only `
            --experiment-id EXP-0010 `
            --experiment-arm corrected-a2 `
            --trial-id $trialName `
            --controller-qualification $expQualification *>> $log
        if ($LASTEXITCODE -ne 0) { throw "Evaluation $trialName failed; see $log" }
        if (-not (Test-ValidEvaluation $output $trial.Mode $trial.PolicySeed)) {
            throw "Evaluation $trialName produced invalid controller evidence"
        }
    }
}

Write-ExperimentStatus "reports-complete"
