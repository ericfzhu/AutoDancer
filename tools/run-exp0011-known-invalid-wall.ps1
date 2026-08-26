param()

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$expRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$expPython = Join-Path $expRepo ".venv\Scripts\python.exe"
$expGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$expMod = Join-Path $expRepo "mods\AutoDancer"
$expReward = Join-Path $expRepo "configs\reward-v2.json"
$expQualification = Join-Path $expRepo "runs\controller-qualification-current\qualification.json"
$expCheckpoint = Join-Path $expRepo "runs\corrected-a2-replication\training\seed-39001\checkpoint-00061440.pt"
$expRoot = Join-Path $expRepo "runs\known-invalid-wall-ablation"
$expEvaluation = Join-Path $expRoot "evaluation"
$expStatus = Join-Path $expRoot "pipeline-status.json"
$expSeeds = (61001..61024) -join ","
$expExpectedCheckpointHash = "dbea9bf9bcd65489057524b6cac82e5d6a6fdb55e9b290b1f0e8d24b73114b38"
$expArms = @(
    @{ Name = "current-11"; Contract = "current" },
    @{ Name = "known-invalid-wall-v1"; Contract = "known-invalid-wall-v1" }
)
$expTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-93001"; Mode = "stochastic"; PolicySeed = 93001 },
    @{ Name = "stochastic-93002"; Mode = "stochastic"; PolicySeed = 93002 }
)

function Write-ExperimentStatus {
    param([string]$Status, [string]$Arm = "", [string]$Trial = "", [string]$Error = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0011"
        status = $Status
        arm = $Arm
        trial = $Trial
        error = $Error
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $expStatus
}

function Test-ValidEvaluation {
    param([string]$Path, [string]$Mode, [int]$PolicySeed, [string]$Contract)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return (
            $report.controller_valid -eq $true -and
            [int]$report.worker_restarts -eq 0 -and
            @($report.infrastructure_events).Count -eq 0 -and
            $report.policy_mode -eq $Mode -and
            [int]$report.policy_seed -eq $PolicySeed -and
            $report.action_contract -eq $Contract -and
            (@($report.seeds) -join ",") -eq $expSeeds
        )
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $expRoot, $expEvaluation -Force | Out-Null
trap {
    Write-ExperimentStatus "failed" "" "" $_.Exception.Message
    throw
}

Set-Location -LiteralPath $expRepo
& $expPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }
$checkpointHash = (Get-FileHash -LiteralPath $expCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHash -ne $expExpectedCheckpointHash) { throw "Checkpoint hash mismatch" }
$qualification = Get-Content -LiteralPath $expQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) { throw "Controller qualification is missing or invalid" }
$cuda = & $expPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0011 requires CUDA" }

foreach ($arm in $expArms) {
    foreach ($trial in $expTrials) {
        $trialName = "$($arm.Name)-$($trial.Name)"
        $directory = Join-Path $expEvaluation $trialName
        $output = Join-Path $directory "report.json"
        if (Test-ValidEvaluation $output $trial.Mode $trial.PolicySeed $arm.Contract) { continue }
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        Write-ExperimentStatus "evaluating" $arm.Name $trial.Name
        $log = Join-Path $directory "console.log"
        & $expPython -m autodancer.training.baseline `
            --game-dir $expGame `
            --mod-dir $expMod `
            --checkpoint $expCheckpoint `
            --output $output `
            --num-instances 8 `
            --seeds $expSeeds `
            --max-steps 5000 `
            --policy-mode $trial.Mode `
            --policy-seed $trial.PolicySeed `
            --reward-config $expReward `
            --reward-lineage-version V2 `
            --action-contract $arm.Contract `
            --device cuda `
            --startup-timeout 60 `
            --turn-timeout 30 `
            --reset-timeout 60 `
            --affinity none `
            --dashboard 8765 `
            --trained-only `
            --experiment-id EXP-0011 `
            --experiment-arm $arm.Name `
            --trial-id $trialName `
            --controller-qualification $expQualification *>> $log
        if ($LASTEXITCODE -ne 0) { throw "Evaluation $trialName failed; see $log" }
        if (-not (Test-ValidEvaluation $output $trial.Mode $trial.PolicySeed $arm.Contract)) {
            throw "Evaluation $trialName produced invalid controller evidence"
        }
    }
}

& $expPython -m autodancer.training.action_contract_compare --root $expRoot
if ($LASTEXITCODE -ne 0) { throw "EXP-0011 comparison failed" }
Write-ExperimentStatus "reports-complete"
