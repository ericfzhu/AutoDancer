param()

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$bossRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bossPython = Join-Path $bossRepo ".venv\Scripts\python.exe"
$bossGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$bossMod = Join-Path $bossRepo "mods\AutoDancer"
$bossReward = Join-Path $bossRepo "configs\reward-v2.json"
$bossQualification = Join-Path $bossRepo "runs\controller-qualification-current\qualification.json"
$bossCheckpoint = Join-Path $bossRepo "runs\corrected-a2-replication\training\seed-39001\checkpoint-00061440.pt"
$bossRoot = Join-Path $bossRepo "runs\boss-feasibility"
$bossEvaluation = Join-Path $bossRoot "evaluation"
$bossStatus = Join-Path $bossRoot "pipeline-status.json"
$bossSeeds = (63001..63024) -join ","
$bossExpectedCheckpointHash = "dbea9bf9bcd65489057524b6cac82e5d6a6fdb55e9b290b1f0e8d24b73114b38"
$bossTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-95001"; Mode = "stochastic"; PolicySeed = 95001 },
    @{ Name = "stochastic-95002"; Mode = "stochastic"; PolicySeed = 95002 }
)

function Write-BossStatus {
    param([string]$Status, [string]$Trial = "", [string]$Error = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0013"
        status = $Status
        trial = $Trial
        error = $Error
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $bossStatus
}

function Test-ValidBossReport {
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
            $report.action_contract -eq "current" -and
            [int]$report.curriculum_start_level -eq 4 -and
            [int]$report.curriculum_target_level -eq 5 -and
            (@($report.seeds) -join ",") -eq $bossSeeds
        )
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $bossRoot, $bossEvaluation -Force | Out-Null
trap {
    Write-BossStatus "failed" "" $_.Exception.Message
    throw
}

Set-Location -LiteralPath $bossRepo
& $bossPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }
$checkpointHash = (Get-FileHash -LiteralPath $bossCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHash -ne $bossExpectedCheckpointHash) { throw "Checkpoint hash mismatch" }
$qualification = Get-Content -LiteralPath $bossQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) { throw "Controller qualification is missing or invalid" }
$cuda = & $bossPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0013 requires CUDA" }

foreach ($trial in $bossTrials) {
    $directory = Join-Path $bossEvaluation $trial.Name
    $output = Join-Path $directory "report.json"
    if (Test-ValidBossReport $output $trial.Mode $trial.PolicySeed) { continue }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    Write-BossStatus "evaluating" $trial.Name
    $log = Join-Path $directory "console.log"
    & $bossPython -m autodancer.training.baseline `
        --game-dir $bossGame `
        --mod-dir $bossMod `
        --checkpoint $bossCheckpoint `
        --output $output `
        --num-instances 8 `
        --seeds $bossSeeds `
        --max-steps 1000 `
        --policy-mode $trial.Mode `
        --policy-seed $trial.PolicySeed `
        --reward-config $bossReward `
        --reward-lineage-version V2 `
        --action-contract current `
        --curriculum-start-level 4 `
        --curriculum-target-level 5 `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only `
        --experiment-id EXP-0013 `
        --experiment-arm boss-start-a2 `
        --trial-id $trial.Name `
        --controller-qualification $bossQualification *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation $($trial.Name) failed; see $log" }
    if (-not (Test-ValidBossReport $output $trial.Mode $trial.PolicySeed)) {
        throw "Evaluation $($trial.Name) produced invalid controller evidence"
    }
}

& $bossPython -m autodancer.training.curriculum_compare --root $bossRoot
if ($LASTEXITCODE -ne 0) { throw "EXP-0013 comparison failed" }
Write-BossStatus "reports-complete"
