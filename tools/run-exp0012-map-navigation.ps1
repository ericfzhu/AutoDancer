param()

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$navRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$navPython = Join-Path $navRepo ".venv\Scripts\python.exe"
$navGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$navMod = Join-Path $navRepo "mods\AutoDancer"
$navReward = Join-Path $navRepo "configs\reward-v2.json"
$navQualification = Join-Path $navRepo "runs\controller-qualification-current\qualification.json"
$navCheckpoint = Join-Path $navRepo "runs\corrected-a2-replication\training\seed-39001\checkpoint-00061440.pt"
$navRoot = Join-Path $navRepo "runs\map-navigation-ablation"
$navEvaluation = Join-Path $navRoot "evaluation"
$navStatus = Join-Path $navRoot "pipeline-status.json"
$navSeeds = (62001..62024) -join ","
$navExpectedCheckpointHash = "dbea9bf9bcd65489057524b6cac82e5d6a6fdb55e9b290b1f0e8d24b73114b38"
$navArms = @(
    @{ Name = "current-11"; Contract = "current" },
    @{ Name = "map-navigation-prior-v1"; Contract = "map-navigation-prior-v1" }
)
$navTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-94001"; Mode = "stochastic"; PolicySeed = 94001 },
    @{ Name = "stochastic-94002"; Mode = "stochastic"; PolicySeed = 94002 }
)

function Write-NavigationStatus {
    param([string]$Status, [string]$Arm = "", [string]$Trial = "", [string]$Error = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0012"
        status = $Status
        arm = $Arm
        trial = $Trial
        error = $Error
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $navStatus
}

function Test-ValidNavigationReport {
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
            (@($report.seeds) -join ",") -eq $navSeeds
        )
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $navRoot, $navEvaluation -Force | Out-Null
trap {
    Write-NavigationStatus "failed" "" "" $_.Exception.Message
    throw
}

Set-Location -LiteralPath $navRepo
& $navPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }
$checkpointHash = (Get-FileHash -LiteralPath $navCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($checkpointHash -ne $navExpectedCheckpointHash) { throw "Checkpoint hash mismatch" }
$qualification = Get-Content -LiteralPath $navQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) { throw "Controller qualification is missing or invalid" }
$cuda = & $navPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0012 requires CUDA" }

foreach ($arm in $navArms) {
    foreach ($trial in $navTrials) {
        $trialName = "$($arm.Name)-$($trial.Name)"
        $directory = Join-Path $navEvaluation $trialName
        $output = Join-Path $directory "report.json"
        if (Test-ValidNavigationReport $output $trial.Mode $trial.PolicySeed $arm.Contract) {
            continue
        }
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        Write-NavigationStatus "evaluating" $arm.Name $trial.Name
        $log = Join-Path $directory "console.log"
        & $navPython -m autodancer.training.baseline `
            --game-dir $navGame `
            --mod-dir $navMod `
            --checkpoint $navCheckpoint `
            --output $output `
            --num-instances 8 `
            --seeds $navSeeds `
            --max-steps 5000 `
            --policy-mode $trial.Mode `
            --policy-seed $trial.PolicySeed `
            --reward-config $navReward `
            --reward-lineage-version V2 `
            --action-contract $arm.Contract `
            --device cuda `
            --startup-timeout 60 `
            --turn-timeout 30 `
            --reset-timeout 60 `
            --affinity none `
            --dashboard 8765 `
            --trained-only `
            --experiment-id EXP-0012 `
            --experiment-arm $arm.Name `
            --trial-id $trialName `
            --controller-qualification $navQualification *>> $log
        if ($LASTEXITCODE -ne 0) { throw "Evaluation $trialName failed; see $log" }
        if (-not (Test-ValidNavigationReport $output $trial.Mode $trial.PolicySeed $arm.Contract)) {
            throw "Evaluation $trialName produced invalid controller evidence"
        }
    }
}

& $navPython -m autodancer.training.map_navigation_compare --root $navRoot
if ($LASTEXITCODE -ne 0) { throw "EXP-0012 comparison failed" }
Write-NavigationStatus "reports-complete"
