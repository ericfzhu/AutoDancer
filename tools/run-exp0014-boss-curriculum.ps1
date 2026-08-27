param()

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$currRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$currPython = Join-Path $currRepo ".venv\Scripts\python.exe"
$currGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$currMod = Join-Path $currRepo "mods\AutoDancer"
$currReward = Join-Path $currRepo "configs\reward-v2.json"
$currQualification = Join-Path $currRepo "runs\controller-qualification-current\qualification.json"
$currSource = Join-Path $currRepo "runs\corrected-a2-replication\training\seed-39001\checkpoint-00061440.pt"
$currSourceHash = "dbea9bf9bcd65489057524b6cac82e5d6a6fdb55e9b290b1f0e8d24b73114b38"
$currRoot = Join-Path $currRepo "runs\boss-curriculum-tactical"
$currTraining = Join-Path $currRoot "training"
$currEvaluation = Join-Path $currRoot "evaluation"
$currStatus = Join-Path $currRoot "pipeline-status.json"
$currTrainingSeeds = "64001-64032"
$currEvaluationSeeds = (66001..66024) -join ","
$currTrainingSeed = 65001
$currTotalSteps = 122880
$currArms = @(
    @{ Name = "a2-boss-curriculum"; Architecture = 2; Freeze = 0 },
    @{ Name = "a8-boss-curriculum"; Architecture = 8; Freeze = 10 }
)
$currTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-96001"; Mode = "stochastic"; PolicySeed = 96001 },
    @{ Name = "stochastic-96002"; Mode = "stochastic"; PolicySeed = 96002 }
)

function Write-CurriculumStatus {
    param([string]$Status, [string]$Arm = "", [string]$Trial = "", [string]$Error = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0014"
        status = $Status
        arm = $Arm
        trial = $Trial
        error = $Error
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $currStatus
}

function Test-CompletedLineage {
    param([string]$Directory)
    $manifest = Join-Path $Directory "lineage.json"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { return $false }
    try {
        return (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).status -eq "completed"
    } catch {
        return $false
    }
}

function Test-CompletedTraining {
    param([string]$Directory)
    $final = Join-Path $Directory "final.pt"
    if (-not (Test-Path -LiteralPath $final -PathType Leaf)) { return $false }
    if (-not (Test-CompletedLineage $Directory)) { return $false }
    $step = & $currPython -c (
        "import torch; print(torch.load(r'{0}', map_location='cpu', weights_only=False).get('global_step',0))" -f $final
    )
    return $LASTEXITCODE -eq 0 -and [int]$step -ge $currTotalSteps
}

function Test-ValidCurriculumReport {
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
            [int]$report.curriculum_start_level -eq 4 -and
            [int]$report.curriculum_target_level -eq 5 -and
            (@($report.seeds) -join ",") -eq $currEvaluationSeeds
        )
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Path $currRoot, $currTraining, $currEvaluation -Force | Out-Null
trap {
    Write-CurriculumStatus "failed" "" "" $_.Exception.Message
    throw
}

Set-Location -LiteralPath $currRepo
& $currPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }
$sourceHash = (Get-FileHash -LiteralPath $currSource -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceHash -ne $currSourceHash) { throw "Source checkpoint hash mismatch" }
$qualification = Get-Content -LiteralPath $currQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) { throw "Controller qualification is missing or invalid" }
$cuda = & $currPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0014 requires CUDA" }

foreach ($arm in $currArms) {
    $run = Join-Path $currTraining $arm.Name
    if (Test-CompletedTraining $run) { continue }
    New-Item -ItemType Directory -Path $run -Force | Out-Null
    Write-CurriculumStatus "training" $arm.Name
    $log = Join-Path $run "console.log"
    $arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $currGame,
        "--mod-dir", $currMod,
        "--num-instances", "8",
        "--total-steps", "$currTotalSteps",
        "--run-dir", $run,
        "--architecture", "$($arm.Architecture)",
        "--device", "cuda",
        "--seed", "$currTrainingSeed",
        "--reward-config", $currReward,
        "--reward-lineage-version", "V2",
        "--action-contract", "current",
        "--training-seed-pool", $currTrainingSeeds,
        "--curriculum-start-level", "4",
        "--curriculum-target-level", "5",
        "--max-turns", "1000",
        "--freeze-base-updates", "$($arm.Freeze)",
        "--checkpoint-interval", "30720",
        "--evaluation-interval", "0",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765",
        "--experiment-id", "EXP-0014",
        "--experiment-arm", $arm.Name,
        "--trial-id", "seed-$currTrainingSeed",
        "--controller-qualification", $currQualification
    )
    $latest = Join-Path $run "latest.pt"
    if (Test-Path -LiteralPath $latest -PathType Leaf) {
        $arguments += @("--resume", $latest)
    } else {
        $arguments += @("--fine-tune-from", $currSource)
    }
    Add-Content -LiteralPath $log -Value (
        "`n=== launch $($arm.Name) at $((Get-Date).ToString('o')) ==="
    )
    & $currPython @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Training $($arm.Name) failed; see $log" }
    if (-not (Test-CompletedTraining $run)) {
        throw "Training $($arm.Name) did not produce a valid completed run"
    }
}

foreach ($arm in $currArms) {
    $checkpoint = Join-Path $currTraining "$($arm.Name)\final.pt"
    foreach ($trial in $currTrials) {
        $directory = Join-Path $currEvaluation "$($arm.Name)\$($trial.Name)"
        $output = Join-Path $directory "report.json"
        if (Test-ValidCurriculumReport $output $trial.Mode $trial.PolicySeed) { continue }
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        Write-CurriculumStatus "evaluating" $arm.Name $trial.Name
        $log = Join-Path $directory "console.log"
        & $currPython -m autodancer.training.baseline `
            --game-dir $currGame `
            --mod-dir $currMod `
            --checkpoint $checkpoint `
            --output $output `
            --num-instances 8 `
            --seeds $currEvaluationSeeds `
            --max-steps 1000 `
            --policy-mode $trial.Mode `
            --policy-seed $trial.PolicySeed `
            --reward-config $currReward `
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
            --experiment-id EXP-0014 `
            --experiment-arm $arm.Name `
            --trial-id $trial.Name `
            --controller-qualification $currQualification *>> $log
        if ($LASTEXITCODE -ne 0) {
            throw "Evaluation $($arm.Name)/$($trial.Name) failed; see $log"
        }
        if (-not (Test-ValidCurriculumReport $output $trial.Mode $trial.PolicySeed)) {
            throw "Evaluation $($arm.Name)/$($trial.Name) produced invalid evidence"
        }
    }
}

& $currPython -m autodancer.training.boss_curriculum_compare --root $currRoot
if ($LASTEXITCODE -ne 0) { throw "EXP-0014 comparison failed" }
Write-CurriculumStatus "reports-complete"
