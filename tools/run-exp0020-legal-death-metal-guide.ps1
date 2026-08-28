param([int]$QualificationPid = 0)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$guideRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$guidePython = Join-Path $guideRepo ".venv\Scripts\python.exe"
$guideGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$guideMod = Join-Path $guideRepo "mods\AutoDancer"
$guideQualification = Join-Path $guideRepo "runs\controller-qualification-player-health-only-world-ready\qualification.json"
$guideSource = Join-Path $guideRepo "runs\assisted-death-metal\training\seed-68002\final.pt"
$guideSourceHash = "bdc7d2e2d381cf7ab873d20ff10eafd6e1d15294988c9450d95f253cd3c3dda5"
$guideReward = Join-Path $guideRepo "configs\reward-death-metal-guide-v1.json"
$guideRewardHash = "650b2d3e8bfdb378bcdf2b63045f9ee2f71a4d229eb49af3474eb767a07732e4"
$guideRoot = Join-Path $guideRepo "runs\legal-death-metal-guide"
$guideTraining = Join-Path $guideRoot "training"
$guideEvaluation = Join-Path $guideRoot "evaluation"
$guideCalibration = Join-Path $guideRoot "calibration"
$guideStatus = Join-Path $guideRoot "pipeline-status.json"
$guideTrackingPath = (Join-Path $guideRepo ".runtime\mlflow\mlflow.db").Replace("\", "/")
$guideTrackingUri = "sqlite:///$guideTrackingPath"
$guideTrainingCandidates = ((80001..80256) -join ",")
$guideEvaluationCandidates = ((81001..81256) -join ",")
$guideTrainingSeeds = @(82001, 82002, 82003)
$guideTotalSteps = 122880
$guideModes = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-83001"; Mode = "stochastic"; PolicySeed = 83001 },
    @{ Name = "stochastic-83002"; Mode = "stochastic"; PolicySeed = 83002 }
)

function Write-GuideStatus {
    param([string]$Status, [string]$Trial = "", [string]$Mode = "", [string]$Error = "")
    [ordered]@{
        schema_version = 1
        experiment_id = "EXP-0020"
        status = $Status
        trial = $Trial
        mode = $Mode
        error = $Error
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $guideStatus -Encoding utf8
}

function Invoke-GuideChecked {
    param([string[]]$Arguments, [string]$Stage, [string]$Log)
    Write-Output ((Get-Date).ToString("o") + " START " + $Stage)
    if ($Log) {
        & $guidePython @Arguments *>> $Log
    } else {
        & $guidePython @Arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE" }
    Write-Output ((Get-Date).ToString("o") + " COMPLETE " + $Stage)
}

function Test-GuideTrainingComplete {
    param([string]$Directory)
    $final = Join-Path $Directory "final.pt"
    $lineage = Join-Path $Directory "lineage.json"
    if (-not (Test-Path $final) -or -not (Test-Path $lineage)) { return $false }
    try {
        if ((Get-Content $lineage -Raw | ConvertFrom-Json).status -ne "completed") { return $false }
        $step = & $guidePython -c (
            "import torch; print(torch.load(r'{0}', map_location='cpu', weights_only=False).get('global_step',0))" -f $final
        )
        return $LASTEXITCODE -eq 0 -and [int]$step -ge $guideTotalSteps
    } catch {
        return $false
    }
}

function Test-GuideEvaluationValid {
    param([string]$Path, [string]$Mode, [int]$PolicySeed, [string]$Seeds)
    if (-not (Test-Path $Path)) { return $false }
    try {
        $report = Get-Content $Path -Raw | ConvertFrom-Json
        return (
            $report.controller_valid -eq $true -and
            [int]$report.worker_restarts -eq 0 -and
            @($report.infrastructure_events).Count -eq 0 -and
            $report.policy_mode -eq $Mode -and
            [int]$report.policy_seed -eq $PolicySeed -and
            $report.curriculum_profile -eq "player20" -and
            [int]$report.curriculum_start_level -eq 4 -and
            [int]$report.curriculum_target_level -eq 5 -and
            (@($report.seeds) -join ",") -eq $Seeds
        )
    } catch {
        return $false
    }
}

function Select-DeathMetalSeeds {
    param([string]$ReportPath, [int]$Count)
    $report = Get-Content $ReportPath -Raw | ConvertFrom-Json
    if ($report.kind -ne "boss-identity-calibration-v1") {
        throw "Seed calibration report is not reset-only identity evidence"
    }
    if ($report.controller_valid -ne $true -or [int]$report.worker_restarts -ne 0) {
        throw "Seed calibration report contains controller failures"
    }
    return @(
        $report.results |
            Where-Object { [int]$_.boss_type -eq 2 } |
            Sort-Object { [int]$_.seed } |
            Select-Object -First $Count -ExpandProperty seed
    )
}

New-Item -ItemType Directory -Force -Path $guideRoot, $guideTraining, $guideEvaluation, $guideCalibration | Out-Null
trap {
    Write-GuideStatus "failed" "" "" $_.Exception.Message
    throw
}
Set-Location -LiteralPath $guideRepo

if ($QualificationPid -gt 0) {
    Write-Output ((Get-Date).ToString("o") + " waiting for qualification PID $QualificationPid")
    while (
        -not (Test-Path $guideQualification) -and
        (Get-Process -Id $QualificationPid -ErrorAction SilentlyContinue)
    ) {
        Start-Sleep -Seconds 30
    }
}
if (-not (Test-Path $guideQualification)) {
    throw "Fresh player-health-only controller qualification is missing"
}
$qualification = Get-Content $guideQualification -Raw | ConvertFrom-Json
if ($qualification.passed -ne $true) { throw "Fresh controller qualification did not pass" }

Invoke-GuideChecked -Stage "experiment registry validation" -Arguments @(
    "-m", "autodancer.experiments.cli", "validate"
) -Log ""
if ((Get-FileHash $guideSource -Algorithm SHA256).Hash.ToLowerInvariant() -ne $guideSourceHash) {
    throw "EXP-0020 source checkpoint hash mismatch"
}
if ((Get-FileHash $guideReward -Algorithm SHA256).Hash.ToLowerInvariant() -ne $guideRewardHash) {
    throw "EXP-0020 reward configuration hash mismatch"
}
$cuda = & $guidePython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0020 requires CUDA" }

$trainingCalibration = Join-Path $guideCalibration "training-candidates.json"
if (-not (Test-Path $trainingCalibration)) {
    Write-GuideStatus "calibrating-training-seeds"
    Invoke-GuideChecked -Stage "training seed boss-identity calibration" -Log (Join-Path $guideCalibration "training-console.log") -Arguments @(
        "-m", "autodancer.training.boss_identity", "--game-dir", $guideGame,
        "--mod-dir", $guideMod, "--output", $trainingCalibration,
        "--num-instances", "8", "--seeds", $guideTrainingCandidates,
        "--curriculum-start-level", "4", "--curriculum-target-level", "5",
        "--curriculum-profile", "player20", "--affinity", "none",
        "--controller-qualification", $guideQualification
    )
}
$trainingPool = Select-DeathMetalSeeds $trainingCalibration 48
if ($trainingPool.Count -ne 48) { throw "Expected 48 training Death Metal seeds" }
$trainingPoolArgument = $trainingPool -join ","
[ordered]@{
    schema_version = 1
    candidate_range = "80001-80256"
    boss_type = 2
    boss_name = "DEATH_METAL"
    rule = "ascending first 48 seeds with official boss_type 2"
    disclosure = "boss identity only"
    seeds = @($trainingPool | ForEach-Object { [int]$_ })
    source_report = $trainingCalibration
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $guideTraining "seed-selection.json") -Encoding utf8

$evaluationCalibration = Join-Path $guideCalibration "evaluation-candidates.json"
if (-not (Test-Path $evaluationCalibration)) {
    Write-GuideStatus "calibrating-evaluation-seeds"
    Invoke-GuideChecked -Stage "evaluation seed boss-identity calibration" -Log (Join-Path $guideCalibration "evaluation-console.log") -Arguments @(
        "-m", "autodancer.training.boss_identity", "--game-dir", $guideGame,
        "--mod-dir", $guideMod, "--output", $evaluationCalibration,
        "--num-instances", "8", "--seeds", $guideEvaluationCandidates,
        "--curriculum-start-level", "4", "--curriculum-target-level", "5",
        "--curriculum-profile", "player20", "--affinity", "none",
        "--controller-qualification", $guideQualification
    )
}
$heldoutSeeds = Select-DeathMetalSeeds $evaluationCalibration 24
if ($heldoutSeeds.Count -ne 24) { throw "Expected 24 held-out Death Metal seeds" }
$heldoutSeedArgument = $heldoutSeeds -join ","
[ordered]@{
    schema_version = 1
    candidate_range = "81001-81256"
    boss_type = 2
    boss_name = "DEATH_METAL"
    rule = "ascending first 24 seeds with official boss_type 2"
    disclosure = "boss identity only"
    seeds = @($heldoutSeeds | ForEach-Object { [int]$_ })
    source_report = $evaluationCalibration
} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $guideEvaluation "heldout-selection.json") -Encoding utf8

$checkpoints = [ordered]@{ parent = $guideSource }
foreach ($trainingSeed in $guideTrainingSeeds) {
    $trial = "seed-$trainingSeed"
    $directory = Join-Path $guideTraining $trial
    if (-not (Test-GuideTrainingComplete $directory)) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        Write-GuideStatus "training" $trial
        $arguments = @(
            "-m", "autodancer.training.train", "--game-dir", $guideGame,
            "--mod-dir", $guideMod, "--num-instances", "8", "--total-steps", "$guideTotalSteps",
            "--run-dir", $directory, "--device", "cuda", "--architecture", "8",
            "--seed", "$trainingSeed", "--checkpoint-interval", "30720", "--evaluation-interval", "0",
            "--max-turns", "500", "--action-contract", "map-navigation-prior-v1",
            "--training-seed-pool", $trainingPoolArgument, "--curriculum-start-level", "4",
            "--curriculum-target-level", "5", "--curriculum-profile", "player20",
            "--reward-config", $guideReward, "--reward-lineage-version", "DeathMetalGuideV1",
            "--freeze-base-updates", "10", "--affinity", "none", "--experiment-id", "EXP-0020",
            "--experiment-arm", "a8-player20-legal-guide", "--trial-id", $trial,
            "--mlflow-tracking-uri", $guideTrackingUri, "--controller-qualification", $guideQualification,
            "--dashboard", "8765"
        )
        $latest = Join-Path $directory "latest.pt"
        if (Test-Path $latest) { $arguments += @("--resume", $latest) }
        else { $arguments += @("--initialize-from", $guideSource) }
        Invoke-GuideChecked -Stage "training $trial" -Log (Join-Path $directory "console.log") -Arguments $arguments
        if (-not (Test-GuideTrainingComplete $directory)) { throw "Incomplete training evidence for $trial" }
    }
    $checkpoints[$trial] = Join-Path $directory "final.pt"
}

foreach ($checkpointEntry in $checkpoints.GetEnumerator()) {
    foreach ($mode in $guideModes) {
        $directory = Join-Path $guideEvaluation "$($checkpointEntry.Key)\$($mode.Name)"
        $output = Join-Path $directory "report.json"
        if (Test-GuideEvaluationValid $output $mode.Mode $mode.PolicySeed $heldoutSeedArgument) { continue }
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
        Write-GuideStatus "evaluating" $checkpointEntry.Key $mode.Name
        Invoke-GuideChecked -Stage "evaluation $($checkpointEntry.Key) $($mode.Name)" -Log (Join-Path $directory "console.log") -Arguments @(
            "-m", "autodancer.training.baseline", "--game-dir", $guideGame,
            "--mod-dir", $guideMod, "--checkpoint", $checkpointEntry.Value, "--output", $output,
            "--num-instances", "8", "--seeds", $heldoutSeedArgument, "--max-steps", "500",
            "--policy-mode", $mode.Mode, "--policy-seed", "$($mode.PolicySeed)", "--trained-only",
            "--device", "cuda", "--reward-config", $guideReward,
            "--reward-lineage-version", "DeathMetalGuideV1", "--action-contract", "map-navigation-prior-v1",
            "--curriculum-start-level", "4", "--curriculum-target-level", "5",
            "--curriculum-profile", "player20", "--affinity", "none",
            "--experiment-id", "EXP-0020", "--experiment-arm", "a8-player20-legal-guide",
            "--trial-id", "$($checkpointEntry.Key)-$($mode.Name)", "--mlflow-tracking-uri", $guideTrackingUri,
            "--controller-qualification", $guideQualification, "--dashboard", "8765"
        )
        if (-not (Test-GuideEvaluationValid $output $mode.Mode $mode.PolicySeed $heldoutSeedArgument)) {
            throw "Invalid evaluation evidence for $($checkpointEntry.Key)/$($mode.Name)"
        }
    }
}

Write-GuideStatus "comparing"
Invoke-GuideChecked -Stage "EXP-0020 comparison" -Log "" -Arguments @(
    "-m", "autodancer.training.death_metal_guide_compare", "--root", $guideRoot
)
Write-GuideStatus "complete"
