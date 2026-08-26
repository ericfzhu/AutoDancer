$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$spRepo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$spPython = Join-Path $spRepo ".venv\Scripts\python.exe"
$spGame = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$spMod = Join-Path $spRepo "mods\AutoDancer"
$spReward = Join-Path $spRepo "configs\reward-v2.json"
$spQualification = Join-Path $spRepo "runs\controller-qualification\qualification.json"
$spExpectedQualificationHash = "126533ab66c31b699709c6e32bfaf9099fbeb4e96b651c68f1f6d602b8572888"
$spRoot = Join-Path $spRepo "runs\stochastic-policy-calibration"
$spEvaluation = Join-Path $spRoot "evaluation"
$spStatus = Join-Path $spRoot "pipeline-status.json"
$spSeeds = (57001..57024) -join ","
$spCheckpoints = @{
    "a2-frozen" = Join-Path $spRepo "runs\reward-v2-250k\final.pt"
    "a2-continuation" = Join-Path $spRepo "runs\architecture8-horizon\training\a2-continuation\final.pt"
    "a8-continuation" = Join-Path $spRepo "runs\architecture8-horizon\training\a8-continuation\final.pt"
}
$spExpectedHashes = @{
    "a2-frozen" = "9fa39f242995555a8b9b3c9556253d0c7ab19945e9cb80292bf312388d6eda5d"
    "a2-continuation" = "671e9e59a909a3e98556127909a78e9c1846692533f78b4ad79d1b389df318ba"
    "a8-continuation" = "fddd4c6eb8526ba1990f3227d40f31fbaed16dd0b522e62f80fd3a54f39aa9c0"
}
$spTrials = @(
    @{ Name = "deterministic"; Mode = "deterministic"; PolicySeed = 0 },
    @{ Name = "stochastic-91001"; Mode = "stochastic"; PolicySeed = 91001 },
    @{ Name = "stochastic-91002"; Mode = "stochastic"; PolicySeed = 91002 }
)

function Write-CalibrationStatus {
    param([string]$Status, [string]$Arm = "", [string]$Trial = "", [string]$Decision = "")
    @{
        schema_version = 1
        experiment_id = "EXP-0009"
        status = $Status
        arm = $Arm
        trial = $Trial
        decision = $Decision
        evaluation_seeds = 57001..57024
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $spStatus
}

function Test-ValidReport {
    param([string]$Path, [string]$Mode, [int]$PolicySeed)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $report = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $actualSeeds = @($report.seeds) -join ","
        return (
            $report.controller_valid -eq $true -and
            [int]$report.worker_restarts -eq 0 -and
            @($report.infrastructure_events).Count -eq 0 -and
            $report.policy_mode -eq $Mode -and
            [int]$report.policy_seed -eq $PolicySeed -and
            $actualSeeds -eq $spSeeds
        )
    } catch {
        return $false
    }
}

function Invoke-CalibrationEvaluation {
    param([string]$Arm, [string]$Trial, [string]$Mode, [int]$PolicySeed)
    $directory = Join-Path $spEvaluation "$Arm\$Trial"
    $output = Join-Path $directory "report.json"
    if (Test-ValidReport $output $Mode $PolicySeed) { return }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    if (Test-Path -LiteralPath $output -PathType Leaf) {
        $stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
        Move-Item -LiteralPath $output -Destination (Join-Path $directory "report.invalid-$stamp.json")
    }
    $log = Join-Path $directory "console.log"
    Add-Content -LiteralPath $log -Value "`n=== $Arm $Trial at $((Get-Date).ToString('o')) ==="
    $arguments = @(
        "-m", "autodancer.training.baseline",
        "--game-dir", $spGame,
        "--mod-dir", $spMod,
        "--checkpoint", $spCheckpoints[$Arm],
        "--output", $output,
        "--num-instances", "8",
        "--seeds", $spSeeds,
        "--max-steps", "5000",
        "--policy-mode", $Mode,
        "--policy-seed", "$PolicySeed",
        "--reward-config", $spReward,
        "--reward-lineage-version", "V2",
        "--action-contract", "current",
        "--device", "cuda",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765",
        "--trained-only",
        "--experiment-id", "EXP-0009",
        "--experiment-arm", $Arm,
        "--trial-id", $Trial,
        "--controller-qualification", $spQualification
    )
    & $spPython @arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation $Arm/$Trial failed; see $log" }
    if (-not (Test-ValidReport $output $Mode $PolicySeed)) {
        throw "Evaluation $Arm/$Trial produced invalid controller evidence"
    }
}

New-Item -ItemType Directory -Path $spRoot, $spEvaluation -Force | Out-Null
Write-CalibrationStatus "preflight"
trap {
    Write-CalibrationStatus "failed" "" "" $_.Exception.Message
    throw
}

foreach ($required in @($spPython, $spReward, $spQualification) + $spCheckpoints.Values) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing file: $required" }
}
foreach ($required in @($spGame, $spMod)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) { throw "Missing directory: $required" }
}
foreach ($arm in $spCheckpoints.Keys) {
    $actual = (Get-FileHash -LiteralPath $spCheckpoints[$arm] -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $spExpectedHashes[$arm]) { throw "Checkpoint hash mismatch for $arm" }
}
$qualificationHash = (Get-FileHash -LiteralPath $spQualification -Algorithm SHA256).Hash.ToLowerInvariant()
if ($qualificationHash -ne $spExpectedQualificationHash) { throw "Qualification report hash mismatch" }
$cuda = & $spPython -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $cuda.Trim() -ne "True") { throw "EXP-0009 requires CUDA" }

Set-Location -LiteralPath $spRepo
& $spPython -m autodancer.experiments.cli validate
if ($LASTEXITCODE -ne 0) { throw "Experiment registry validation failed" }

foreach ($arm in @("a2-frozen", "a2-continuation", "a8-continuation")) {
    foreach ($trial in $spTrials) {
        Write-CalibrationStatus "evaluating" $arm $trial.Name
        Invoke-CalibrationEvaluation $arm $trial.Name $trial.Mode $trial.PolicySeed
    }
}

Write-CalibrationStatus "comparison"
$comparison = Join-Path $spRoot "comparison.json"
& $spPython -m autodancer.training.stochastic_policy_compare --root $spRoot --output $comparison
if ($LASTEXITCODE -ne 0) { throw "EXP-0009 comparison is invalid or incomplete" }
$result = Get-Content -LiteralPath $comparison -Raw | ConvertFrom-Json

$lineage = Join-Path $spRoot "artifact-lineage\comparison"
if (-not (Test-Path -LiteralPath (Join-Path $lineage "lineage.json") -PathType Leaf)) {
    & $spPython -m autodancer.experiments.cli attach `
        --experiment-id EXP-0009 `
        --arm aggregate `
        --trial execution-mode-comparison `
        --stage comparison `
        --run-dir $lineage `
        --game-dir $spGame `
        --mod-dir $spMod `
        --qualification-report $spQualification `
        --device cuda `
        --artifact $comparison `
        --notes "Predeclared deterministic versus reproducible stochastic execution comparison."
    if ($LASTEXITCODE -ne 0) { throw "Could not attach comparison lineage" }
}
Write-CalibrationStatus "completed" "" "" $result.decision
