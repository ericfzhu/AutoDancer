param(
    [Parameter(Mandatory = $true)][string]$GameDir,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$SeedSelection,
    [Parameter(Mandatory = $true)][string]$ExcludedSeedSelection,
    [Parameter(Mandatory = $true)][string]$RunDir,
    [string]$ModDir = "",
    [int[]]$PolicySeeds = (96001..96032),
    [int]$NumInstances = 8,
    [int]$MaxSteps = 500,
    [int]$MinQualifiedDistinctSeeds = 3,
    [string]$ActionContract = "map-navigation-prior-v1",
    [string]$RewardConfig = "configs/reward-death-metal-potential-v5.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
function Resolve-RepositoryPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository Python environment does not exist: $python"
}
if ([string]::IsNullOrWhiteSpace($ModDir)) {
    $ModDir = Join-Path $repoRoot "mods\AutoDancer"
}
$resolvedCheckpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
$resolvedSeedSelection = (Resolve-Path -LiteralPath $SeedSelection).Path
$resolvedExcludedSelection = (Resolve-Path -LiteralPath $ExcludedSeedSelection).Path
$resolvedReward = (Resolve-Path -LiteralPath (Resolve-RepositoryPath $RewardConfig)).Path
$selection = Get-Content -LiteralPath $resolvedSeedSelection -Raw | ConvertFrom-Json
$excluded = Get-Content -LiteralPath $resolvedExcludedSelection -Raw | ConvertFrom-Json
$trainingSeeds = @($selection.seeds | ForEach-Object { [int]$_ })
$excludedSeeds = [System.Collections.Generic.HashSet[int]]::new()
foreach ($value in @($excluded.seeds)) {
    [void]$excludedSeeds.Add([int]$value)
}
if ($trainingSeeds.Count -eq 0 -or $trainingSeeds.Count -ne @($trainingSeeds | Select-Object -Unique).Count) {
    throw "Training seed selection must be non-empty and unique"
}
$leaked = @($trainingSeeds | Where-Object { $excludedSeeds.Contains($_) })
if ($leaked.Count -ne 0) {
    throw "Training trace search overlaps excluded evaluation seeds: $($leaked -join ',')"
}
if ($PolicySeeds.Count -eq 0 -or $PolicySeeds.Count -ne @($PolicySeeds | Select-Object -Unique).Count) {
    throw "Policy seeds must be non-empty and unique"
}
if ($NumInstances -le 0 -or $MaxSteps -le 0 -or $MinQualifiedDistinctSeeds -le 0) {
    throw "NumInstances, MaxSteps, and MinQualifiedDistinctSeeds must be positive"
}

$traceRoot = Resolve-RepositoryPath $RunDir
[System.IO.Directory]::CreateDirectory($traceRoot) | Out-Null
$reports = [System.Collections.Generic.List[string]]::new()
$seedCsv = $trainingSeeds -join ","
function Write-TraceBank([System.Collections.Generic.List[string]]$SourceReports, [string]$Output) {
    $buildArguments = [System.Collections.Generic.List[string]]::new()
    foreach ($value in @("-m", "autodancer.training.demonstration_replay", "build")) {
        $buildArguments.Add($value)
    }
    foreach ($reportPath in $SourceReports) {
        $buildArguments.Add("--episodes")
        $buildArguments.Add($reportPath)
    }
    $buildArguments.Add("--output")
    $buildArguments.Add($Output)
    & $python @buildArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Demonstration-bank construction failed"
    }
}

$bank = Join-Path $traceRoot "demonstration-bank.json"
$candidateDistinctSeeds = @()
foreach ($policySeed in $PolicySeeds) {
    $reportDir = Join-Path $traceRoot ("reports\stochastic-{0}" -f $policySeed)
    [System.IO.Directory]::CreateDirectory($reportDir) | Out-Null
    $report = Join-Path $reportDir "report.json"
    if (Test-Path -LiteralPath $report -PathType Leaf) {
        $existing = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
        $existingSeeds = @($existing.seeds | ForEach-Object { [int]$_ })
        $sameSeeds = ($existingSeeds -join ",") -eq $seedCsv
        $sameCheckpoint = [System.IO.Path]::GetFullPath([string]$existing.checkpoint) -eq $resolvedCheckpoint
        if (-not ($existing.controller_valid -and $sameSeeds -and $sameCheckpoint -and [int]$existing.policy_seed -eq $policySeed)) {
            throw "Existing trace-search report does not match the requested run: $report"
        }
    } else {
        & $python -m autodancer.training.baseline `
            --game-dir $GameDir `
            --mod-dir $ModDir `
            --checkpoint $resolvedCheckpoint `
            --output $report `
            --num-instances $NumInstances `
            --seeds $seedCsv `
            --max-steps $MaxSteps `
            --policy-seed $policySeed `
            --policy-mode stochastic `
            --reward-config $resolvedReward `
            --policy-feedback-reward-config $resolvedReward `
            --trained-only `
            --device cuda `
            --action-contract $ActionContract `
            --curriculum-start-level 4 `
            --curriculum-target-level 5 `
            --curriculum-profile player20
        if ($LASTEXITCODE -ne 0) {
            throw "Trace-search evaluation failed for policy seed $policySeed"
        }
    }
    $reports.Add($report)
    Write-TraceBank $reports $bank
    $candidateBank = Get-Content -LiteralPath $bank -Raw | ConvertFrom-Json
    $candidateDistinctSeeds = @(
        $candidateBank.traces |
            ForEach-Object { [int]$_.seed } |
            Sort-Object -Unique
    )
    if ($candidateDistinctSeeds.Count -ge $MinQualifiedDistinctSeeds) {
        break
    }
}

$bankPayload = Get-Content -LiteralPath $bank -Raw | ConvertFrom-Json
if (@($bankPayload.traces).Count -eq 0) {
    throw "No successful full-reset traces were found on the declared training seed bank"
}
if ($candidateDistinctSeeds.Count -lt $MinQualifiedDistinctSeeds) {
    throw (
        "Trace search exhausted {0} policy streams but found successes on only {1} distinct training seeds; at least {2} are required" -f `
            $reports.Count, $candidateDistinctSeeds.Count, $MinQualifiedDistinctSeeds
    )
}

$qualification = Join-Path $traceRoot "qualification.json"
$recurrentDemonstrations = Join-Path $traceRoot "recurrent-demonstrations.npz"
& $python -m autodancer.training.demonstration_replay qualify `
    --game-dir $GameDir `
    --mod-dir $ModDir `
    --bank $bank `
    --output $qualification `
    --num-instances $NumInstances `
    --action-contract $ActionContract `
    --policy-feedback-reward-config $resolvedReward `
    --recurrent-output $recurrentDemonstrations
if ($LASTEXITCODE -ne 0) {
    throw "Fresh-launch live trace qualification failed"
}
$qualificationPayload = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
$qualifiedDistinctSeeds = @(
    $qualificationPayload.results |
        Where-Object { [bool]$_.valid } |
        ForEach-Object { [int]$_.seed } |
        Sort-Object -Unique
)
if (-not [bool]$qualificationPayload.valid) {
    throw "Fresh-launch live trace qualification report is invalid"
}
if ($qualifiedDistinctSeeds.Count -lt $MinQualifiedDistinctSeeds) {
    throw (
        "Qualified traces cover only {0} distinct training seeds; at least {1} are required" -f `
            $qualifiedDistinctSeeds.Count, $MinQualifiedDistinctSeeds
    )
}
$status = [ordered]@{
    schema_version = 1
    status = "complete"
    checkpoint = $resolvedCheckpoint
    training_seed_selection = $resolvedSeedSelection
    excluded_seed_selection = $resolvedExcludedSelection
    policy_seeds = @($reports | ForEach-Object {
        [int]((Split-Path (Split-Path $_ -Parent) -Leaf) -replace '^stochastic-', '')
    })
    maximum_policy_seeds = @($PolicySeeds)
    reports = @($reports)
    bank = $bank
    bank_sha256 = [string]$bankPayload.bank_sha256
    trace_count = @($bankPayload.traces).Count
    qualification = $qualification
    recurrent_demonstrations = $recurrentDemonstrations
    recurrent_demonstrations_manifest = "$recurrentDemonstrations.manifest.json"
    qualified_trace_count = [int]$qualificationPayload.qualified_trace_count
    qualified_distinct_seed_count = $qualifiedDistinctSeeds.Count
    qualified_distinct_seeds = $qualifiedDistinctSeeds
    minimum_qualified_distinct_seeds = $MinQualifiedDistinctSeeds
    valid = [bool]$qualificationPayload.valid
}
$temporaryStatus = Join-Path $traceRoot ".status.json.tmp"
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryStatus -Encoding utf8
Move-Item -LiteralPath $temporaryStatus -Destination (Join-Path $traceRoot "status.json") -Force
$status | ConvertTo-Json -Depth 8
