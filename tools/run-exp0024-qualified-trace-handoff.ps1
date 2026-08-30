param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$ExperimentRoot = "runs\legal-death-metal-full-boss-potential",
    [string]$RunDir = "runs\qualified-death-metal-trace-search",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepositoryPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}
if ($PollSeconds -le 0) { throw "PollSeconds must be positive" }

$experiment = Resolve-RepositoryPath $ExperimentRoot
$traceRoot = Resolve-RepositoryPath $RunDir
$pipelineStatus = Join-Path $experiment "pipeline-status.json"
$comparisonPath = Join-Path $experiment "comparison.json"
$handoffStatus = Join-Path $traceRoot "handoff-status.json"
[System.IO.Directory]::CreateDirectory($traceRoot) | Out-Null

function Write-HandoffStatus([string]$Status, [string]$Trial = "", [string]$Reason = "", [string]$ErrorText = "") {
    $payload = [ordered]@{
        schema_version = 1
        status = $Status
        selected_trial = $Trial
        selection_reason = $Reason
        error = $ErrorText
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $temporary = "$handoffStatus.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $handoffStatus -Force
}

$selectedTrial = ""
$selectionReason = ""
try {
    Write-HandoffStatus "waiting-for-exp0024"
    while ($true) {
        if (-not (Test-Path -LiteralPath $pipelineStatus -PathType Leaf)) {
            throw "EXP-0024 pipeline status does not exist: $pipelineStatus"
        }
        $pipeline = Get-Content -LiteralPath $pipelineStatus -Raw | ConvertFrom-Json
        if ([string]$pipeline.status -eq "complete") { break }
        if ([string]$pipeline.status -in @("failed", "error")) {
            throw "EXP-0024 failed: $($pipeline.error)"
        }
        Start-Sleep -Seconds $PollSeconds
    }

    if (Test-Path -LiteralPath $comparisonPath -PathType Leaf) {
        $comparison = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
        if (-not [string]::IsNullOrWhiteSpace([string]$comparison.selected_trial)) {
            $selectedTrial = [string]$comparison.selected_trial
            $selectionReason = "predeclared-heldout-gate-winner"
        }
    }
    if ([string]::IsNullOrWhiteSpace($selectedTrial)) {
        $candidates = foreach ($directory in Get-ChildItem (Join-Path $experiment "training") -Directory) {
            $metrics = Join-Path $directory.FullName "metrics.jsonl"
            $checkpoint = Join-Path $directory.FullName "final.pt"
            if (-not (Test-Path -LiteralPath $metrics -PathType Leaf) -or -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                continue
            }
            $last = Get-Content -LiteralPath $metrics -Tail 1 | ConvertFrom-Json
            $outcomes = $last.curriculum_schedule_state.outcomes.fixed
            $completions = [int]$outcomes.curriculum_complete
            $episodes = $completions + [int]$outcomes.dead + [int]$outcomes.time_limit
            [pscustomobject]@{
                trial = $directory.Name
                completions = $completions
                completion_rate = $completions / [math]::Max($episodes, 1)
            }
        }
        $winner = $candidates | Sort-Object completions, completion_rate -Descending | Select-Object -First 1
        if ($null -eq $winner -or [int]$winner.completions -le 0) {
            throw "No EXP-0024 training checkpoint produced a legal full-boss completion"
        }
        $selectedTrial = [string]$winner.trial
        $selectionReason = "training-only-full-boss-completions"
    }

    $checkpoint = Join-Path $experiment "training\$selectedTrial\final.pt"
    $trainingSelection = Join-Path $experiment "training\seed-selection.json"
    $excludedSelection = Join-Path $experiment "evaluation\heldout-selection.json"
    Write-HandoffStatus "searching" $selectedTrial $selectionReason
    & (Join-Path $PSScriptRoot "run-qualified-trace-search.ps1") `
        -GameDir $GameDir `
        -Checkpoint $checkpoint `
        -SeedSelection $trainingSelection `
        -ExcludedSeedSelection $excludedSelection `
        -RunDir $traceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Qualified trace search exited with code $LASTEXITCODE"
    }
    Write-HandoffStatus "complete" $selectedTrial $selectionReason
} catch {
    Write-HandoffStatus "failed" $selectedTrial $selectionReason $_.Exception.Message
    throw
}
