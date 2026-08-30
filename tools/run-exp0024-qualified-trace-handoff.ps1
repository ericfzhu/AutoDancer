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

function Write-HandoffStatus(
    [string]$Status,
    [string]$Trial = "",
    [string]$Reason = "",
    [string]$Checkpoint = "",
    [string]$ErrorText = ""
) {
    $payload = [ordered]@{
        schema_version = 1
        status = $Status
        selected_trial = $Trial
        selection_reason = $Reason
        selected_checkpoint = $Checkpoint
        error = $ErrorText
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $temporary = "$handoffStatus.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $handoffStatus -Force
}

$selectedTrial = ""
$selectionReason = ""
$selectedCheckpoint = ""
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

    $trainingSelection = Join-Path $experiment "training\seed-selection.json"
    $excludedSelection = Join-Path $experiment "evaluation\heldout-selection.json"
    $publishedStatus = Join-Path $traceRoot "status.json"
    if (Test-Path -LiteralPath $publishedStatus -PathType Leaf) {
        $existing = Get-Content -LiteralPath $publishedStatus -Raw | ConvertFrom-Json
        if ([bool]$existing.valid -and [int]$existing.qualified_distinct_seed_count -ge 3) {
            $selectedCheckpoint = [string]$existing.checkpoint
            Write-HandoffStatus "complete" $selectedTrial $selectionReason $selectedCheckpoint
            return
        }
    }

    $trialDirectory = Join-Path $experiment "training\$selectedTrial"
    $checkpointNames = @("final.pt", "checkpoint-00092160.pt", "checkpoint-00061440.pt")
    $checkpointCandidates = @(
        $checkpointNames |
            ForEach-Object { Join-Path $trialDirectory $_ } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    )
    if ($checkpointCandidates.Count -eq 0) {
        throw "Selected EXP-0024 trial has no searchable checkpoint"
    }
    $candidateFailures = [System.Collections.Generic.List[object]]::new()
    foreach ($checkpoint in $checkpointCandidates) {
        $candidateName = [System.IO.Path]::GetFileNameWithoutExtension($checkpoint)
        $candidateRoot = Join-Path $traceRoot "candidates\$selectedTrial\$candidateName"
        Write-HandoffStatus "searching" $selectedTrial $selectionReason $checkpoint
        try {
            & (Join-Path $PSScriptRoot "run-qualified-trace-search.ps1") `
                -GameDir $GameDir `
                -Checkpoint $checkpoint `
                -SeedSelection $trainingSelection `
                -ExcludedSeedSelection $excludedSelection `
                -RunDir $candidateRoot
            if ($LASTEXITCODE -ne 0) {
                throw "Qualified trace search exited with code $LASTEXITCODE"
            }
            $candidateStatusPath = Join-Path $candidateRoot "status.json"
            $candidateStatus = Get-Content -LiteralPath $candidateStatusPath -Raw | ConvertFrom-Json
            if (-not [bool]$candidateStatus.valid -or [int]$candidateStatus.qualified_distinct_seed_count -lt 3) {
                throw "Qualified trace search did not meet the three-seed gate"
            }
            $selectedCheckpoint = $checkpoint
            $published = [ordered]@{}
            foreach ($property in $candidateStatus.psobject.Properties) {
                $published[$property.Name] = $property.Value
            }
            $published["selected_trial"] = $selectedTrial
            $published["selection_reason"] = $selectionReason
            $published["candidate_run"] = $candidateRoot
            $published["checkpoint_candidates"] = $checkpointCandidates
            $published["failed_candidates"] = @($candidateFailures)
            $temporaryStatus = Join-Path $traceRoot ".status.json.tmp"
            $published | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $temporaryStatus -Encoding utf8
            Move-Item -LiteralPath $temporaryStatus -Destination $publishedStatus -Force
            break
        } catch {
            $candidateError = $_.Exception.Message
            $competenceMiss = $candidateError -match (
                "No successful full-reset traces|Trace search exhausted|Qualified traces cover only"
            )
            if (-not $competenceMiss) {
                throw
            }
            $candidateFailures.Add(
                [ordered]@{
                    checkpoint = $checkpoint
                    error = $candidateError
                }
            )
        }
    }
    if ([string]::IsNullOrWhiteSpace($selectedCheckpoint)) {
        $detail = @($candidateFailures | ForEach-Object { "$($_.checkpoint): $($_.error)" }) -join "; "
        throw "Every competence-window checkpoint failed qualified trace search: $detail"
    }
    Write-HandoffStatus "complete" $selectedTrial $selectionReason $selectedCheckpoint
} catch {
    Write-HandoffStatus "failed" $selectedTrial $selectionReason $selectedCheckpoint $_.Exception.Message
    throw
}
