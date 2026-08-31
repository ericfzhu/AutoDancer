param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$TraceSearchRoot = "runs\qualified-death-metal-trace-search",
    [string]$RunDir = "runs\qualified-death-metal-tail1-warm-boundary",
    [int[]]$TrainingSeeds = @(97001, 97002, 97003),
    [int]$TotalSteps = 61440,
    [int]$RequestedTailActions = 1,
    [int[]]$CandidateTailActions = @(),
    [int[]]$CalibrationPolicySeeds = @(0, 98001, 98002, 98003, 98004),
    [string]$ExperimentId = "EXP-0027",
    [string]$ExperimentArm = "a8-live-calibrated-tail1",
    [string]$TrainingDistributionVersion = "qualified-trace-tail-v5",
    [string]$ControllerQualification = "runs\controller-qualification-player-health-only-world-ready-memory-controlled\qualification.json",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepositoryPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}
if ($TrainingSeeds.Count -ne 3 -or @($TrainingSeeds | Select-Object -Unique).Count -ne 3) {
    throw "The trace-tail pilot requires exactly three distinct optimizer seeds"
}
if ($TotalSteps -le 0 -or $RequestedTailActions -le 0 -or $PollSeconds -le 0) {
    throw "TotalSteps, RequestedTailActions, and PollSeconds must be positive"
}
if ($CalibrationPolicySeeds.Count -eq 0 -or $CalibrationPolicySeeds[0] -ne 0) {
    throw "CalibrationPolicySeeds must begin with deterministic policy seed 0"
}
if (@($CalibrationPolicySeeds | Select-Object -Unique).Count -ne $CalibrationPolicySeeds.Count) {
    throw "CalibrationPolicySeeds must be unique"
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$traceRoot = Resolve-RepositoryPath $TraceSearchRoot
$root = Resolve-RepositoryPath $RunDir
$searchStatusPath = Join-Path $traceRoot "status.json"
$pipelineStatusPath = Join-Path $root "pipeline-status.json"
$reward = Join-Path $repoRoot "configs\reward-death-metal-potential-v5.json"
$mod = Join-Path $repoRoot "mods\AutoDancer"
$qualification = Resolve-RepositoryPath $ControllerQualification
$trackingPath = (Join-Path $repoRoot ".runtime\mlflow\mlflow.db").Replace("\", "/")
$trackingUri = "sqlite:///$trackingPath"
$pilotStartedAt = (Get-Date).ToUniversalTime().ToString("o")
[System.IO.Directory]::CreateDirectory($root) | Out-Null

function Write-PilotStatus([string]$Status, [string]$Trial = "", [string]$ErrorText = "") {
    $payload = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        process_id = $PID
        process_started_at = $pilotStartedAt
        status = $Status
        trial = $Trial
        error = $ErrorText
        heartbeat_unix_seconds = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $temporary = "$pipelineStatusPath.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $pipelineStatusPath -Force
}

function Test-CurrentQualification {
    if (-not (Test-Path -LiteralPath $qualification -PathType Leaf)) { return $false }
    try {
        $payload = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
        if (-not [bool]$payload.passed) { return $false }
        $configuration = $payload.configuration
        $preflight = $payload.phases.preflight
        if ([int]$preflight.schema_version -ne 10) { return $false }
        if (
            [System.IO.Path]::GetFullPath([string]$configuration.game_dir) -ne
                [System.IO.Path]::GetFullPath($GameDir) -or
            [System.IO.Path]::GetFullPath([string]$configuration.mod_dir) -ne
                [System.IO.Path]::GetFullPath($mod)
        ) { return $false }
        foreach ($entry in $preflight.mod_files.PSObject.Properties) {
            $path = Join-Path $mod ([string]$entry.Name)
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
            if ($actual -ne ([string]$entry.Value).ToLowerInvariant()) { return $false }
        }
        return $true
    } catch {
        return $false
    }
}

function Get-QualificationHeartbeatAge {
    $progress = Join-Path (Split-Path -Parent $qualification) "qualification-progress.json"
    $soakTraces = Join-Path (Split-Path -Parent $qualification) "soak-traces"
    $candidates = [System.Collections.Generic.List[datetime]]::new()
    if (Test-Path -LiteralPath $progress -PathType Leaf) {
        $candidates.Add((Get-Item -LiteralPath $progress).LastWriteTimeUtc)
    }
    if (Test-Path -LiteralPath $soakTraces -PathType Container) {
        Get-ChildItem -LiteralPath $soakTraces -Filter "*.jsonl" -File | ForEach-Object {
            $candidates.Add($_.LastWriteTimeUtc)
        }
    }
    if (-not $candidates.Count) { return $null }
    $latest = ($candidates | Sort-Object -Descending | Select-Object -First 1)
    return ([datetime]::UtcNow - $latest).TotalSeconds
}

function Read-TrialSummary([string]$Directory, [int]$Seed, [string[]]$Reports) {
    $metrics = Join-Path $Directory "metrics.jsonl"
    if (-not (Test-Path -LiteralPath $metrics -PathType Leaf)) {
        throw "Trace-tail trial lacks metric evidence: $Directory"
    }
    $payloads = @($Reports | ForEach-Object { Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json })
    if (@($payloads | Where-Object { -not [bool]$_.controller_valid }).Count) {
        throw "Trace-tail final-policy evaluation contains controller contamination: $Directory"
    }
    $episodes = @($payloads | ForEach-Object { $_.trained.results } | Where-Object {
        [string]$_.natural_prefix.kind -eq "qualified-live-trace-prefix-v1"
    })
    if ($episodes.Count -eq 0) { throw "Trace-tail trial contains no frozen final-policy episodes: $Directory" }
    $completed = @($episodes | Where-Object { [string]$_.status -eq "curriculum_complete" })
    $last = Get-Content -LiteralPath $metrics -Tail 1 | ConvertFrom-Json
    $lossesFinite = @("policy_loss", "value_loss", "entropy", "approx_kl") | ForEach-Object {
        $value = [double]$last.$_
        -not [double]::IsNaN($value) -and -not [double]::IsInfinity($value)
    }
    [pscustomobject]@{
        trial = "seed-$Seed"
        training_seed = $Seed
        episodes = $episodes.Count
        completions = $completed.Count
        completion_rate = $completed.Count / [math]::Max($episodes.Count, 1)
        distinct_completion_seeds = @($completed | ForEach-Object { [int]$_.seed } | Sort-Object -Unique)
        mean_completion_turns = if ($completed.Count) {
            [double](($completed | Measure-Object turns -Average).Average)
        } else { 0.0 }
        worker_restarts = [int]$last.worker_restarts + [int](($payloads | Measure-Object worker_restarts -Sum).Sum)
        finite_losses = -not ($lossesFinite -contains $false)
        final_global_step = [int]$last.global_step
        final_checkpoint = Join-Path $Directory "final.pt"
        evaluation_reports = $Reports
    }
}

try {
    Write-PilotStatus "waiting-for-controller-qualification"
    while (-not (Test-CurrentQualification)) {
        $heartbeatAge = Get-QualificationHeartbeatAge
        if ($null -ne $heartbeatAge -and $heartbeatAge -gt [math]::Max(20 * $PollSeconds, 900)) {
            throw "Controller qualification heartbeat is stale after $([math]::Round($heartbeatAge, 1)) seconds"
        }
        if (Test-Path -LiteralPath $qualification -PathType Leaf) {
            $candidate = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
            $candidateTime = (Get-Item -LiteralPath $qualification).LastWriteTimeUtc
            if (-not [bool]$candidate.passed -and $candidateTime -gt [datetime]::Parse($pilotStartedAt)) {
                throw "Controller qualification failed: $($candidate.failure)"
            }
        }
        Write-PilotStatus "waiting-for-controller-qualification"
        Start-Sleep -Seconds $PollSeconds
    }
    Write-PilotStatus "waiting-for-qualified-traces"
    while (-not (Test-Path -LiteralPath $searchStatusPath -PathType Leaf)) {
        Write-PilotStatus "waiting-for-qualified-traces"
        $handoff = Join-Path $traceRoot "handoff-status.json"
        if (Test-Path -LiteralPath $handoff -PathType Leaf) {
            $handoffPayload = Get-Content -LiteralPath $handoff -Raw | ConvertFrom-Json
            if ([string]$handoffPayload.status -eq "failed") { throw "Trace search failed: $($handoffPayload.error)" }
            if ([string]$handoffPayload.status -eq "waiting-for-exp0024") {
                $heartbeatAge = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [long]$handoffPayload.heartbeat_unix_seconds
                $heartbeatTimeout = [math]::Max(4 * $PollSeconds, 120)
                if ($heartbeatAge -gt $heartbeatTimeout) {
                    throw "Trace-search handoff heartbeat is stale after $([math]::Round($heartbeatAge, 1)) seconds"
                }
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
    $search = Get-Content -LiteralPath $searchStatusPath -Raw | ConvertFrom-Json
    if (-not [bool]$search.valid -or [int]$search.qualified_distinct_seed_count -lt 3) {
        throw "Trace search did not produce three distinct qualified training seeds"
    }
    $bank = [string]$search.bank
    $traceQualification = [string]$search.qualification
    $sourceCheckpoint = [string]$search.checkpoint
    $bankPayload = Get-Content -LiteralPath $bank -Raw | ConvertFrom-Json
    $minimumTraceLength = ($bankPayload.traces | ForEach-Object { @($_.action_sequence).Count } | Measure-Object -Minimum).Minimum
    $maximumTailActions = [int]$minimumTraceLength - 1
    if ($maximumTailActions -le 0) {
        throw "Qualified traces are too short to leave a legal replay prefix"
    }
    $tailCandidates = if ($CandidateTailActions.Count) {
        @($CandidateTailActions | Where-Object { $_ -gt 0 -and $_ -le $maximumTailActions } |
            Sort-Object -Unique)
    } else {
        @([math]::Min($RequestedTailActions, $maximumTailActions))
    }
    if ($tailCandidates.Count -eq 0) {
        throw "No requested trace-tail candidate leaves a legal replay prefix"
    }
    $gameSeeds = @($search.qualified_distinct_seeds | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    $gameSeedCsv = $gameSeeds -join ","
    $evaluationModes = @(
        [pscustomobject]@{ name = "deterministic"; mode = "deterministic"; policy_seed = 0 },
        [pscustomobject]@{ name = "stochastic-98001"; mode = "stochastic"; policy_seed = 98001 },
        [pscustomobject]@{ name = "stochastic-98002"; mode = "stochastic"; policy_seed = 98002 }
    )
    $calibrationModes = @(
        $CalibrationPolicySeeds | ForEach-Object {
            if ($_ -eq 0) {
                [pscustomobject]@{ name = "deterministic"; mode = "deterministic"; policy_seed = 0 }
            } else {
                [pscustomobject]@{ name = "stochastic-$_"; mode = "stochastic"; policy_seed = [int]$_ }
            }
        }
    )

    # Measure the declared source policy at this exact live handoff before
    # spending transitions on it. Reverse curricula require an intermediate
    # success boundary; action likelihood is useful diagnosis but gameplay is
    # the authoritative calibration.
    $calibrationSummaries = [System.Collections.Generic.List[object]]::new()
    $selectedCalibration = $null
    foreach ($candidateTail in $tailCandidates) {
        $candidateReports = [System.Collections.Generic.List[string]]::new()
        foreach ($mode in $calibrationModes) {
            $calibrationDirectory = Join-Path $root "calibration\source\tail-$candidateTail\$($mode.name)"
            [System.IO.Directory]::CreateDirectory($calibrationDirectory) | Out-Null
            $report = Join-Path $calibrationDirectory "report.json"
            if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
                Write-PilotStatus "calibrating-source" "tail-$candidateTail-$($mode.name)"
                & $python -m autodancer.training.baseline `
                    --game-dir $GameDir `
                    --mod-dir $mod `
                    --checkpoint $sourceCheckpoint `
                    --output $report `
                    --num-instances 8 `
                    --seeds $gameSeedCsv `
                    --max-steps 64 `
                    --policy-mode $mode.mode `
                    --policy-seed $mode.policy_seed `
                    --trained-only `
                    --source-reference `
                    --device cuda `
                    --reward-config $reward `
                    --policy-feedback-reward-config $reward `
                    --reward-lineage-version DeathMetalPotentialV5 `
                    --action-contract map-navigation-prior-v1 `
                    --curriculum-start-level 4 `
                    --curriculum-target-level 5 `
                    --curriculum-profile player20 `
                    --trace-prefix-bank $bank `
                    --trace-prefix-qualification $traceQualification `
                    --trace-prefix-tail-actions $candidateTail `
                    --trace-prefix-recurrent-state warm `
                    --affinity none `
                    --experiment-id $ExperimentId `
                    --experiment-arm $ExperimentArm `
                    --trial-id "source-tail-$candidateTail-$($mode.name)" `
                    --mlflow-tracking-uri $trackingUri `
                    --controller-qualification $qualification `
                    --dashboard 8765
                if ($LASTEXITCODE -ne 0) {
                    throw "Source calibration failed for tail $candidateTail/$($mode.name)"
                }
            }
            $payload = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            if (
                -not [bool]$payload.controller_valid -or
                -not [bool]$payload.source_reference -or
                [System.IO.Path]::GetFullPath([string]$payload.checkpoint) -ne [System.IO.Path]::GetFullPath($sourceCheckpoint) -or
                (@($payload.seeds | ForEach-Object { [int]$_ }) -join ",") -ne $gameSeedCsv -or
                [string]$payload.policy_mode -ne [string]$mode.mode -or
                [int]$payload.policy_seed -ne [int]$mode.policy_seed -or
                [int]$payload.trace_prefix.tail_actions -ne $candidateTail -or
                [string]$payload.trace_prefix.bank_sha256 -ne [string]$bankPayload.bank_sha256
            ) {
                throw "Source calibration report identity mismatch: $report"
            }
            $candidateReports.Add($report)
        }
        $candidatePayloads = @(
            $candidateReports | ForEach-Object {
                Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json
            }
        )
        $candidateEpisodes = @(
            $candidatePayloads | ForEach-Object { $_.trained.results } | Where-Object {
                [string]$_.natural_prefix.kind -eq "qualified-live-trace-prefix-v1"
            }
        )
        if ($candidateEpisodes.Count -ne $calibrationModes.Count * $gameSeeds.Count) {
            throw "Source calibration did not produce the complete tail-$candidateTail episode matrix"
        }
        $candidateCompletions = @(
            $candidateEpisodes | Where-Object { [string]$_.status -eq "curriculum_complete" }
        )
        $candidateRate = $candidateCompletions.Count / $candidateEpisodes.Count
        $summary = [pscustomobject]@{
            tail_actions = [int]$candidateTail
            episodes = $candidateEpisodes.Count
            completions = $candidateCompletions.Count
            completion_rate = $candidateRate
            distinct_completion_seeds = @(
                $candidateCompletions | ForEach-Object { [int]$_.seed } | Sort-Object -Unique
            )
            reports = $candidateReports.ToArray()
            inside_intermediate_competence_band = (
                $candidateRate -ge 0.10 -and $candidateRate -le 0.90
            )
        }
        $calibrationSummaries.Add($summary)
        if ($summary.inside_intermediate_competence_band) {
            $selectedCalibration = $summary
            break
        }
    }
    $calibrationPayload = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        source_checkpoint = $sourceCheckpoint
        candidate_tail_actions = $tailCandidates
        policy_modes = $calibrationModes
        qualified_training_seeds = $gameSeeds
        candidates = @($calibrationSummaries)
        selected_tail_actions = if ($null -eq $selectedCalibration) { $null } else { $selectedCalibration.tail_actions }
    }
    $calibrationTemporary = Join-Path $root ".calibration.json.tmp"
    $calibrationPayload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $calibrationTemporary -Encoding utf8
    Move-Item -LiteralPath $calibrationTemporary -Destination (Join-Path $root "calibration.json") -Force
    if ($null -eq $selectedCalibration) {
        $rates = @($calibrationSummaries | ForEach-Object {
            "tail-$($_.tail_actions)=$($_.completions)/$($_.episodes)"
        }) -join ", "
        throw "No trace-tail candidate lies inside the 10-90 percent live competence band: $rates"
    }
    $tailActions = [int]$selectedCalibration.tail_actions
    $sourceCalibrationEpisodes = [int]$selectedCalibration.episodes
    $sourceCalibrationCompletions = [int]$selectedCalibration.completions
    $sourceCalibrationRate = [double]$selectedCalibration.completion_rate
    $sourceCalibrationReports = @($selectedCalibration.reports)

    $summaries = [System.Collections.Generic.List[object]]::new()
    foreach ($trainingSeed in $TrainingSeeds) {
        $trial = "seed-$trainingSeed"
        $directory = Join-Path $root "training\$trial"
        [System.IO.Directory]::CreateDirectory($directory) | Out-Null
        $final = Join-Path $directory "final.pt"
        if (-not (Test-Path -LiteralPath $final -PathType Leaf)) {
            Write-PilotStatus "training" $trial
            $arguments = @(
                "-m", "autodancer.training.train", "--game-dir", $GameDir,
                "--mod-dir", $mod, "--num-instances", "8",
                "--total-steps", "$TotalSteps", "--run-dir", $directory, "--device", "cuda",
                "--architecture", "8", "--seed", "$trainingSeed", "--checkpoint-interval", "30720",
                "--evaluation-interval", "0", "--max-turns", "500", "--action-contract", "map-navigation-prior-v1",
                "--training-seed-pool", $gameSeedCsv, "--curriculum-start-level", "4",
                "--curriculum-target-level", "5", "--curriculum-profile", "player20",
                "--reward-config", $reward, "--policy-feedback-reward-config", $reward,
                "--reward-lineage-version", "DeathMetalPotentialV5", "--freeze-base-updates", "10",
                "--trace-prefix-bank", $bank, "--trace-prefix-qualification", $traceQualification,
                "--trace-prefix-tail-actions", "$tailActions", "--trace-prefix-recurrent-state", "warm",
                "--initialize-from", $sourceCheckpoint, "--affinity", "none", "--experiment-id", $ExperimentId,
                "--experiment-arm", $ExperimentArm, "--trial-id", $trial,
                "--training-level-distribution-version", $TrainingDistributionVersion,
                "--mlflow-tracking-uri", $trackingUri, "--controller-qualification", $qualification,
                "--dashboard", "8765"
            )
            $latest = Join-Path $directory "latest.pt"
            if (Test-Path -LiteralPath $latest -PathType Leaf) {
                $initializeIndex = [array]::IndexOf($arguments, "--initialize-from")
                $arguments[$initializeIndex] = "--resume"
                $arguments[$initializeIndex + 1] = $latest
            }
            & $python @arguments
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $final -PathType Leaf)) {
                throw "Trace-tail training failed for $trial"
            }
        }
        $evaluationReports = [System.Collections.Generic.List[string]]::new()
        foreach ($mode in $evaluationModes) {
            $evaluationDirectory = Join-Path $root "evaluation\$trial\$($mode.name)"
            [System.IO.Directory]::CreateDirectory($evaluationDirectory) | Out-Null
            $report = Join-Path $evaluationDirectory "report.json"
            if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
                Write-PilotStatus "evaluating" "$trial-$($mode.name)"
                & $python -m autodancer.training.baseline `
                    --game-dir $GameDir `
                    --mod-dir $mod `
                    --checkpoint $final `
                    --output $report `
                    --num-instances 8 `
                    --seeds $gameSeedCsv `
                    --max-steps 64 `
                    --policy-mode $mode.mode `
                    --policy-seed $mode.policy_seed `
                    --trained-only `
                    --device cuda `
                    --reward-config $reward `
                    --policy-feedback-reward-config $reward `
                    --reward-lineage-version DeathMetalPotentialV5 `
                    --action-contract map-navigation-prior-v1 `
                    --curriculum-start-level 4 `
                    --curriculum-target-level 5 `
                    --curriculum-profile player20 `
                    --trace-prefix-bank $bank `
                    --trace-prefix-qualification $traceQualification `
                    --trace-prefix-tail-actions $tailActions `
                    --trace-prefix-recurrent-state warm `
                    --affinity none `
                    --experiment-id $ExperimentId `
                    --experiment-arm $ExperimentArm `
                    --trial-id "$trial-$($mode.name)" `
                    --mlflow-tracking-uri $trackingUri `
                    --controller-qualification $qualification `
                    --dashboard 8765
                if ($LASTEXITCODE -ne 0) { throw "Trace-tail evaluation failed for $trial/$($mode.name)" }
            }
            $payload = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
            if (
                -not [bool]$payload.controller_valid -or
                [System.IO.Path]::GetFullPath([string]$payload.checkpoint) -ne [System.IO.Path]::GetFullPath($final) -or
                (@($payload.seeds | ForEach-Object { [int]$_ }) -join ",") -ne $gameSeedCsv -or
                [string]$payload.policy_mode -ne [string]$mode.mode -or
                [int]$payload.policy_seed -ne [int]$mode.policy_seed -or
                [string]$payload.trace_prefix.bank_sha256 -ne [string]$bankPayload.bank_sha256
            ) {
                throw "Trace-tail evaluation report identity mismatch: $report"
            }
            $evaluationReports.Add($report)
        }
        $summaries.Add((Read-TrialSummary $directory $trainingSeed $evaluationReports.ToArray()))
    }

    $allEpisodes = ($summaries | Measure-Object episodes -Sum).Sum
    $allCompletions = ($summaries | Measure-Object completions -Sum).Sum
    $distinctSuccesses = @($summaries | ForEach-Object { $_.distinct_completion_seeds } | Sort-Object -Unique)
    $reproducibleTrials = @($summaries | Where-Object { $_.completion_rate -ge 0.05 }).Count
    $passed = (
        $allCompletions / [math]::Max($allEpisodes, 1) -ge 0.10 -and
        $reproducibleTrials -ge 2 -and
        $distinctSuccesses.Count -ge 3 -and
        @($summaries | Where-Object { $_.completions -le 0 }).Count -eq 0 -and
        @($summaries | Where-Object { $_.worker_restarts -ne 0 -or -not $_.finite_losses }).Count -eq 0
    )
    $selected = $summaries | Sort-Object @{Expression={@($_.distinct_completion_seeds).Count};Descending=$true}, @{Expression={$_.completion_rate};Descending=$true}, @{Expression={-$_.mean_completion_turns};Descending=$true} | Select-Object -First 1
    $comparison = [ordered]@{
        schema_version = 1
        experiment_id = $ExperimentId
        source_checkpoint = $sourceCheckpoint
        bank = $bank
        qualification = $traceQualification
        tail_actions = $tailActions
        qualified_training_seeds = $gameSeeds
        source_calibration = [ordered]@{
            episodes = $sourceCalibrationEpisodes
            completions = $sourceCalibrationCompletions
            completion_rate = $sourceCalibrationRate
            distinct_completion_seeds = @($selectedCalibration.distinct_completion_seeds)
            reports = $sourceCalibrationReports
            inside_intermediate_competence_band = $true
            candidate_matrix = @($calibrationSummaries)
        }
        trials = @($summaries)
        aggregate = [ordered]@{
            episodes = $allEpisodes
            completions = $allCompletions
            completion_rate = $allCompletions / [math]::Max($allEpisodes, 1)
            distinct_completion_seeds = $distinctSuccesses
            reproducible_trials = $reproducibleTrials
        }
        gate = [ordered]@{
            passed = $passed
            completion_rate_at_least_10_percent = $allCompletions / [math]::Max($allEpisodes, 1) -ge 0.10
            at_least_two_reproducible_trials = $reproducibleTrials -ge 2
            at_least_three_distinct_completion_seeds = $distinctSuccesses.Count -ge 3
            every_trial_completed = @($summaries | Where-Object { $_.completions -le 0 }).Count -eq 0
            controller_and_losses_valid = @($summaries | Where-Object { $_.worker_restarts -ne 0 -or -not $_.finite_losses }).Count -eq 0
        }
        selected_trial = if ($passed) { $selected.trial } else { $null }
        decision = if ($passed) { "expand_trace_tail" } else { "retain_tail_boundary" }
    }
    $comparisonTemporary = Join-Path $root ".comparison.json.tmp"
    $comparison | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $comparisonTemporary -Encoding utf8
    Move-Item -LiteralPath $comparisonTemporary -Destination (Join-Path $root "comparison.json") -Force
    Write-PilotStatus "complete"
} catch {
    Write-PilotStatus "failed" "" $_.Exception.Message
    throw
}
