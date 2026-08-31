param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$TraceSearchRoot = "runs\qualified-death-metal-trace-search",
    [string]$RunDir = "runs\qualified-death-metal-tail1",
    [int[]]$TrainingSeeds = @(97001, 97002, 97003),
    [int]$TotalSteps = 61440,
    [int]$RequestedTailActions = 1,
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

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$traceRoot = Resolve-RepositoryPath $TraceSearchRoot
$root = Resolve-RepositoryPath $RunDir
$searchStatusPath = Join-Path $traceRoot "status.json"
$pipelineStatusPath = Join-Path $root "pipeline-status.json"
$reward = Join-Path $repoRoot "configs\reward-death-metal-potential-v5.json"
$mod = Join-Path $repoRoot "mods\AutoDancer"
$qualification = Join-Path $repoRoot "runs\controller-qualification-player-health-only-world-ready-memory-controlled\qualification.json"
$trackingPath = (Join-Path $repoRoot ".runtime\mlflow\mlflow.db").Replace("\", "/")
$trackingUri = "sqlite:///$trackingPath"
$pilotStartedAt = (Get-Date).ToUniversalTime().ToString("o")
[System.IO.Directory]::CreateDirectory($root) | Out-Null

function Write-PilotStatus([string]$Status, [string]$Trial = "", [string]$ErrorText = "") {
    $payload = [ordered]@{
        schema_version = 1
        experiment_id = "EXP-0025"
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
    $tailActions = [math]::Min($RequestedTailActions, [int]$minimumTraceLength - 1)
    if ($tailActions -le 0) { throw "Qualified traces are too short to leave a legal replay prefix" }
    $gameSeeds = @($search.qualified_distinct_seeds | ForEach-Object { [int]$_ } | Sort-Object -Unique)
    $gameSeedCsv = $gameSeeds -join ","
    $evaluationModes = @(
        [pscustomobject]@{ name = "deterministic"; mode = "deterministic"; policy_seed = 0 },
        [pscustomobject]@{ name = "stochastic-98001"; mode = "stochastic"; policy_seed = 98001 },
        [pscustomobject]@{ name = "stochastic-98002"; mode = "stochastic"; policy_seed = 98002 }
    )

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
                "--initialize-from", $sourceCheckpoint, "--affinity", "none", "--experiment-id", "EXP-0025",
                "--experiment-arm", "a8-qualified-trace-tail", "--trial-id", $trial,
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
                    --experiment-id EXP-0025 `
                    --experiment-arm a8-qualified-trace-tail `
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
        experiment_id = "EXP-0025"
        source_checkpoint = $sourceCheckpoint
        bank = $bank
        qualification = $traceQualification
        tail_actions = $tailActions
        qualified_training_seeds = $gameSeeds
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
