param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$ExperimentRoot = "runs\legal-death-metal-full-boss-potential",
    [string]$RunDir = "runs\exp0024-exact-success-recovery",
    [string]$TraceRoot = "runs\qualified-death-metal-trace-search",
    [int]$TargetSteps = 108544
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
function Resolve-RepositoryPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$experiment = Resolve-RepositoryPath $ExperimentRoot
$run = Resolve-RepositoryPath $RunDir
$trace = Resolve-RepositoryPath $TraceRoot
$checkpoint = Join-Path $experiment "training\seed-94001\checkpoint-00092160.pt"
$seedSelection = Join-Path $experiment "training\seed-selection.json"
$excludedSelection = Join-Path $experiment "evaluation\heldout-selection.json"
$reward = Join-Path $repoRoot "configs\reward-death-metal-potential-v5.json"
$mod = Join-Path $repoRoot "mods\AutoDancer"
$episodes = Join-Path $run "episodes.jsonl"
$priorSearchReport = Join-Path $trace (
    "candidates\seed-94001\checkpoint-00092160\reports\stochastic-96003\report.json"
)
$bank = Join-Path $trace "demonstration-bank.json"
$qualification = Join-Path $trace "qualification.json"
$recurrent = Join-Path $trace "recurrent-demonstrations.npz"
$status = Join-Path $trace "status.json"
$recoveryStatus = Join-Path $run "status.json"

if ($TargetSteps -lt 108544) { throw "TargetSteps must include the three known update-97/102/106 clears" }
foreach ($required in @($python, $checkpoint, $seedSelection, $excludedSelection, $reward)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required exact-continuation input is missing: $required"
    }
}
[System.IO.Directory]::CreateDirectory($run) | Out-Null
[System.IO.Directory]::CreateDirectory($trace) | Out-Null

function Write-RecoveryStatus([string]$State, [string]$ErrorText = "") {
    $payload = [ordered]@{
        schema_version = 1
        status = $State
        checkpoint = $checkpoint
        target_steps = $TargetSteps
        error = $ErrorText
        updated_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    $temporary = "$recoveryStatus.tmp"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $recoveryStatus -Force
}

try {
    $selection = Get-Content -LiteralPath $seedSelection -Raw | ConvertFrom-Json
    $excluded = Get-Content -LiteralPath $excludedSelection -Raw | ConvertFrom-Json
    $trainingSeeds = @($selection.seeds | ForEach-Object { [int]$_ })
    $heldout = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($seed in @($excluded.seeds)) { [void]$heldout.Add([int]$seed) }
    if ($trainingSeeds.Count -ne 48 -or @($trainingSeeds | Where-Object { $heldout.Contains($_) }).Count) {
        throw "EXP-0024 training and held-out seed banks are invalid or overlap"
    }
    $seedCsv = $trainingSeeds -join ","

    $final = Join-Path $run "final.pt"
    if (-not (Test-Path -LiteralPath $final -PathType Leaf)) {
        Write-RecoveryStatus "replaying-exact-continuation"
        & $python -m autodancer.training.train `
            --game-dir $GameDir `
            --mod-dir $mod `
            --num-instances 8 `
            --total-steps $TargetSteps `
            --run-dir $run `
            --resume $checkpoint `
            --device cuda `
            --architecture 8 `
            --seed 94001 `
            --checkpoint-interval 16384 `
            --evaluation-interval 0 `
            --max-turns 500 `
            --action-contract map-navigation-prior-v1 `
            --training-seed-pool $seedCsv `
            --curriculum-start-level 4 `
            --curriculum-target-level 5 `
            --curriculum-profile player20 `
            --reward-config $reward `
            --freeze-base-updates 10 `
            --affinity none `
            --dashboard 8765
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $final -PathType Leaf)) {
            throw "Exact EXP-0024 continuation failed"
        }
    }
    if (-not (Test-Path -LiteralPath $episodes -PathType Leaf)) {
        throw "Exact continuation produced no episode-action journal"
    }
    $successful = @(
        Get-Content -LiteralPath $episodes |
            ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object {
                [string]$_.status -eq "curriculum_complete" -and
                [bool]$_.infrastructure_valid -and
                @($_.successful_action_sequence).Count -gt 0
            }
    )
    $priorSuccessful = @()
    if (Test-Path -LiteralPath $priorSearchReport -PathType Leaf) {
        $prior = Get-Content -LiteralPath $priorSearchReport -Raw | ConvertFrom-Json
        $sameCheckpoint = [System.IO.Path]::GetFullPath([string]$prior.checkpoint) -eq $checkpoint
        if (-not [bool]$prior.controller_valid -or -not $sameCheckpoint) {
            throw "Preserved update-90 search report has invalid controller or checkpoint identity"
        }
        $priorSuccessful = @(
            $prior.trained.results |
                Where-Object {
                    [string]$_.status -eq "curriculum_complete" -and
                    @($_.successful_action_sequence).Count -gt 0
                }
        )
    }
    $successfulSeeds = @(
        @($successful) + @($priorSuccessful) |
            ForEach-Object { [int]$_.seed } |
            Sort-Object -Unique
    )
    if ($successfulSeeds.Count -lt 3) {
        throw "Exact continuation recovered successes on only $($successfulSeeds.Count) distinct seeds"
    }

    Write-RecoveryStatus "building-bank"
    $buildArguments = @(
        "-m", "autodancer.training.demonstration_replay", "build",
        "--episodes", $episodes
    )
    if ($priorSuccessful.Count) { $buildArguments += @("--episodes", $priorSearchReport) }
    $buildArguments += @("--output", $bank)
    & $python @buildArguments
    if ($LASTEXITCODE -ne 0) { throw "Recovered demonstration-bank construction failed" }
    $bankPayload = Get-Content -LiteralPath $bank -Raw | ConvertFrom-Json

    Write-RecoveryStatus "qualifying-live-replay"
    & $python -m autodancer.training.demonstration_replay qualify `
        --game-dir $GameDir `
        --mod-dir $mod `
        --bank $bank `
        --output $qualification `
        --num-instances 8 `
        --action-contract map-navigation-prior-v1 `
        --policy-feedback-reward-config $reward `
        --recurrent-output $recurrent
    if ($LASTEXITCODE -ne 0) { throw "Recovered live trace qualification failed" }
    $qualified = Get-Content -LiteralPath $qualification -Raw | ConvertFrom-Json
    $qualifiedSeeds = @(
        $qualified.results |
            Where-Object { [bool]$_.valid } |
            ForEach-Object { [int]$_.seed } |
            Sort-Object -Unique
    )
    if (-not [bool]$qualified.valid -or $qualifiedSeeds.Count -lt 3) {
        throw "Recovered traces failed the three-distinct-seed fresh-replay gate"
    }

    $published = [ordered]@{
        schema_version = 1
        status = "complete"
        checkpoint = $checkpoint
        selection_reason = "exact-rng-schedule-continuation-action-log-recovery"
        training_seed_selection = $seedSelection
        excluded_seed_selection = $excludedSelection
        policy_seeds = @()
        reports = @($episodes)
        bank = $bank
        bank_sha256 = [string]$bankPayload.bank_sha256
        trace_count = @($bankPayload.traces).Count
        qualification = $qualification
        recurrent_demonstrations = $recurrent
        recurrent_demonstrations_manifest = "$recurrent.manifest.json"
        qualified_trace_count = [int]$qualified.qualified_trace_count
        qualified_distinct_seed_count = $qualifiedSeeds.Count
        qualified_distinct_seeds = $qualifiedSeeds
        minimum_qualified_distinct_seeds = 3
        valid = $true
    }
    $temporary = Join-Path $trace ".status.json.tmp"
    $published | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $status -Force
    Write-RecoveryStatus "complete"
    $published | ConvertTo-Json -Depth 8
} catch {
    Write-RecoveryStatus "failed" $_.Exception.Message
    throw
}
