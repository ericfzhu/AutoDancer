$ErrorActionPreference = "Stop"

$a8Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$a8Python = Join-Path $a8Repo ".venv\Scripts\python.exe"
$a8Game = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$a8Mod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$a8Source = Join-Path $a8Repo "runs\reward-v2-250k\final.pt"
$a8Reward = Join-Path $a8Repo "configs\reward-v2.json"
$a8Root = Join-Path $a8Repo "runs\architecture8-controls"
$a8Training = Join-Path $a8Root "training"
$a8Curve = Join-Path $a8Root "curve-evaluation"
$a8Broad = Join-Path $a8Root "broad-evaluation"
$a8TrainingSeed = 36001
$a8CurveSeeds = (45001..45016) -join ","
$a8BroadSeeds = (46001..46030) -join ","
$a8FinalSteps = 30720
$a8WarmupSteps = 10240

foreach ($a8Required in @($a8Python, $a8Source, $a8Reward)) {
    if (-not (Test-Path -LiteralPath $a8Required -PathType Leaf)) {
        throw "Required file is missing: $a8Required"
    }
}
foreach ($a8RequiredDirectory in @($a8Game, $a8Mod)) {
    if (-not (Test-Path -LiteralPath $a8RequiredDirectory -PathType Container)) {
        throw "Required directory is missing: $a8RequiredDirectory"
    }
}
$a8CudaAvailable = & $a8Python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $a8CudaAvailable.Trim() -ne "True") {
    throw "The Architecture-8 controls require CUDA-enabled PyTorch"
}

New-Item -ItemType Directory -Path $a8Training, $a8Curve, $a8Broad -Force | Out-Null
Set-Location -LiteralPath $a8Repo
$env:PYTHONPATH = Join-Path $a8Repo "src"

function Invoke-A8Training {
    param(
        [string]$Arm,
        [int]$Architecture,
        [string]$ActionContract,
        [int]$TotalSteps,
        [int]$FreezeBaseUpdates = 0
    )
    $a8Run = Join-Path $a8Training $Arm
    $a8Final = Join-Path $a8Run "final.pt"
    if ((Test-Path -LiteralPath $a8Final -PathType Leaf)) {
        $a8PayloadStep = & $a8Python -c (
            "import torch; print(torch.load(r'{0}', map_location='cpu', weights_only=False).get('global_step', 0))" -f $a8Final
        )
        if ($LASTEXITCODE -eq 0 -and [int]$a8PayloadStep -ge $TotalSteps) {
            return
        }
    }
    New-Item -ItemType Directory -Path $a8Run -Force | Out-Null
    $a8Log = Join-Path $a8Run "console.log"
    $a8Arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $a8Game,
        "--mod-dir", $a8Mod,
        "--num-instances", "8",
        "--total-steps", "$TotalSteps",
        "--run-dir", $a8Run,
        "--architecture", "$Architecture",
        "--device", "cuda",
        "--seed", "$a8TrainingSeed",
        "--reward-config", $a8Reward,
        "--action-contract", $ActionContract,
        "--freeze-base-updates", "$FreezeBaseUpdates",
        "--checkpoint-interval", "10240",
        "--evaluation-interval", "0",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765"
    )
    $a8Latest = Join-Path $a8Run "latest.pt"
    if (Test-Path -LiteralPath $a8Latest -PathType Leaf) {
        $a8Arguments += @("--resume", $a8Latest)
    } else {
        $a8Arguments += @("--fine-tune-from", $a8Source)
    }
    Add-Content -LiteralPath $a8Log -Value (
        "`n=== launch $Arm to $TotalSteps at $((Get-Date).ToString('o')) ==="
    )
    & $a8Python @a8Arguments *>> $a8Log
    if ($LASTEXITCODE -ne 0) {
        throw "Training $Arm failed; see $a8Log"
    }
}

function Invoke-A8Representation {
    param([string]$Name, [string]$Checkpoint)
    $a8Output = Join-Path $a8Root "representation-$Name.json"
    & $a8Python -m autodancer.training.representation `
        $Checkpoint `
        --output $a8Output `
        --require-material-new-groups
    return $LASTEXITCODE -eq 0
}

function Invoke-A8Evaluation {
    param(
        [string]$Name,
        [string]$Checkpoint,
        [string]$ActionContract,
        [string]$Seeds,
        [int]$MaxSteps,
        [string]$Directory
    )
    $a8Output = Join-Path $Directory "$Name.json"
    if (Test-Path -LiteralPath $a8Output -PathType Leaf) {
        return
    }
    $a8Log = Join-Path $Directory "$Name.log"
    & $a8Python -m autodancer.training.baseline `
        --game-dir $a8Game `
        --mod-dir $a8Mod `
        --checkpoint $Checkpoint `
        --output $a8Output `
        --num-instances 8 `
        --seeds $Seeds `
        --max-steps $MaxSteps `
        --reward-config $a8Reward `
        --action-contract $ActionContract `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only *>> $a8Log
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation $Name failed; see $a8Log"
    }
}

$a8Parity = Join-Path $a8Root "parity.json"
& $a8Python -m autodancer.training.architecture8_parity `
    --checkpoint $a8Source `
    --output $a8Parity
if ($LASTEXITCODE -ne 0) {
    throw "Architecture-8 parity/gradient preflight failed"
}

Invoke-A8Training "a2-legacy" 2 "legacy-no-wait" $a8FinalSteps
Invoke-A8Training "a2-fixed" 2 "current" $a8FinalSteps
Invoke-A8Training "a8" 8 "current" $a8WarmupSteps 10

$a8WarmupCheckpoint = Join-Path $a8Training "a8\checkpoint-00010240.pt"
if (-not (Invoke-A8Representation "warmup" $a8WarmupCheckpoint)) {
    & $a8Python -m autodancer.training.architecture8_compare --root $a8Root
    @{ completed_at = (Get-Date).ToString("o"); decision = "a8_failed_warmup_representation_gate" } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a8Root "pipeline-complete.json")
    exit 0
}

Invoke-A8Training "a8" 8 "current" $a8FinalSteps 10
$a8FinalCheckpoint = Join-Path $a8Training "a8\final.pt"
if (-not (Invoke-A8Representation "final" $a8FinalCheckpoint)) {
    & $a8Python -m autodancer.training.architecture8_compare --root $a8Root
    @{ completed_at = (Get-Date).ToString("o"); decision = "a8_failed_final_representation_gate" } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a8Root "pipeline-complete.json")
    exit 0
}

foreach ($a8Arm in @("a2-legacy", "a2-fixed", "a8")) {
    $a8Contract = if ($a8Arm -eq "a2-legacy") { "legacy-no-wait" } else { "current" }
    foreach ($a8Step in @(0, 10240, 20480, 30720)) {
        if ($a8Step -eq 0) {
            if ($a8Arm -eq "a8") { continue }
            $a8Checkpoint = $a8Source
        } else {
            $a8Checkpoint = Join-Path $a8Training (
                "$a8Arm\checkpoint-{0:D8}.pt" -f $a8Step
            )
        }
        Invoke-A8Evaluation "$a8Arm-step-$('{0:D8}' -f $a8Step)" `
            $a8Checkpoint $a8Contract $a8CurveSeeds 1500 $a8Curve
    }
}

& $a8Python -m autodancer.training.architecture8_compare --root $a8Root
$a8Comparison = Get-Content -LiteralPath (Join-Path $a8Root "comparison.json") -Raw |
    ConvertFrom-Json
if (-not $a8Comparison.curve_gate_passed) {
    @{ completed_at = (Get-Date).ToString("o"); decision = $a8Comparison.decision } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a8Root "pipeline-complete.json")
    exit 0
}

foreach ($a8Arm in @("a2-legacy", "a2-fixed", "a8")) {
    $a8Contract = if ($a8Arm -eq "a2-legacy") { "legacy-no-wait" } else { "current" }
    $a8Checkpoint = Join-Path $a8Training "$a8Arm\final.pt"
    Invoke-A8Evaluation $a8Arm $a8Checkpoint $a8Contract $a8BroadSeeds 3000 $a8Broad
}

& $a8Python -m autodancer.training.architecture8_compare --root $a8Root
$a8Comparison = Get-Content -LiteralPath (Join-Path $a8Root "comparison.json") -Raw |
    ConvertFrom-Json
@{
    completed_at = (Get-Date).ToString("o")
    decision = $a8Comparison.decision
    training_seed = $a8TrainingSeed
    curve_evaluation_seeds = 45001..45016
    broad_evaluation_seeds = 46001..46030
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a8Root "pipeline-complete.json")
