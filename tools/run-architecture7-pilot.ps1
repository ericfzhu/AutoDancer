$ErrorActionPreference = "Stop"

$a7Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$a7Python = Join-Path $a7Repo ".venv\Scripts\python.exe"
$a7Game = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$a7Mod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$a7Source = Join-Path $a7Repo "runs\reward-v2-250k\final.pt"
$a7Reward = Join-Path $a7Repo "configs\reward-v2.json"
$a7Root = Join-Path $a7Repo "runs\architecture7-v2-pilot"
$a7TrainingRoot = Join-Path $a7Root "training"
$a7EvaluationRoot = Join-Path $a7Root "evaluation"
$a7Seeds = @(35001, 35002, 35003)
$a7EvaluationSeeds = (44001..44030) -join ","

foreach ($a7Required in @($a7Python, $a7Source, $a7Reward)) {
    if (-not (Test-Path -LiteralPath $a7Required -PathType Leaf)) {
        throw "Required file is missing: $a7Required"
    }
}
foreach ($a7RequiredDirectory in @($a7Game, $a7Mod)) {
    if (-not (Test-Path -LiteralPath $a7RequiredDirectory -PathType Container)) {
        throw "Required directory is missing: $a7RequiredDirectory"
    }
}
$a7CudaAvailable = & $a7Python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $a7CudaAvailable.Trim() -ne "True") {
    throw "The Architecture-7 pilot requires CUDA-enabled PyTorch"
}

New-Item -ItemType Directory -Path $a7TrainingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $a7EvaluationRoot -Force | Out-Null
Set-Location -LiteralPath $a7Repo
$env:PYTHONPATH = Join-Path $a7Repo "src"

$a7Parity = Join-Path $a7Root "parity.json"
& $a7Python -m autodancer.training.architecture7_parity `
    --checkpoint $a7Source `
    --output $a7Parity
if ($LASTEXITCODE -ne 0) {
    throw "Architecture-7 parity preflight failed; no live training was started"
}

foreach ($a7Seed in $a7Seeds) {
    $a7Run = Join-Path $a7TrainingRoot "seed-$a7Seed"
    $a7Final = Join-Path $a7Run "final.pt"
    if (Test-Path -LiteralPath $a7Final -PathType Leaf) {
        continue
    }
    New-Item -ItemType Directory -Path $a7Run -Force | Out-Null
    $a7Log = Join-Path $a7Run "console.log"
    $a7Arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $a7Game,
        "--mod-dir", $a7Mod,
        "--num-instances", "8",
        "--total-steps", "51200",
        "--run-dir", $a7Run,
        "--architecture", "7",
        "--device", "cuda",
        "--seed", "$a7Seed",
        "--reward-config", $a7Reward,
        "--evaluation-interval", "0",
        "--startup-timeout", "60",
        "--turn-timeout", "30",
        "--reset-timeout", "60",
        "--affinity", "none",
        "--dashboard", "8765"
    )
    $a7Latest = Join-Path $a7Run "latest.pt"
    if (Test-Path -LiteralPath $a7Latest -PathType Leaf) {
        $a7Arguments += @("--resume", $a7Latest)
    } else {
        $a7Arguments += @("--initialize-from", $a7Source)
    }
    Add-Content -LiteralPath $a7Log -Value (
        "`n=== launch Architecture 7 seed $a7Seed at $((Get-Date).ToString('o')) ==="
    )
    & $a7Python @a7Arguments *>> $a7Log
    if ($LASTEXITCODE -ne 0) {
        throw "Architecture-7 training seed $a7Seed failed; see $a7Log"
    }
}

$a7Checkpoints = [ordered]@{ "v2-final" = $a7Source }
foreach ($a7Seed in $a7Seeds) {
    $a7Checkpoints["arch7-seed-$a7Seed"] = Join-Path $a7TrainingRoot "seed-$a7Seed\final.pt"
}
foreach ($a7Checkpoint in $a7Checkpoints.GetEnumerator()) {
    $a7Output = Join-Path $a7EvaluationRoot "$($a7Checkpoint.Key).json"
    if (Test-Path -LiteralPath $a7Output -PathType Leaf) {
        continue
    }
    $a7Log = Join-Path $a7EvaluationRoot "$($a7Checkpoint.Key).log"
    & $a7Python -m autodancer.training.baseline `
        --game-dir $a7Game `
        --mod-dir $a7Mod `
        --checkpoint $a7Checkpoint.Value `
        --output $a7Output `
        --num-instances 8 `
        --seeds $a7EvaluationSeeds `
        --max-steps 3000 `
        --reward-config $a7Reward `
        --device cuda `
        --startup-timeout 60 `
        --turn-timeout 30 `
        --reset-timeout 60 `
        --affinity none `
        --dashboard 8765 `
        --trained-only *>> $a7Log
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation $($a7Checkpoint.Key) failed; see $a7Log"
    }
}

& $a7Python -m autodancer.training.architecture7_compare --root $a7Root
if ($LASTEXITCODE -ne 0) {
    throw "Architecture-7 comparison failed"
}
$a7Comparison = Get-Content -LiteralPath (Join-Path $a7Root "comparison.json") -Raw |
    ConvertFrom-Json
@{
    completed_at = (Get-Date).ToString("o")
    decision = $a7Comparison.decision
    training_seeds = $a7Seeds
    evaluation_seeds = 44001..44030
    parity_passed = $true
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a7Root "pipeline-complete.json")
