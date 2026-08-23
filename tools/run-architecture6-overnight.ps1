$ErrorActionPreference = "Stop"

$a6Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$a6Python = Join-Path $a6Repo ".venv\Scripts\python.exe"
$a6Game = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$a6Mod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$a6Source = Join-Path $a6Repo "runs\reward-v2-250k\final.pt"
$a6Reward = Join-Path $a6Repo "configs\reward-v2.json"
$a6Root = Join-Path $a6Repo "runs\architecture6-v2-overnight"
$a6TrainingRoot = Join-Path $a6Root "training"
$a6EvaluationRoot = Join-Path $a6Root "evaluation"
$a6Seeds = @(33001, 33002, 33003)
$a6EvaluationSeeds = (43001..43030) -join ","

foreach ($a6Required in @($a6Python, $a6Source, $a6Reward)) {
    if (-not (Test-Path -LiteralPath $a6Required -PathType Leaf)) {
        throw "Required file is missing: $a6Required"
    }
}
$a6CudaAvailable = & $a6Python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $a6CudaAvailable.Trim() -ne "True") {
    throw "The architecture-6 overnight pipeline requires CUDA-enabled PyTorch"
}

New-Item -ItemType Directory -Path $a6TrainingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $a6EvaluationRoot -Force | Out-Null
Set-Location -LiteralPath $a6Repo
$env:PYTHONPATH = Join-Path $a6Repo "src"

foreach ($a6Seed in $a6Seeds) {
    $a6Run = Join-Path $a6TrainingRoot "seed-$a6Seed"
    $a6Final = Join-Path $a6Run "final.pt"
    if (Test-Path -LiteralPath $a6Final -PathType Leaf) {
        continue
    }
    New-Item -ItemType Directory -Path $a6Run -Force | Out-Null
    $a6Log = Join-Path $a6Run "console.log"
    $a6Arguments = @(
        "-m", "autodancer.training.train",
        "--game-dir", $a6Game,
        "--mod-dir", $a6Mod,
        "--num-instances", "8",
        "--total-steps", "51200",
        "--run-dir", $a6Run,
        "--device", "cuda",
        "--seed", "$a6Seed",
        "--reward-config", $a6Reward,
        "--evaluation-interval", "0",
        "--dashboard", "8765"
    )
    $a6Latest = Join-Path $a6Run "latest.pt"
    if (Test-Path -LiteralPath $a6Latest -PathType Leaf) {
        $a6Arguments += @("--resume", $a6Latest)
    } else {
        $a6Arguments += @("--initialize-from", $a6Source)
    }
    Add-Content -LiteralPath $a6Log -Value (
        "`n=== launch architecture 6 seed $a6Seed at $((Get-Date).ToString('o')) ==="
    )
    & $a6Python @a6Arguments *>> $a6Log
    if ($LASTEXITCODE -ne 0) {
        throw "Architecture-6 training seed $a6Seed failed; see $a6Log"
    }
}

$a6Checkpoints = [ordered]@{ "v2-final" = $a6Source }
foreach ($a6Seed in $a6Seeds) {
    $a6Checkpoints["arch6-seed-$a6Seed"] = Join-Path $a6TrainingRoot "seed-$a6Seed\final.pt"
}
foreach ($a6Checkpoint in $a6Checkpoints.GetEnumerator()) {
    $a6Output = Join-Path $a6EvaluationRoot "$($a6Checkpoint.Key).json"
    if (Test-Path -LiteralPath $a6Output -PathType Leaf) {
        continue
    }
    $a6Log = Join-Path $a6EvaluationRoot "$($a6Checkpoint.Key).log"
    & $a6Python -m autodancer.training.baseline `
        --game-dir $a6Game `
        --mod-dir $a6Mod `
        --checkpoint $a6Checkpoint.Value `
        --output $a6Output `
        --num-instances 8 `
        --seeds $a6EvaluationSeeds `
        --max-steps 3000 `
        --reward-config $a6Reward `
        --device cuda `
        --dashboard 8765 `
        --trained-only *>> $a6Log
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation $($a6Checkpoint.Key) failed; see $a6Log"
    }
}

& $a6Python -m autodancer.training.architecture_compare --root $a6Root
if ($LASTEXITCODE -ne 0) {
    throw "Architecture-6 comparison failed"
}
$a6Comparison = Get-Content -LiteralPath (Join-Path $a6Root "comparison.json") -Raw |
    ConvertFrom-Json

if ($null -ne $a6Comparison.selected_seed) {
    $a6SelectedSeed = [int]$a6Comparison.selected_seed
    $a6Selected = Join-Path $a6TrainingRoot "seed-$a6SelectedSeed\final.pt"
    $a6LongRun = Join-Path $a6Root "selected-250k"
    $a6LongFinal = Join-Path $a6LongRun "final.pt"
    if (-not (Test-Path -LiteralPath $a6LongFinal -PathType Leaf)) {
        New-Item -ItemType Directory -Path $a6LongRun -Force | Out-Null
        $a6LongLog = Join-Path $a6LongRun "console.log"
        $a6LongResume = Join-Path $a6LongRun "latest.pt"
        if (-not (Test-Path -LiteralPath $a6LongResume -PathType Leaf)) {
            $a6LongResume = $a6Selected
        }
        & $a6Python -m autodancer.training.train `
            --game-dir $a6Game `
            --mod-dir $a6Mod `
            --num-instances 8 `
            --total-steps 250880 `
            --run-dir $a6LongRun `
            --device cuda `
            --seed 34001 `
            --reward-config $a6Reward `
            --evaluation-interval 0 `
            --dashboard 8765 `
            --resume $a6LongResume *>> $a6LongLog
        if ($LASTEXITCODE -ne 0) {
            throw "Selected architecture-6 continuation failed; see $a6LongLog"
        }
    }
}

@{
    completed_at = (Get-Date).ToString("o")
    decision = $a6Comparison.decision
    selected_seed = $a6Comparison.selected_seed
    training_seeds = $a6Seeds
    evaluation_seeds = 43001..43030
    long_run_completed = Test-Path -LiteralPath (Join-Path $a6Root "selected-250k\final.pt")
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $a6Root "pipeline-complete.json")
