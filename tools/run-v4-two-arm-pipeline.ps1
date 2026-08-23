$ErrorActionPreference = "Stop"

$v4Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$v4Python = Join-Path $v4Repo ".venv\Scripts\python.exe"
$v4Game = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64"
$v4Mod = Join-Path $env:LOCALAPPDATA "NecroDancer\mods\AutoDancer"
$v4Source = Join-Path $v4Repo "runs\reward-v2-250k\final.pt"
$v4Root = Join-Path $v4Repo "runs\reward-v4-two-arm"
$v4TrainingRoot = Join-Path $v4Root "training"
$v4EvaluationRoot = Join-Path $v4Root "evaluation"
$v4Seeds = @(32001, 32002, 32003)
$v4EvaluationSeeds = (42001..42030) -join ","
$v4Arms = [ordered]@{
    "v4a" = Join-Path $v4Repo "configs\reward-v4a.json"
    "v4b" = Join-Path $v4Repo "configs\reward-v4b.json"
}

if (-not (Test-Path -LiteralPath $v4Python -PathType Leaf)) {
    throw "Project Python is missing at $v4Python"
}
$v4CudaAvailable = & $v4Python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $v4CudaAvailable.Trim() -ne "True") {
    throw "The V4 pipeline requires CUDA-enabled PyTorch in the project environment"
}
if (-not (Test-Path -LiteralPath $v4Source -PathType Leaf)) {
    throw "V2 initialization checkpoint is missing at $v4Source"
}
foreach ($v4Config in $v4Arms.Values) {
    if (-not (Test-Path -LiteralPath $v4Config -PathType Leaf)) {
        throw "Reward configuration is missing at $v4Config"
    }
}

New-Item -ItemType Directory -Path $v4TrainingRoot -Force | Out-Null
New-Item -ItemType Directory -Path $v4EvaluationRoot -Force | Out-Null
Set-Location -LiteralPath $v4Repo
$env:PYTHONPATH = Join-Path $v4Repo "src"

foreach ($v4Arm in $v4Arms.GetEnumerator()) {
    foreach ($v4Seed in $v4Seeds) {
        $v4Run = Join-Path $v4TrainingRoot ("{0}\seed-{1}" -f $v4Arm.Key, $v4Seed)
        $v4Final = Join-Path $v4Run "final.pt"
        if (Test-Path -LiteralPath $v4Final -PathType Leaf) {
            continue
        }
        New-Item -ItemType Directory -Path $v4Run -Force | Out-Null
        $v4Log = Join-Path $v4Run "console.log"
        $v4Arguments = @(
            "-m", "autodancer.training.train",
            "--game-dir", $v4Game,
            "--mod-dir", $v4Mod,
            "--num-instances", "8",
            "--total-steps", "51200",
            "--run-dir", $v4Run,
            "--device", "auto",
            "--seed", "$v4Seed",
            "--reward-config", $v4Arm.Value,
            "--evaluation-interval", "0",
            "--dashboard", "8765"
        )
        $v4Latest = Join-Path $v4Run "latest.pt"
        if (Test-Path -LiteralPath $v4Latest -PathType Leaf) {
            $v4Arguments += @("--resume", $v4Latest)
        } else {
            $v4Arguments += @("--initialize-from", $v4Source)
        }
        Add-Content -LiteralPath $v4Log -Value ("`n=== launch {0} seed {1} at {2:o} ===" -f $v4Arm.Key, $v4Seed, (Get-Date))
        & $v4Python @v4Arguments *>> $v4Log
        if ($LASTEXITCODE -ne 0) {
            throw "Training $($v4Arm.Key) seed $v4Seed failed with exit code $LASTEXITCODE; see $v4Log"
        }
    }
}

$v4Checkpoints = [ordered]@{ "v2-final" = $v4Source }
foreach ($v4Arm in $v4Arms.GetEnumerator()) {
    foreach ($v4Seed in $v4Seeds) {
        $v4Checkpoints["$($v4Arm.Key)-seed-$v4Seed"] = Join-Path $v4TrainingRoot ("{0}\seed-{1}\final.pt" -f $v4Arm.Key, $v4Seed)
    }
}

foreach ($v4Checkpoint in $v4Checkpoints.GetEnumerator()) {
    $v4Output = Join-Path $v4EvaluationRoot ("{0}.json" -f $v4Checkpoint.Key)
    if (Test-Path -LiteralPath $v4Output -PathType Leaf) {
        continue
    }
    $v4Log = Join-Path $v4EvaluationRoot ("{0}.log" -f $v4Checkpoint.Key)
    & $v4Python -m autodancer.training.baseline `
        --game-dir $v4Game `
        --mod-dir $v4Mod `
        --checkpoint $v4Checkpoint.Value `
        --output $v4Output `
        --num-instances 8 `
        --seeds $v4EvaluationSeeds `
        --max-steps 3000 `
        --reward-config $v4Arms["v4a"] `
        --device auto `
        --dashboard 8765 `
        --trained-only *>> $v4Log
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation $($v4Checkpoint.Key) failed with exit code $LASTEXITCODE; see $v4Log"
    }
}

& $v4Python -m autodancer.training.reward_compare --root $v4Root
if ($LASTEXITCODE -ne 0) {
    throw "V4 comparison failed with exit code $LASTEXITCODE"
}

$v4Comparison = Get-Content -LiteralPath (Join-Path $v4Root "comparison.json") -Raw | ConvertFrom-Json
@{
    completed_at = (Get-Date).ToString("o")
    training_seeds = $v4Seeds
    evaluation_seeds = 42001..42030
    selected_arm = $v4Comparison.selected_arm
    decision = $v4Comparison.decision
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $v4Root "pilot-complete.json")

if ($null -ne $v4Comparison.selected_arm) {
    $v4SelectedArm = [string]$v4Comparison.selected_arm
    $v4SelectedConfig = $v4Arms[$v4SelectedArm]
    $v4LongRun = Join-Path $v4Root "selected-250k"
    $v4LongFinal = Join-Path $v4LongRun "final.pt"
    if (-not (Test-Path -LiteralPath $v4LongFinal -PathType Leaf)) {
        New-Item -ItemType Directory -Path $v4LongRun -Force | Out-Null
        $v4LongArguments = @(
            "-m", "autodancer.training.train",
            "--game-dir", $v4Game,
            "--mod-dir", $v4Mod,
            "--num-instances", "8",
            "--total-steps", "250880",
            "--run-dir", $v4LongRun,
            "--device", "auto",
            "--seed", "33001",
            "--reward-config", $v4SelectedConfig,
            "--evaluation-interval", "0",
            "--dashboard", "8765"
        )
        $v4LongLatest = Join-Path $v4LongRun "latest.pt"
        if (Test-Path -LiteralPath $v4LongLatest -PathType Leaf) {
            $v4LongArguments += @("--resume", $v4LongLatest)
        } else {
            $v4LongArguments += @("--initialize-from", $v4Source)
        }
        & $v4Python @v4LongArguments *>> (Join-Path $v4LongRun "console.log")
        if ($LASTEXITCODE -ne 0) {
            throw "Selected-arm 250k run failed with exit code $LASTEXITCODE"
        }
    }

    $v4FinalEvaluation = Join-Path $v4Root "final-evaluation"
    New-Item -ItemType Directory -Path $v4FinalEvaluation -Force | Out-Null
    $v4FinalSeeds = (43001..43064) -join ","
    $v4FinalCheckpoints = [ordered]@{
        "v2-final" = $v4Source
        "$v4SelectedArm-final" = $v4LongFinal
    }
    foreach ($v4Checkpoint in $v4FinalCheckpoints.GetEnumerator()) {
        $v4Output = Join-Path $v4FinalEvaluation ("{0}.json" -f $v4Checkpoint.Key)
        if (Test-Path -LiteralPath $v4Output -PathType Leaf) {
            continue
        }
        & $v4Python -m autodancer.training.baseline `
            --game-dir $v4Game `
            --mod-dir $v4Mod `
            --checkpoint $v4Checkpoint.Value `
            --output $v4Output `
            --num-instances 8 `
            --seeds $v4FinalSeeds `
            --max-steps 5000 `
            --reward-config $v4Arms["v4a"] `
            --device auto `
            --dashboard 8765 `
            --trained-only *>> (Join-Path $v4FinalEvaluation ("{0}.log" -f $v4Checkpoint.Key))
        if ($LASTEXITCODE -ne 0) {
            throw "Final evaluation $($v4Checkpoint.Key) failed with exit code $LASTEXITCODE"
        }
    }
    & $v4Python -m autodancer.training.reward_final_compare `
        --reference (Join-Path $v4FinalEvaluation "v2-final.json") `
        --candidate (Join-Path $v4FinalEvaluation "$v4SelectedArm-final.json") `
        --candidate-name $v4SelectedArm `
        --output (Join-Path $v4FinalEvaluation "comparison.json")
    if ($LASTEXITCODE -ne 0) {
        throw "Final V4 comparison failed with exit code $LASTEXITCODE"
    }
}

$v4FinalComparisonPath = Join-Path $v4Root "final-evaluation\comparison.json"
$v4FinalDecision = if (Test-Path -LiteralPath $v4FinalComparisonPath -PathType Leaf) {
    (Get-Content -LiteralPath $v4FinalComparisonPath -Raw | ConvertFrom-Json).decision
} else {
    $v4Comparison.decision
}
@{
    completed_at = (Get-Date).ToString("o")
    pilot_decision = $v4Comparison.decision
    selected_arm = $v4Comparison.selected_arm
    final_decision = $v4FinalDecision
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $v4Root "pipeline-complete.json")
