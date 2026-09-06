param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$RunDir = "runs/performance-project/live",
    [string]$Checkpoint = "runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    # Exactly two single-rollout measurements. Never overwrite prior evidence.
    if (Test-Path -LiteralPath $RunDir) { throw "Performance output already exists: $RunDir" }
    foreach ($encoderBatch in @(0, 16)) {
        $trialDir = Join-Path $RunDir "encoder-$encoderBatch"
        & .venv/Scripts/python.exe -m autodancer.training.train `
            --game-dir $GameDir --mod-dir mods/AutoDancer --num-instances 8 `
            --total-steps 1024 --run-dir $trialDir --device cuda --architecture 8 `
            --seed 103001 --fine-tune-from $Checkpoint `
            --sequence-encoding-batch-size $encoderBatch `
            --checkpoint-interval 1000000 --evaluation-interval 0 --max-turns 500 `
            --action-contract map-navigation-prior-v1 `
            --training-seed-pool 92008,92096,92116 `
            --curriculum-start-level 4 --curriculum-target-level 5 --curriculum-profile player20 `
            --reward-config configs/reward-death-metal-potential-v5.json `
            --policy-feedback-reward-config configs/reward-death-metal-potential-v5.json `
            --reward-lineage-version DeathMetalPotentialV5 `
            --trace-prefix-bank runs/qualified-death-metal-trace-search/demonstration-bank.json `
            --trace-prefix-qualification runs/qualified-death-metal-trace-search/qualification.json `
            --trace-prefix-tail-actions 60 --trace-prefix-tail-window 32,36,40,44,48,52,56,60 `
            --trace-prefix-recurrent-state warm --affinity none --steam-presence-worker 0
        if ($LASTEXITCODE -ne 0) { throw "Performance trial $encoderBatch failed" }
    }
} finally {
    Pop-Location
}
