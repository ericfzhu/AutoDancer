param(
    [string]$GameDir = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64",
    [string]$RunDir = "runs/performance-project/prefix-warmup",
    [string]$Checkpoint = "runs/retained-death-metal-trace-window-5/training/seed-103001/final.pt"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    # Validate live state parity first, then time one online and one batched update.
    if (Test-Path -LiteralPath $RunDir) { throw "Performance output already exists: $RunDir" }
    foreach ($mode in @("verify", "online", "batched")) {
        $trialDir = Join-Path $RunDir $mode
        $warmupBatch = if ($mode -eq "online") { 0 } else { 16 }
        $extraArgs = @()
        if ($mode -eq "verify") { $extraArgs += "--trace-prefix-warmup-verify" }
        & .venv/Scripts/python.exe -m autodancer.training.train `
            --game-dir $GameDir --mod-dir mods/AutoDancer --num-instances 8 `
            --total-steps 1024 --run-dir $trialDir --device cuda --architecture 8 `
            --seed 103001 --fine-tune-from $Checkpoint `
            --sequence-encoding-batch-size 16 --inference-transfer-mode legacy `
            --trace-prefix-warmup-batch-size $warmupBatch @extraArgs `
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
        if ($LASTEXITCODE -ne 0) { throw "Warm-up trial $mode failed" }
        if ($mode -eq "verify") {
            $metrics = Get-Content -LiteralPath (Join-Path $trialDir "metrics.jsonl") | Select-Object -Last 1 | ConvertFrom-Json
            if ($metrics.inference_scheduler.warmup_verified -le 0) {
                throw "No live warm-up handoffs were verified; timing trials aborted"
            }
        }
    }
} finally {
    Pop-Location
}
