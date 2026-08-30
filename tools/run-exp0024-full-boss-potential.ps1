param([int]$QualificationPid = 0)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run-exp0020-legal-death-metal-guide.ps1") `
    -QualificationPid $QualificationPid `
    -ExperimentId "EXP-0024"
exit $LASTEXITCODE
