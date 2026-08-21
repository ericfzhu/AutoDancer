$ErrorActionPreference = "Stop"

$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "Visual Studio 2022 C++ build tools were not found"
}

$source = Join-Path $PSScriptRoot "autodancer_native.c"
$outputDir = Join-Path $PSScriptRoot "build"
$output = Join-Path $outputDir "autodancer_native.dll"
$compilerSource = Join-Path $PSScriptRoot "compile_lua.c"
$compiler = Join-Path $outputDir "compile_lua.exe"
$shimSource = Join-Path $PSScriptRoot "autodancer_native.lua"
$shimOutput = Join-Path $outputDir "autodancer_native.luac"
$gameLua = "X:\Steam\steamapps\common\Crypt of the NecroDancer\NecroDancer64\lua51.dll"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$command = '"' + $vsDevCmd + '" -arch=x64 -host_arch=x64 >nul && cl /nologo /W4 /WX /O2 /LD "' + $source + '" /link /OUT:"' + $output + '"'
cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Native bridge compilation failed with exit code $LASTEXITCODE"
}
$compilerCommand = '"' + $vsDevCmd + '" -arch=x64 -host_arch=x64 >nul && cl /nologo /W4 /WX /O2 "' + $compilerSource + '" /link /OUT:"' + $compiler + '"'
cmd.exe /d /s /c $compilerCommand
if ($LASTEXITCODE -ne 0) {
    throw "Lua compiler build failed with exit code $LASTEXITCODE"
}
& $compiler $gameLua $shimSource $shimOutput
if ($LASTEXITCODE -ne 0) {
    throw "Lua shim compilation failed with exit code $LASTEXITCODE"
}
Write-Output $output
Write-Output $shimOutput
