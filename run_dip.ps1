param(
    [string]$Photo = ".\demo_input",
    [string]$Python = ".\mlee-env\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & $Python -m src.leaf_demo $Photo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m src.evaluate_leaf_demo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m src.dip_stage_storage
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
