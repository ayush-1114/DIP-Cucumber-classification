# Run from the repository root after the documented setup.
param(
    [Parameter(Mandatory=$true)][string]$Photo,
    [string]$Python = ".\mlee-env\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$resolvedPhoto = (Resolve-Path -LiteralPath $Photo).Path
& $Python -m src.leaf_demo $resolvedPhoto
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m src.evaluate_leaf_demo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m src.dip_stage_storage
exit $LASTEXITCODE
