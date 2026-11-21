param(
    [string]$Python = ".\mlee-env\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    foreach ($module in "src.fetch_field_data","src.train_corrected","src.evaluate_leaf_demo","src.dip_stage_storage","src.plot_corrected") {
        & $Python -m $module
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} finally {
    Pop-Location
}
