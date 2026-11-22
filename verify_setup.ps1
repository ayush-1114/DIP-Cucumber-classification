$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:TEMP = "$PSScriptRoot\.runtime\temp"
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = "$PSScriptRoot\.runtime\pip-cache"
    $env:PYTHONUSERBASE = "$PSScriptRoot\.runtime\userbase"
    $env:TORCH_HOME = "$PSScriptRoot\.runtime\torch"
    $env:MPLCONFIGDIR = "$PSScriptRoot\.runtime\matplotlib"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $pythonExe = "$PSScriptRoot\.fresh-env\Scripts\python.exe"
    & $pythonExe -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
    foreach ($module in 'src.evaluate_leaf_demo','src.dip_stage_storage','src.plot_corrected','src.leaf_demo') {
        & $pythonExe -m $module
        if ($LASTEXITCODE -ne 0) { throw "Verification failed for $module" }
    }
} finally {
    Pop-Location
}
