param([string]$Python = '.\.fresh-env\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:TEMP = "$PSScriptRoot\.runtime\temp"
    $env:TMP = $env:TEMP
    $env:TORCH_HOME = "$PSScriptRoot\.runtime\torch"
    $env:MPLCONFIGDIR = "$PSScriptRoot\.runtime\matplotlib"
    $env:PYTHONDONTWRITEBYTECODE = '1'
    foreach ($module in 'src.extra_corrected','src.extra_svm_augmentation','src.extra_neural','src.check_cam_localization','src.extra_legacy_noise','src.summarize_extras') {
        & $Python -m $module
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed for $module" }
    }
} finally { Pop-Location }
