$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\YOLO-CLIP-QAT-20261010'
if (-not (Test-Path -LiteralPath $taskRoot)) { New-Item -ItemType Directory -Path $taskRoot | Out-Null }
foreach ($taskDirectory in @('assets','artifacts','reference_six')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $taskRoot $taskDirectory) | Out-Null
}
$taskBasePython = 'D:\anaconda3\envs\yolov11\python.exe'
$taskVenv = Join-Path $taskRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $taskVenv 'Scripts\python.exe'))) {
    & $taskBasePython -m venv --system-site-packages $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& (Join-Path $taskVenv 'Scripts\python.exe') -c "import torch,numpy,yaml,cv2;print(torch.__version__,torch.version.cuda,torch.cuda.is_available());print(torch.cuda.get_device_name(0));print(numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'isolated CUDA environment check failed' }
Write-Output ('TASK_ROOT ' + $taskRoot)
