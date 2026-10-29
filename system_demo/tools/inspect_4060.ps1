$taskProbe = @'
import sys,json,importlib.util
report={'python':sys.version,'executable':sys.executable,'torch_installed':importlib.util.find_spec('torch') is not None}
if report['torch_installed']:
    import torch
    report.update(torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        report['device']=torch.cuda.get_device_name(0)
        report['cuda_matmul_sum']=(torch.ones((4,4),device='cuda') @ torch.ones((4,4),device='cuda')).sum().item()
report['packages']={name:importlib.util.find_spec(name) is not None for name in ('numpy','PIL','cv2','onnx')}
print(json.dumps(report))
'@
$taskCandidates = @('D:\anaconda3\envs\my_pytorch\python.exe','D:\anaconda3\envs\cv\python.exe','D:\anaconda3\envs\yolov11\python.exe','D:\ProgramData\anaconda3\envs\WDMoE\python.exe')
foreach ($taskPython in $taskCandidates) {
    if (Test-Path -LiteralPath $taskPython) {
        & $taskPython -c $taskProbe
    }
}
