"""Single small upload + hash check + independent hidden Windows training process."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import paramiko

parser = argparse.ArgumentParser()
parser.add_argument('--inspect-only',action='store_true')
parser.add_argument('--collect',action='store_true',help='collect completed pilot evidence/checkpoints')
parser.add_argument('--experiment',choices=['pilot','ranking-ablation','projection-ablation'],default='pilot')
args = parser.parse_args()
workspace = Path(__file__).resolve().parents[1]
bundle = workspace/'quantization/artifacts/qat_pilot_bundle.zip'
destination = 'D:/YOLO-CLIP-QAT-20261010'
client = paramiko.SSHClient()
client.load_host_keys(str(Path('C:/Users/23987/.ssh/known_hosts')))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
record = {'connected':False,'launched':False}
runs = {'pilot':['qat_ln_pilot_20261010'],
        'ranking-ablation':['qat_ln_kl5_20261010','qat_ln_kl5_rank1_20261010'],
        'projection-ablation':['qat_ln_projection_20261010']}[args.experiment]
record['experiment'] = args.experiment


def execute(script,timeout=45):
    prelude = "$ProgressPreference='SilentlyContinue';[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding;$ErrorActionPreference='Stop';"
    encoded = base64.b64encode((prelude+script).encode('utf-16-le')).decode('ascii')
    channel = client.get_transport().open_session(timeout=timeout)
    channel.settimeout(timeout)
    channel.exec_command('powershell.exe -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand '+encoded)
    out,err = bytearray(),bytearray()
    started = time.monotonic()
    while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready(): out.extend(channel.recv(65536))
        if channel.recv_stderr_ready(): err.extend(channel.recv_stderr(65536))
        if time.monotonic()-started > timeout:
            channel.close()
            raise TimeoutError('remote command exceeded bounded wait')
        time.sleep(.1)
    code = channel.recv_exit_status()
    if code:
        raise RuntimeError(err.decode('utf-8',errors='replace'))
    return out.decode('utf-8',errors='replace')


try:
    client.connect('127.0.0.1',port=2222,username='atlantis',
                   key_filename='C:/Users/23987/.ssh/id_ed25519',
                   look_for_keys=False,allow_agent=False,timeout=12,banner_timeout=15,auth_timeout=15)
    record['connected'] = True
    client.get_transport().set_keepalive(10)
    if not args.inspect_only and not args.collect:
        # Refuse to update a running pilot's source code or restart failed/completed runs silently.
        guards=';'.join(f"if (Test-Path 'runs/{run}/run.lock') {{ throw 'Run already started: {run}' }}" for run in runs)
        print(execute(f"Set-Location -LiteralPath '{destination}'; {guards}; if (Test-Path pipeline_status.json) {{ $taskStatus=Get-Content pipeline_status.json -Raw | ConvertFrom-Json; if($taskStatus.phase -notin @('completed','failed')) {{ throw 'Pipeline is not in terminal state' }} }}"),flush=True)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(25)
        sftp.put(str(bundle),destination+'/qat_pilot_bundle.zip',confirm=False)
        sftp.close()
        expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
        command = f"""
Set-Location -LiteralPath '{destination}'
if ((Get-FileHash 'qat_pilot_bundle.zip' -Algorithm SHA256).Hash.ToLower() -ne '{expected}') {{ throw 'Bundle hash mismatch' }}
Expand-Archive -LiteralPath 'qat_pilot_bundle.zip' -DestinationPath '{destination}' -Force
$taskReply=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='"{destination}/.venv/Scripts/python.exe" -u "{destination}/run_qat_pipeline.py" --experiment {args.experiment}';CurrentDirectory='{destination}'}}
if ($taskReply.ReturnValue -ne 0) {{ throw "Independent process creation failed: $($taskReply.ReturnValue)" }}
$taskReply.ProcessId | Set-Content -LiteralPath pipeline_pid.txt
@{{pipelinePid=$taskReply.ProcessId;launchMethod='WMI';bundleSHA='{expected}'}} | ConvertTo-Json -Compress
"""
        reply = execute(command)
        print(reply,flush=True)
        record['launch_reply'] = reply
        record['launched'] = True
    run_files=','.join("'runs/"+run+"/status.json'" for run in runs)
    inspect = f"""
Set-Location -LiteralPath '{destination}'
if(Test-Path pipeline_pid.txt) {{ $taskPid=[int](Get-Content pipeline_pid.txt); Get-Process -Id $taskPid -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU | ConvertTo-Json -Compress }}
foreach ($taskFile in @('pipeline_status.json',{run_files})) {{ if(Test-Path $taskFile) {{ Get-Content $taskFile -Raw }} }}
if(Test-Path pipeline_stdout.log) {{ Get-Content pipeline_stdout.log -Tail 6 }}
if(Test-Path pipeline_stderr.log) {{ Get-Content pipeline_stderr.log -Tail 6 }}
"""
    record['inspection'] = execute(inspect)
    print(record['inspection'],flush=True)
    if args.collect:
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(25)
        files = ['status.json','provenance.json','baseline.json','metrics.jsonl',
                 'proxy_activation_parameters.json','best_accuracy_delta.pt','best_cosine_delta.pt','latest_delta.pt']
        record['collected'] = []
        for run_name in runs:
            run = 'runs/'+run_name
            with sftp.open(destination+'/'+run+'/status.json') as stream:
                training_status = json.loads(stream.read().decode('utf-8'))
            if training_status['phase'] != 'completed':
                raise RuntimeError('run not completed; refuse partial checkpoint collection')
            local = workspace/'results/remote_4060_20261010'/run_name
            local.mkdir(parents=True,exist_ok=True)
            for name in files:
                sftp.get(destination+'/'+run+'/'+name,str(local/name))
                record['collected'].append(run_name+'/'+name)
                print('COLLECTED '+run_name+'/'+name,flush=True)
        sftp.close()
except Exception as error:
    record['error'] = repr(error)
    raise
finally:
    client.close()
    log_dir = workspace/'results/remote_4060_20261010'
    log_dir.mkdir(parents=True,exist_ok=True)
    name = 'collect_status.json' if args.collect else ('inspect_status.json' if args.inspect_only else 'pilot_deploy_status.json')
    if args.experiment!='pilot': name=args.experiment+'_'+name
    (log_dir/name).write_text(json.dumps(record,indent=2),encoding='utf-8')
