"""One verified SSH transport for scoped SFTP deployment and remote commands."""
import argparse
import base64
import json
from pathlib import Path
import time
import paramiko

parser=argparse.ArgumentParser()
parser.add_argument('--command',choices=['download','inspect'],default='download')
args=parser.parse_args()
workspace=Path(__file__).resolve().parents[1]
project=workspace/'quantization'
destination='D:/YOLO-CLIP-QAT-20261010'
log_dir=workspace/'results/remote_4060_20261010'
log_dir.mkdir(parents=True,exist_ok=True)
client=paramiko.SSHClient()
client.load_host_keys(str(Path('C:/Users/23987/.ssh/known_hosts')))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
state={'connected':False,'uploads':[],'command':args.command}
try:
    client.connect('127.0.0.1',port=2222,username='atlantis',
                   key_filename='C:/Users/23987/.ssh/id_ed25519',
                   look_for_keys=False,allow_agent=False,timeout=15,banner_timeout=20,auth_timeout=15)
    state['connected']=True
    client.get_transport().set_keepalive(15)
    print('CONNECTED 4060, one SSH transport',flush=True)
    sftp=client.open_sftp()
    sftp.get_channel().settimeout(25)
    files=[(project/'remote_download_assets.py',destination+'/remote_download_assets.py'),
           (project/'assets/clip_model_official.py',destination+'/clip_model_official.py'),
           (project/'qat_visual_core.py',destination+'/qat_visual_core.py'),
           (project/'test_qat_visual_core.py',destination+'/test_qat_visual_core.py')]
    for source in (project/'artifacts/diagnosis_20261010/inputs').glob('*.npy'):
        files.append((source,destination+'/reference_six/'+source.name))
    for source,target in files:
        sftp.put(str(source),target,confirm=True)
        state['uploads'].append({'file':source.name,'bytes':source.stat().st_size})
        print('UPLOADED '+source.name,flush=True)
    sftp.close()
    if args.command=='download':
        script=f"Set-Location -LiteralPath '{destination}'; & '.\\.venv\\Scripts\\python.exe' -u remote_download_assets.py; exit $LASTEXITCODE"
    else:
        script=f"Get-ChildItem -LiteralPath '{destination}/assets' -File | Select-Object Name,Length | ConvertTo-Json"
    prelude="$ProgressPreference='SilentlyContinue';[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding;"
    encoded=base64.b64encode((prelude+script).encode('utf-16-le')).decode('ascii')
    channel=client.get_transport().open_session(timeout=25)
    channel.exec_command('powershell.exe -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand '+encoded)
    with (log_dir/'remote_command.log').open('wb') as log:
        tick=time.monotonic()
        while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
            for ready,receive in ((channel.recv_ready,channel.recv),(channel.recv_stderr_ready,channel.recv_stderr)):
                if ready():
                    data=receive(65536)
                    log.write(data);log.flush()
                    message=data.decode('utf-8',errors='replace')
                    if 'DOWNLOAD' in message or 'VERIFIED' in message:
                        print(message.strip(),flush=True)
            if time.monotonic()-tick>=30:
                print('Remote command active; detailed output retained',flush=True)
                tick=time.monotonic()
            if not client.get_transport().is_active():
                raise RuntimeError('UU SSH transport disconnected')
            time.sleep(.2)
    state['returncode']=channel.recv_exit_status()
    if state['returncode']:
        raise RuntimeError('remote command failed: '+str(state['returncode']))
    print('REMOTE COMMAND COMPLETED',flush=True)
except Exception as error:
    state['error']=str(error)
    raise
finally:
    client.close()
    (log_dir/'connection_status.json').write_text(json.dumps(state,indent=2))
