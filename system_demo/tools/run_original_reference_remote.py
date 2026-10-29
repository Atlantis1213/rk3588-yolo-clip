"""Upload only exact inputs and keep one SSH session for launch and results."""
import base64
import json
from pathlib import Path
import time
import paramiko

root = Path(__file__).resolve().parents[1]
remote_root = "D:/YOLO-CLIP-QAT-20261010"
client = paramiko.SSHClient()
known_hosts = Path.home() / ".ssh/known_hosts"
client.load_host_keys(str(known_hosts))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
config = paramiko.SSHConfig()
if (Path.home() / ".ssh/config").exists():
    with (Path.home() / ".ssh/config").open() as handle:
        config.parse(handle)
keys = [str(Path(path).expanduser()) for path in config.lookup("127.0.0.1").get("identityfile", [])]
try:
    client.connect("127.0.0.1", port=2222, username="atlantis", key_filename=keys or None,
        timeout=10, banner_timeout=15, auth_timeout=15, allow_agent=True, look_for_keys=True)
    client.get_transport().set_keepalive(10)
    sftp = client.open_sftp()
    sftp.get_channel().settimeout(30)
    files = {root / "results/real_eval_200_20261011/exact_frozen_inputs.zip": remote_root + "/real_eval_200_20261011/exact_frozen_inputs.zip",
        root / "quantization/run_remote_project_fp32.py": remote_root + "/run_remote_project_fp32.py",
        root / "quantization/launch_remote_project_reference.py": remote_root + "/launch_remote_project_reference.py"}
    for source, destination in files.items():
        try:
            sftp.stat(destination)
        except IOError:
            pass
        else:
            raise RuntimeError("remote file exists; inspect before replacing: " + destination)
        print("UPLOAD " + source.name, flush=True)
        sftp.put(str(source), destination)
    code = '$taskStarted = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="D:\\YOLO-CLIP-QAT-20261010\\.venv\\Scripts\\python.exe D:\\YOLO-CLIP-QAT-20261010\\launch_remote_project_reference.py"; CurrentDirectory="D:\\YOLO-CLIP-QAT-20261010"}; $taskStarted | ConvertTo-Json -Compress'
    # PowerShell literal paths need single backslashes, not Python string representation escaping.
    code = code.replace("\\\\", "\\")
    encoded = base64.b64encode(code.encode("utf-16le")).decode()
    _, stdout, stderr = client.exec_command("powershell.exe -NoProfile -NonInteractive -EncodedCommand " + encoded)
    reply = stdout.read().decode("utf-8-sig", errors="replace")
    errors = stderr.read().decode("utf-8-sig", errors="replace")
    print(reply, flush=True)
    if stdout.channel.recv_exit_status() or '"ReturnValue":0' not in reply.replace(" ", ""):
        raise RuntimeError("detached launch not confirmed: " + errors)
    remote_output = remote_root + "/real_eval_200_20261011/recovered_project_fp32"
    while True:
        try:
            with sftp.open(remote_output + "/status.json", "r") as handle:
                status = json.load(handle)
        except IOError:
            time.sleep(10)
            continue
        print("ORIGINAL REFERENCE " + status["phase"] + " " + str(status["count_completed"]), flush=True)
        if status["phase"] == "failed":
            with sftp.open(remote_root + "/real_eval_200_20261011/original_reference_stderr.log") as handle:
                print(handle.read().decode(errors="replace"), flush=True)
            raise RuntimeError("remote original reference failed")
        if status["completed"]:
            break
        time.sleep(20)
    output = root / "results/real_eval_200_20261011/recovered_project_fp32_verified"
    output.mkdir(exist_ok=False)
    for name in ["status.json", "manifest.json", "embeddings.f32"]:
        sftp.get(remote_output + "/" + name, str(output / name))
    print("Original FP32 outputs received; exact frozen inputs verified remotely", flush=True)
finally:
    client.close()
