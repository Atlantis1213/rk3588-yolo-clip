"""WMI-launched detached supervisor; records worker status and never restarts a run."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
if (root / "runs/adaround_down_kl120_20261011").exists():
    raise RuntimeError("training run already exists; inspect it rather than starting another")
with (root / "adaround_wmi_20261011.lock").open("x") as lock:
    lock.write("independent one-shot reconstruction supervisor")
status = root / "adaround_wmi_20261011_status.json"
with (root / "adaround_wmi_20261011_stdout.log").open("x") as stdout, (root / "adaround_wmi_20261011_stderr.log").open("x") as stderr:
    process = subprocess.Popen([sys.executable, "-u", str(root / "reconstruct_rounding.py"), "--steps", "500"],
        cwd=root, stdout=stdout, stderr=stderr)
    status.write_text(json.dumps({"phase": "running", "worker_pid": process.pid, "completed": False}))
    result = process.wait()
    status.write_text(json.dumps({"phase": "completed" if result == 0 else "failed",
        "worker_pid": process.pid, "returncode": result, "completed": result == 0}))
