"""Detached original-model reference so a tunnel outage cannot terminate computation."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
with (root / "real_eval_200_20261011/original_reference_launch.lock").open("x") as lock:
    lock.write("one-shot original FP32 evaluation")
with (root / "real_eval_200_20261011/original_reference_stdout.log").open("x") as stdout, (root / "real_eval_200_20261011/original_reference_stderr.log").open("x") as stderr:
    worker = subprocess.Popen([sys.executable, "-u", str(root / "run_remote_project_fp32.py")],
        cwd=root, stdout=stdout, stderr=stderr)
    status = root / "real_eval_200_20261011/original_reference_launch.json"
    status.write_text(json.dumps({"phase": "running", "worker_pid": worker.pid}))
    result = worker.wait()
    status.write_text(json.dumps({"phase": "completed" if result == 0 else "failed", "returncode": result}))
