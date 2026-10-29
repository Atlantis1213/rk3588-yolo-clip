"""Queue copied-IR reconstruction after the existing CPU sweep releases its lock."""
import fcntl
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--attempt", required=True)
args = parser.parse_args()
if not args.attempt.replace("_", "").replace("-", "").isalnum():
    parser.error("invalid attempt directory name")
sweep = root / "artifacts/method_sweep_20261011"
output = sweep / ("reconstruction_queue_" + args.attempt)
output.mkdir(exist_ok=False)
info = {"phase": "waiting_cpu_sweep", "completed": False, "pid": os.getpid(), "jobs": []}
lock = (sweep / ".cpu_queue.lock").open("a")


def update(phase, **details):
    info.update(phase=phase, **details)
    (output / "status.json").write_text(json.dumps(info, indent=2))
    print("RECONSTRUCTION QUEUE " + phase, flush=True)


try:
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            update("waiting_cpu_sweep")
            time.sleep(30)
    for profile in ["bias_down", "adaround_down", "adaround_bias_down"]:
        update("build_" + profile)
        with (output / (profile + ".log")).open("x") as log:
            subprocess.run([sys.executable, "-u", str(root / "build_reconstruction_candidate.py"), "--profile", profile,
                "--attempt", args.attempt],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        info["jobs"].append({"profile": profile, "completed": True})
    update("completed", completed=True)
except Exception as error:
    update("failed", error=repr(error))
    raise
