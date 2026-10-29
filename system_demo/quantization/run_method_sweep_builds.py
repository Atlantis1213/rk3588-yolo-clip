"""One sequential CPU queue; never duplicate existing or active conversion work."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
lock = (sweep / ".cpu_queue.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
queue = sweep / "cpu_queue_20261011"
queue.mkdir(exist_ok=False)
info = {"phase": "started", "completed": False, "jobs": [], "started_at": time.time()}


def status(phase, **values):
    info.update(phase=phase, **values)
    (queue / "status.json").write_text(json.dumps(info, indent=2))
    print("QUEUE " + phase, flush=True)


def run(command, directory, label):
    if directory.exists():
        existing = json.loads((directory / "status.json").read_text()) if (directory / "status.json").exists() else {}
        if existing.get("completed"):
            info["jobs"].append({"label": label, "skipped_already_completed": True})
            return
        raise RuntimeError("existing incomplete job requires inspection, not automatic restart: " + str(directory))
    status("running_" + label)
    with (queue / (label + ".log")).open("x") as log:
        result = subprocess.run([sys.executable, *command], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    info["jobs"].append({"label": label, "returncode": result.returncode})
    if result.returncode:
        raise RuntimeError("job failed; retained artifacts and stopped queue: " + label)


try:
    # The current normal30 supervisor is independent. Inspect and wait, never start it twice.
    normal = sweep / "normal30/status.json"
    while True:
        current = json.loads(normal.read_text())
        if current.get("completed"):
            break
        if current.get("phase") == "failed":
            raise RuntimeError("normal30 failed; investigate before proceeding")
        process = subprocess.run(["ps", "-p", str(current["pid"]), "-o", "args="], capture_output=True, text=True)
        if process.returncode or "build_sweep_calibration.py" not in process.stdout:
            raise RuntimeError("normal30 process is not alive and its status is incomplete")
        status("waiting_normal30")
        time.sleep(30)
    run(["build_sweep_calibration.py", "--profile", "kl30"], sweep / "kl30", "kl30")
    # Predeclared candidates; development results do not alter the calibration or export.
    for profile in ["repq_ln2", "smooth_ln2_a25", "smooth_ln2_a50", "smooth_ln2_a75", "repq_all", "smooth_all_a50"]:
        run(["export_affine_sweep.py", "--profile", profile], sweep / "affine_sources" / profile, "export_" + profile)
        run(["build_affine_candidate.py", "--profile", profile], sweep / "affine_rknn" / profile, "build_" + profile)
    status("completed", completed=True)
except Exception as error:
    status("failed", error=repr(error))
    raise
