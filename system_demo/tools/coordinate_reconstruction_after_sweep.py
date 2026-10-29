"""Wait for both existing pipelines before validating the three reconstruction controls."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
output = root / "results/method_sweep_20261011/reconstruction_wait_20261011"
output.mkdir(exist_ok=False)
checks = [root / "quantization/artifacts/method_sweep_20261011/reconstruction_queue_20261011/status.json",
    root / "results/method_sweep_20261011/coordinator_20261011/status.json"]
try:
    while True:
        states = [json.loads(path.read_text()) if path.exists() else {} for path in checks]
        if any(state.get("phase") == "failed" for state in states):
            raise RuntimeError("earlier pipeline failed; inspect preserved artifacts before starting another board run")
        if all(state.get("completed") for state in states):
            break
        (output / "status.json").write_text(json.dumps({"phase": "waiting_serial_pipelines", "completed": False}))
        time.sleep(30)
    (output / "status.json").write_text(json.dumps({"phase": "board_validation", "completed": False}))
    subprocess.run([sys.executable, "-u", str(root / "tools/coordinate_method_sweep.py"),
        "--profiles", "bias_down", "adaround_down", "adaround_bias_down", "--run-directory", "coordinator_reconstruction_20261011"],
        cwd=root, check=True)
    (output / "status.json").write_text(json.dumps({"phase": "completed", "completed": True}))
except Exception as error:
    (output / "status.json").write_text(json.dumps({"phase": "failed", "completed": False, "error": repr(error)}))
    raise
