"""Continue the predeclared reconstruction controls only after affine/NPU queue exits."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
base = root / "results/real_eval_200_20261011/proposal_diagnostic"
output = base / "reconstruction_followup_20261011"
output.mkdir(exist_ok=False)
waiting = [base / "affine_sweep_20261011/status.json",
    root / "quantization/artifacts/method_sweep_20261011/reconstruction_queue_irfix_20261011/status.json"]
started = time.monotonic()
try:
    while True:
        states = [json.loads(path.read_text()) if path.exists() else {} for path in waiting]
        if any(state.get("phase") == "failed" for state in states):
            raise RuntimeError("earlier queue failed; inspect it before further board work")
        if all(state.get("completed") for state in states):
            break
        if time.monotonic() - started > 7200:
            raise TimeoutError("previous queue did not finish within 2h")
        (output / "status.json").write_text(json.dumps({"completed": False, "phase": "waiting_prior_queues"}))
        time.sleep(20)
    (output / "status.json").write_text(json.dumps({"completed": False, "phase": "board_reconstruction_controls"}))
    with (output / "board.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(root / "tools/run_real_candidate_sweep.py"),
            "--run", "reconstruction_sweep_20261011", "--reconstruction-attempt", "irfix_20261011",
            "--profiles", "bias_down", "adaround_down", "adaround_bias_down"], cwd=root,
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    (output / "status.json").write_text(json.dumps({"completed": True, "phase": "completed"}))
except Exception as error:
    (output / "status.json").write_text(json.dumps({"completed": False, "phase": "failed", "error": repr(error)}))
    raise
