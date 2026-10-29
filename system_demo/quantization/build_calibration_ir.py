"""Fresh bounded SDK step1 calibration; never reuse the normal algorithm's ranges."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", choices=["kl_divergence"], required=True)
parser.add_argument("--worker", action="store_true")
parser.add_argument("--max-seconds", type=int, default=1800)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / "artifacts/calibration_algorithms_20261010" / args.algorithm

if not args.worker:
    if args.max_seconds <= 0:
        raise ValueError("positive build deadline required")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "build.log").open("w") as log:
        process = subprocess.Popen([sys.executable, __file__, "--algorithm", args.algorithm, "--worker"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(5)
                elapsed = time.monotonic() - started
                if int(elapsed) // 30 != int(elapsed - 5) // 30:
                    print(f"CALIBRATION {args.algorithm}: {elapsed:.0f}s (native log retained)", flush=True)
                if elapsed > args.max_seconds:
                    raise TimeoutError("fresh calibration exceeded its bounded deadline")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            (output / "supervisor_status.json").write_text(json.dumps({"completed": False,
                "phase": "interrupted_or_deadline", "worker_pid": process.pid,
                "elapsed_seconds": time.monotonic() - started}, indent=2))
            raise
    print(f"CALIBRATION {args.algorithm} exit {process.returncode}", flush=True)
    if process.returncode:
        print((output / "build.log").read_text(errors="replace")[-4000:].replace("\r", "\n"))
    raise SystemExit(process.returncode)

import yaml
from rknn.api import RKNN
from calibration_policy import reapply_precision_policy
from record_ir_provenance import fingerprint, digest

os.chdir(output)
started = time.monotonic()
metadata = {"phase": "verify", "completed": False, "pid": os.getpid(), "algorithm": args.algorithm}


def status(phase, **details):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("PHASE " + phase, flush=True)


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}: {code}")


model = RKNN(verbose=False)
try:
    status("verify")
    previous = json.loads((root / "artifacts/manual_hybrid_20261010/ir_provenance.json").read_text())
    current = fingerprint(root)
    if current != previous or len(current["calibration_inputs"]) != 120:
        raise RuntimeError("original source, 120 calibration inputs or SDK configuration changed")
    provenance = fingerprint(root, algorithm=args.algorithm)
    unchanged = dict(provenance, algorithm="normal")
    if unchanged != previous:
        raise RuntimeError("new calibration must differ only in algorithm")
    (output / "ir_provenance.json").write_text(json.dumps(provenance, indent=2))
    config = {"target_platform": "rk3588", "quantized_dtype": "w8a8", "quantized_method": "channel",
        "quantized_algorithm": args.algorithm, "mean_values": [[0, 0, 0]], "std_values": [[1, 1, 1]],
        "optimization_level": 3, "auto_hybrid_cos_thresh": 0.995}
    status("configure", sdk_config=config, calibration_count=120, source_sha=provenance["source_sha256"])
    check(model.config(**config), "config")
    check(model.load_onnx(model=str(root / "artifacts/clip_visual_reference_fp32.onnx")), "load_onnx")
    status("fresh_hybrid_step1")
    check(model.hybrid_quantization_step1(dataset=str(root / "artifacts/calibration.txt"), proposal=False), "step1")
    files = {}
    for suffix in ["model", "data", "quantization.cfg"]:
        found = list(output.glob("*." + suffix))
        if len(found) != 1:
            raise RuntimeError("expected exactly one step1 file: " + suffix)
        files[suffix] = found[0]
    reference = yaml.safe_load((root / "artifacts/sdpa_safe_full_20261010/sensitive_except_sdpa/policy.cfg").read_text())
    fresh = yaml.safe_load(files["quantization.cfg"].read_text())
    transferred = reapply_precision_policy(fresh, reference)
    if len(transferred["custom_quantize_layers"]) != 126:
        raise RuntimeError("expected the same 126 FP16 protection boundaries")
    if provenance != fingerprint(root, algorithm=args.algorithm):
        raise RuntimeError("source/calibration changed while building")
    status("completed", completed=True,
        step1_files={suffix: {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                     for suffix, path in files.items()},
        policy_mapping_verified=True, protected_tensor_count=126, sdpa_int8_count=24)
except Exception as error:
    status("failed", error=repr(error))
    raise
finally:
    model.release()
