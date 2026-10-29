"""Rebuild original INT8 settings once, compare simulator and dump layer errors.

The parent retains native compiler output in a log. No board deployment is changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--output", default="artifacts/diagnosis_20261010/layer_analysis")
parser.add_argument("--sample", default="cardboard_020.png")
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / args.output
output.mkdir(parents=True, exist_ok=True)

if not args.worker:
    with (output / "analysis.log").open("w") as log:
        process = subprocess.Popen([sys.executable, __file__, "--worker", "--output", str(output),
                                    "--sample", args.sample], stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while process.poll() is None:
            time.sleep(10)
            elapsed = time.monotonic() - start
            if int(elapsed) // 30 != int(elapsed - 10) // 30:
                metadata = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
                print(f"Layer diagnostic running: {elapsed:.0f}s, phase={metadata.get('phase', 'startup')}", flush=True)
    print("Layer diagnostic exit: " + str(process.returncode), flush=True)
    if process.returncode:
        print((output / "analysis.log").read_text(errors="replace")[-4000:].replace("\r", "\n"))
    raise SystemExit(process.returncode)

import numpy as np
from rknn.api import RKNN

os.chdir(output)
model = RKNN(verbose=False)
start = time.monotonic()
metadata = {"worker_pid": os.getpid(), "phase": "configure", "completed": False,
            "sample": args.sample, "target_platform": "rk3588", "algorithm": "normal",
            "quantized_method": "channel", "optimization_level": 3,
            "calibration": str(root / "artifacts/calibration.txt")}


def status(phase):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - start)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("PHASE " + phase, flush=True)


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} returned {code}")


try:
    status("configure")
    check(model.config(target_platform="rk3588", quantized_dtype="w8a8", quantized_method="channel",
                       quantized_algorithm="normal", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                       optimization_level=3, auto_hybrid_cos_thresh=0.995), "config")
    check(model.load_onnx(model=str(root / "artifacts/clip_visual_reference_fp32.onnx")), "load_onnx")
    status("build_same_configuration")
    check(model.build(do_quantization=True, dataset=str(root / "artifacts/calibration.txt")), "build")
    check(model.export_rknn(str(output / "diagnostic_int8.rknn")), "export_rknn")
    status("accuracy_analysis_simulator")
    sample = root / "artifacts/diagnosis_20261010/inputs" / (args.sample + ".npy")
    check(model.accuracy_analysis(inputs=[str(sample)], output_dir=str(output / "snapshot")), "accuracy_analysis")
    status("simulator_six_samples")
    check(model.init_runtime(), "init_runtime")
    diagnosis = root / "artifacts/diagnosis_20261010"
    names = [row["image"] for row in json.loads((diagnosis / "reference_comparison.json").read_text())["rows"]]
    outputs = []
    for name in names:
        tensor = np.load(diagnosis / "inputs" / (name + ".npy"))
        result = model.inference(inputs=[tensor], data_format=["nchw"])
        if result is None or len(result) != 1:
            raise RuntimeError("simulator returned invalid outputs")
        outputs.append(result[0].astype(np.float32).reshape(-1))
        print("SIMULATOR " + name, flush=True)
    np.save(output / "simulator_int8.f32.npy", np.asarray(outputs))
    metadata["completed"] = True
    status("completed")
except Exception as error:
    metadata["error"] = str(error)
    status("failed")
    raise
finally:
    model.release()
