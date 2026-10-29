"""Save editable SDK IR once, apply an evidence-derived FP16 policy, build candidate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--worker", action="store_true")
parser.add_argument("--policy", default="artifacts/diagnosis_20261010/hybrid_policy_sensitive.json")
parser.add_argument("--output", default="artifacts/manual_hybrid_20261010")
parser.add_argument("--reuse-ir", action="store_true", help="reuse this experiment's completed SDK step1 files")
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / args.output
output.mkdir(parents=True, exist_ok=True)
policy_path = root / args.policy

if not args.worker:
    log_path = output / ("build_reuse.log" if args.reuse_ir else "build.log")
    with log_path.open("w") as log:
        command = [sys.executable, __file__, "--worker", "--policy", str(policy_path), "--output", str(output)]
        if args.reuse_ir:
            command.append("--reuse-ir")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while process.poll() is None:
            time.sleep(10)
            elapsed = time.monotonic() - start
            if int(elapsed) // 30 != int(elapsed - 10) // 30:
                print(f"Manual hybrid build running: {elapsed:.0f}s", flush=True)
    print("Manual hybrid build exit: " + str(process.returncode), flush=True)
    if process.returncode:
        print(log_path.read_text(errors="replace")[-5000:].replace("\r", "\n"))
    raise SystemExit(process.returncode)

import yaml
from rknn.api import RKNN
from record_ir_provenance import fingerprint

os.chdir(output)
model = RKNN(verbose=False)
start = time.monotonic()
metadata = {"phase": "configure", "completed": False, "worker_pid": os.getpid(),
            "policy_file": str(policy_path), "source": str(root / "artifacts/clip_visual_reference_fp32.onnx"),
            "calibration": str(root / "artifacts/calibration.txt")}


def status(phase):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - start)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("PHASE " + phase, flush=True)


def check(code, phase):
    if code != 0:
        raise RuntimeError(f"{phase} returned {code}")


try:
    provenance_file = output / "ir_provenance.json"
    current_fingerprint = fingerprint(root)
    if args.reuse_ir:
        if not provenance_file.exists() or json.loads(provenance_file.read_text()) != current_fingerprint:
            raise RuntimeError("IR reuse requires identical source, calibration hashes and SDK configuration")
    elif provenance_file.exists() and json.loads(provenance_file.read_text()) != current_fingerprint:
        raise RuntimeError("experiment directory belongs to different source/calibration; choose a new output")
    else:
        provenance_file.write_text(json.dumps(current_fingerprint, indent=2))
    metadata["provenance_file"] = str(provenance_file)
    status("configure")
    check(model.config(target_platform="rk3588", quantized_dtype="w8a8", quantized_method="channel",
                       quantized_algorithm="normal", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                       optimization_level=3, auto_hybrid_cos_thresh=.995), "config")
    check(model.load_onnx(model=metadata["source"]), "load_onnx")
    if args.reuse_ir:
        status("reuse_completed_step1_ir")
    else:
        status("hybrid_step1")
        check(model.hybrid_quantization_step1(dataset=metadata["calibration"], proposal=False), "hybrid_step1")
    def single_file(pattern):
        files = list(output.glob(pattern))
        if len(files) != 1:
            raise RuntimeError(f"expected one {pattern}, got {len(files)}")
        return files[0]
    temporary_model = single_file("*.model")
    temporary_data = single_file("*.data")
    config = single_file("*.quantization.cfg")
    parsed = yaml.safe_load(config.read_text())
    policy = json.loads(policy_path.read_text())
    parameters = parsed.get("quantize_parameters", {})
    missing = [name for name in policy["tensors"] if name not in parameters]
    metadata.update(policy=policy["policy"], requested_tensors=len(policy["tensors"]), missing_tensors=missing,
                    temporary_model=str(temporary_model), temporary_data=str(temporary_data), original_cfg=str(config))
    if missing:
        status("policy_tensor_mapping_required")
        raise RuntimeError("policy names are not all in SDK quantization parameters; map saved IR names explicitly")
    # Do not silently drop requested sensitive tensors or alter scale/zero-point.
    parsed["custom_quantize_layers"] = {**(parsed.get("custom_quantize_layers") or {}),
                                       **{name: "float16" for name in policy["tensors"]}}
    selected_config = output / "sensitive_policy.cfg"
    selected_config.write_text(yaml.safe_dump(parsed, sort_keys=False))
    status("hybrid_step2")
    check(model.hybrid_quantization_step2(model_input=str(temporary_model), data_input=str(temporary_data),
                                         model_quantization_cfg=str(selected_config)), "hybrid_step2")
    artifact = output / "clip_manual_sensitive.rknn"
    check(model.export_rknn(str(artifact)), "export_rknn")
    metadata.update(completed=True, model=str(artifact), bytes=artifact.stat().st_size,
                    selected_cfg=str(selected_config))
    status("completed")
except Exception as error:
    metadata["error"] = str(error)
    status("failed")
    raise
finally:
    model.release()
