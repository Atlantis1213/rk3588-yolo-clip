"""Normal/KL/MMSE on the same balanced 30-image subset, with safe mixed precision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=["normal30", "kl30", "mmse30", "kl240"], required=True)
parser.add_argument("--worker", action="store_true")
parser.add_argument("--max-seconds", type=int, default=43200)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / "artifacts/method_sweep_20261011" / args.profile
algorithm = {"normal30": "normal", "kl30": "kl_divergence", "mmse30": "mmse", "kl240": "kl_divergence"}[args.profile]
if not args.worker:
    if args.max_seconds <= 0:
        raise ValueError("positive deadline required")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "build.log").open("w") as log:
        process = subprocess.Popen([sys.executable, __file__, "--profile", args.profile, "--worker"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(5)
                elapsed = time.monotonic() - started
                if int(elapsed) // 30 != int(elapsed - 5) // 30:
                    print(f"SWEEP {args.profile}: {elapsed:.0f}s (native log retained)", flush=True)
                if elapsed > args.max_seconds:
                    raise TimeoutError("native calibration exceeded its recorded compute deadline")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            (output / "supervisor_status.json").write_text(json.dumps({"phase": "interrupted_or_deadline",
                "worker_pid": process.pid, "completed": False, "elapsed_seconds": time.monotonic() - started}, indent=2))
            raise
    print(f"SWEEP {args.profile} exit {process.returncode}", flush=True)
    if process.returncode:
        print((output / "build.log").read_text(errors="replace")[-4000:].replace("\r", "\n"))
    raise SystemExit(process.returncode)

import numpy as np
import yaml
from rknn.api import RKNN
from calibration_policy import reapply_precision_policy
from record_ir_provenance import fingerprint, digest
from calibration_selection import expanded_calibration

os.chdir(output)
started = time.monotonic()
metadata = {"profile": args.profile, "algorithm": algorithm, "pid": os.getpid(), "completed": False}


def status(phase, **details):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("PHASE " + phase, flush=True)


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}: {code}")


context = RKNN(verbose=False)
try:
    status("verify")
    previous = json.loads((root / "artifacts/manual_hybrid_20261010/ir_provenance.json").read_text())
    if fingerprint(root) != previous:
        raise RuntimeError("original source, calibration or SDK changed")
    original_paths = [Path(line) for line in (root / "artifacts/calibration.txt").read_text().splitlines() if line]
    if args.profile == "kl240":
        selected = expanded_calibration(root, original_paths, digest)
    else:
        grouped = {}
        for path in original_paths:
            grouped.setdefault(path.stem.split("_")[0], []).append(path)
        selected = [path for label in sorted(grouped) for path in grouped[label][:5]]
        if len(grouped) != 6 or len(selected) != 30 or len(set(selected)) != 30:
            raise RuntimeError("expected identical balanced 30-image calibration across profiles")
    for path in selected:
        value = np.load(path, allow_pickle=False)
        if (value.shape != (1, 3, 224, 224) or value.dtype != np.float32 or not np.isfinite(value).all()
            or not np.array_equal(value, value.astype(np.float16).astype(np.float32))):
            raise RuntimeError("normalized NCHW FP16-rounded FP32 calibration required")
    calibration = output / ("calibration" + str(len(selected)) + ".txt")
    calibration.write_text("\n".join(map(str, selected)) + "\n")
    source = root / "artifacts/clip_visual_reference_fp32.onnx"
    provenance = {"algorithm": algorithm, "source_sha256": digest(source),
        "toolkit_version": previous["toolkit_version"], "target_platform": "rk3588",
        "calibration_inputs": [{"name": path.name, "sha256": digest(path)} for path in selected],
        "config": {"target_platform": "rk3588", "quantized_dtype": "w8a8", "quantized_method": "channel",
            "quantized_algorithm": algorithm, "mean_values": [[0, 0, 0]], "std_values": [[1, 1, 1]],
            "optimization_level": 3, "auto_hybrid_cos_thresh": 0.995}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    status("configure", source_sha=provenance["source_sha256"], calibration_count=len(selected),
        same_source_as_reference=True, same_30_inputs_across_algorithms=args.profile != "kl240",
        calibration_domain="public TrashNet, not user real-scene dataset")
    check(context.config(**provenance["config"]), "config")
    check(context.load_onnx(model=str(source)), "load_onnx")
    status("fresh_hybrid_step1")
    check(context.hybrid_quantization_step1(dataset=str(calibration), proposal=False), "step1")
    reference = yaml.safe_load((root / "artifacts/sdpa_safe_full_20261010/sensitive_except_sdpa/policy.cfg").read_text())
    original_cfg = output / "clip_visual_reference_fp32.quantization.cfg"
    fresh = yaml.safe_load(original_cfg.read_text())
    policy = reapply_precision_policy(fresh, reference)
    if len(policy["custom_quantize_layers"]) != 126:
        raise RuntimeError("precision policy differs from baseline")
    policy_file = output / "policy.cfg"
    policy_file.write_text(yaml.safe_dump(policy, sort_keys=False))
    status("hybrid_step2", protected_tensor_count=126, sdpa_int8_count=24)
    check(context.hybrid_quantization_step2(model_input=str(output / "clip_visual_reference_fp32.model"),
        data_input=str(output / "clip_visual_reference_fp32.data"), model_quantization_cfg=str(policy_file)), "step2")
    artifact = output / "clip.rknn"
    check(context.export_rknn(str(artifact)), "export")
    status("simulator_calibration_sample", model_sha=digest(artifact), model_bytes=artifact.stat().st_size)
    check(context.init_runtime(), "simulator init")
    result = context.inference(inputs=[np.load(selected[0])], data_format=["nchw"])
    if result is None or len(result) != 1 or result[0].size != 768 or not np.isfinite(result[0]).all():
        raise RuntimeError("invalid simulator calibration output")
    np.save(output / "simulator_calibration.npy", result[0])
    if fingerprint(root) != previous:
        raise RuntimeError("original source/calibration changed while compiling")
    status("completed", completed=True, policy_sha=digest(policy_file))
except Exception as error:
    status("failed", error=repr(error))
    raise
finally:
    context.release()
