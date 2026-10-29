"""Single-variable precision controls using fingerprint-verified, read-only SDK IR."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["all_fp16", "sensitive_gelu", "all_fp16_io", "sensitive_gelu_io", "all_fp16_io_opt0"], required=True)
parser.add_argument("--worker", action="store_true")
parser.add_argument("--simulate", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
ir = root / "artifacts/manual_hybrid_20261010"
output = root / "artifacts/precision_controls_20261010" / args.mode
output.mkdir(parents=True, exist_ok=True)

if not args.worker:
    log_path = output / "build.log"
    if (output / "clip.rknn").exists():
        raise SystemExit("control already exported; use existing artifact instead of overwriting")
    with log_path.open("w") as log:
        command = [sys.executable, __file__, "--mode", args.mode, "--worker"]
        if args.simulate:
            command.append("--simulate")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(10)
                elapsed = time.monotonic() - start
                if int(elapsed) // 30 != int(elapsed - 10) // 30:
                    print(f"Precision control {args.mode}: {elapsed:.0f}s (log retained)", flush=True)
        except BaseException:
            process.terminate()
            process.wait(timeout=20)
            raise
    print(f"Precision control {args.mode} exit: {process.returncode}", flush=True)
    if process.returncode:
        print(log_path.read_text(errors="replace")[-5000:].replace("\r", "\n"))
    raise SystemExit(process.returncode)

import numpy as np
import yaml
from rknn.api import RKNN
from record_ir_provenance import fingerprint, digest

os.chdir(output)
started = time.monotonic()
metadata = {"mode": args.mode, "worker_pid": os.getpid(), "phase": "verify_ir", "completed": False}
model = RKNN(verbose=False)


def status(phase):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - started)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("PHASE " + phase, flush=True)


def check(code, phase):
    if code != 0:
        raise RuntimeError(f"{phase} failed: {code}")


try:
    status("verify_ir")
    recorded = json.loads((ir / "ir_provenance.json").read_text())
    if recorded != fingerprint(root):
        raise RuntimeError("source/calibration/SDK fingerprint differs from saved IR")
    config_path = ir / "clip_visual_reference_fp32.quantization.cfg"
    config = yaml.safe_load(config_path.read_text())
    parameters = config["quantize_parameters"]
    layers = json.loads((root / "artifacts/diagnosis_20261010/layer_summary.json").read_text())["layers"]
    policy_mode = 'all_fp16_io' if args.mode=='all_fp16_io_opt0' else args.mode
    if policy_mode in {"all_fp16", "all_fp16_io"}:
        requested = [row["tensor"] for row in layers if row["op"] != "Input" and
                     (policy_mode == "all_fp16_io" or
                      (row["op"] != "exDataConvert" and not row["tensor"].endswith("_int8")))]
        # Only actual tensor names present in the saved SDK configuration can be assigned.
        # Report virtual layout/IO wrappers separately, rather than silently claiming coverage.
        selected = [name for name in requested if name in parameters]
        unsupported = [name for name in requested if name not in parameters]
    else:
        requested = [row["tensor"] for row in layers if row["op"] in {"exNorm", "Add", "exSDPAttention", "exSwish"} or
                     (args.mode == "sensitive_gelu_io" and row["tensor"] in {"pixel_values_int8", "image_embedding_int8", "image_embedding"})]
        unsupported = [name for name in requested if name not in parameters]
        if unsupported:
            raise RuntimeError("sensitive policy contains unmapped tensors: " + str(unsupported))
        selected = requested
    selected = list(dict.fromkeys(selected))
    config["custom_quantize_layers"] = {name: "float16" for name in selected}
    policy_path = output / "policy.cfg"
    policy_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata.update(selected_tensor_count=len(selected), unsupported_virtual_tensors=unsupported,
                    step2_optimization_level=0 if args.mode=='all_fp16_io_opt0' else 3,
                    note='Step1 IR remains the same opt3 artifact; only step2 optimization config varies',
                    excluded_io_wrappers=[row["tensor"] for row in layers if row["op"] in {"Input", "exDataConvert"}
                                          or row["tensor"].endswith("_int8")],
                    original_cfg_sha256=digest(config_path), selected_cfg_sha256=digest(policy_path),
                    protected_tensors=selected, source_sha256=recorded["source_sha256"])
    status("hybrid_step2")
    check(model.config(target_platform="rk3588", quantized_dtype="w8a8", quantized_method="channel",
                       quantized_algorithm="normal", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]],
                       optimization_level=metadata['step2_optimization_level'], auto_hybrid_cos_thresh=.995), "config")
    check(model.hybrid_quantization_step2(model_input=str(ir / "clip_visual_reference_fp32.model"),
                                         data_input=str(ir / "clip_visual_reference_fp32.data"),
                                         model_quantization_cfg=str(policy_path)), "hybrid_step2")
    artifact = output / "clip.rknn"
    check(model.export_rknn(str(artifact)), "export_rknn")
    metadata.update(model=str(artifact), model_bytes=artifact.stat().st_size, model_sha256=digest(artifact))
    if args.simulate:
        status("simulator_six_samples")
        check(model.init_runtime(), "init_runtime")
        diagnosis = root / "artifacts/diagnosis_20261010"
        names = [row["image"] for row in json.loads((diagnosis / "reference_comparison.json").read_text())["rows"]]
        embeddings = []
        for name in names:
            tensor = np.load(diagnosis / "inputs" / (name + ".npy"))
            result = model.inference(inputs=[tensor], data_format=["nchw"])
            if result is None or len(result) != 1 or not np.isfinite(result[0]).all():
                raise RuntimeError("invalid simulator outputs")
            embeddings.append(result[0].reshape(-1))
            print("SIMULATOR " + name, flush=True)
        np.save(output / "simulator.f32.npy", np.asarray(embeddings, dtype=np.float32))
    metadata["completed"] = True
    status("completed")
except Exception as error:
    metadata["error"] = str(error)
    status("failed")
    raise
finally:
    model.release()
