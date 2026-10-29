"""Fresh KL120 calibration for a float-equivalent LN-folded ONNX source."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROFILES = ["repq_ln2", "repq_all", "smooth_ln2_a25", "smooth_ln2_a50", "smooth_ln2_a75", "smooth_all_a50"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=43200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sweep = root / "artifacts/method_sweep_20261011"
    source_dir = sweep / "affine_sources" / args.profile
    output = sweep / "affine_rknn" / args.profile
    if not args.worker:
        if args.max_seconds <= 0:
            raise ValueError("positive deadline required")
        # Refuse an unverified ONNX before creating any conversion artifacts.
        if not json.loads((source_dir / "status.json").read_text()).get("completed"):
            raise RuntimeError("float equivalence must pass before RKNN conversion")
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
                        print(f"AFFINE BUILD {args.profile}: {elapsed:.0f}s", flush=True)
                    if elapsed > args.max_seconds:
                        raise TimeoutError("conversion exceeded its recorded compute deadline")
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
        print(f"AFFINE BUILD {args.profile} exit {process.returncode}", flush=True)
        if process.returncode:
            print((output / "build.log").read_text(errors="replace")[-6000:].replace("\r", "\n"))
        return process.returncode

    import numpy as np
    import yaml
    from rknn.api import RKNN
    from calibration_policy import reapply_precision_policy
    from record_ir_provenance import fingerprint, digest
    os.chdir(output)
    started = time.monotonic()
    info = {"profile": args.profile, "algorithm": "kl_divergence", "pid": os.getpid(), "completed": False,
        "calibration_count": 120, "same_calibration_as_kl120": True, "extra_inference_nodes": 0}
    def status(phase, **values):
        info.update(phase=phase, elapsed_seconds=time.monotonic() - started, **values)
        (output / "status.json").write_text(json.dumps(info, indent=2))
        print("PHASE " + phase, flush=True)
    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"{operation}: {code}")
    context = RKNN(verbose=False)
    try:
        status("verify_source")
        original = json.loads((root / "artifacts/manual_hybrid_20261010/ir_provenance.json").read_text())
        if fingerprint(root) != original:
            raise RuntimeError("original source/calibration/SDK fingerprint changed")
        affine = json.loads((source_dir / "status.json").read_text())
        source = source_dir / "source.onnx"
        weights = source_dir / "weights.bin"
        if (not affine["completed"] or not affine["operator_signature_unchanged"]
            or affine["original_source_sha"] != original["source_sha256"]
            or digest(source) != affine["onnx_sha"] or digest(weights) != affine["external_weights_sha"]
            or len(affine["float_equivalence"]) != 6
            or any(row["cosine"] < .99999 or row["max_abs_error"] > .001 for row in affine["float_equivalence"])):
            raise RuntimeError("verified affine source gate failed")
        paths = [Path(line) for line in (root / "artifacts/calibration.txt").read_text().splitlines() if line]
        if len(paths) != 120:
            raise RuntimeError("original 120 calibration inputs required")
        calibration = output / "calibration120.txt"
        calibration.write_text("\n".join(map(str, paths)) + "\n")
        config = {"target_platform": "rk3588", "quantized_dtype": "w8a8", "quantized_method": "channel",
            "quantized_algorithm": "kl_divergence", "mean_values": [[0, 0, 0]], "std_values": [[1, 1, 1]],
            "optimization_level": 3, "auto_hybrid_cos_thresh": .995}
        (output / "provenance.json").write_text(json.dumps({"source_affine_status": affine,
            "original_fingerprint": original, "config": config,
            "calibration_inputs": [{"name": path.name, "sha256": digest(path)} for path in paths]}, indent=2))
        status("configure", source_sha=digest(source), external_weights_sha=digest(weights))
        check(context.config(**config), "config")
        # Keep external data beside source.onnx; the SDK resolves weights.bin relative to it.
        check(context.load_onnx(model=str(source)), "load_onnx")
        status("fresh_hybrid_step1")
        check(context.hybrid_quantization_step1(dataset=str(calibration), proposal=False), "step1")
        reference = yaml.safe_load((root / "artifacts/sdpa_safe_full_20261010/sensitive_except_sdpa/policy.cfg").read_text())
        fresh = yaml.safe_load((output / "source.quantization.cfg").read_text())
        policy = reapply_precision_policy(fresh, reference)
        if len(policy["custom_quantize_layers"]) != 126:
            raise RuntimeError("precision policy differs from KL120")
        policy_file = output / "policy.cfg"
        policy_file.write_text(yaml.safe_dump(policy, sort_keys=False))
        status("hybrid_step2", protected_tensor_count=126, sdpa_int8_count=24)
        check(context.hybrid_quantization_step2(model_input=str(output / "source.model"),
            data_input=str(output / "source.data"), model_quantization_cfg=str(policy_file)), "step2")
        artifact = output / "clip.rknn"
        check(context.export_rknn(str(artifact)), "export")
        status("simulator_calibration_sample", model_sha=digest(artifact), model_bytes=artifact.stat().st_size)
        check(context.init_runtime(), "simulator init")
        result = context.inference(inputs=[np.load(paths[0])], data_format=["nchw"])
        if result is None or len(result) != 1 or result[0].size != 768 or not np.isfinite(result[0]).all():
            raise RuntimeError("invalid simulator calibration output")
        np.save(output / "simulator_calibration.npy", result[0])
        if fingerprint(root) != original or digest(source) != affine["onnx_sha"] or digest(weights) != affine["external_weights_sha"]:
            raise RuntimeError("source changed during conversion")
        status("completed", completed=True, policy_sha=digest(policy_file))
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        context.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
