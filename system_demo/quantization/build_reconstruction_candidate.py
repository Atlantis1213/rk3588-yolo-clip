"""Fixed-KL-grid candidate: change only existing MLP-down weights/biases in a copied IR."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import onnx
from onnx import numpy_helper
from rknn.api import RKNN
from record_ir_provenance import digest, fingerprint
from sdk_ir_validation import snapshot_ir, validate_ir

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=["bias_down", "adaround_down", "adaround_bias_down"], required=True)
parser.add_argument("--attempt", required=True, help="new single-directory experiment name; preserve prior attempts")
args = parser.parse_args()
if not args.attempt.replace("_", "").replace("-", "").isalnum():
    parser.error("attempt must contain only letters, digits, underscore or hyphen")
root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
output = sweep / "reconstruction_rknn" / args.attempt / args.profile
output.mkdir(parents=True, exist_ok=False)
os.chdir(output)
started = time.monotonic()
info = {"profile": args.profile, "completed": False, "pid": os.getpid(), "calibration_count": 120,
    "extra_inference_nodes": 0, "scope": "fixed original KL120 grid and existing 24 MLP down-projections only"}


def status(phase, **values):
    info.update(phase=phase, elapsed_seconds=time.monotonic() - started, **values)
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, output / "status.json")
    print(args.profile + " " + phase, flush=True)


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation}: {code}")


context = None
try:
    status("verify")
    original = root / "artifacts/calibration_algorithms_20261010/kl_divergence"
    provenance = json.loads((original / "ir_provenance.json").read_text())
    if fingerprint(root, "kl_divergence") != provenance:
        raise RuntimeError("original source/calibration/SDK fingerprint differs")
    build = json.loads((original / "status.json").read_text())
    if not build["completed"]:
        raise RuntimeError("original KL120 step1 incomplete")
    source_files = {suffix: original / ("clip_visual_reference_fp32." + suffix)
        for suffix in ["model", "data", "quantization.cfg"]}
    for suffix, path in source_files.items():
        if digest(path) != build["step1_files"][suffix]["sha256"]:
            raise RuntimeError("original SDK artifact hash differs")
    training = sweep / "reconstruction_gpu/verified"
    verified = json.loads((sweep / "reconstruction_gpu/verification.json").read_text())
    results = json.loads((training / "status.json").read_text())
    contract = json.loads((sweep / "reconstruction_contract_kl120.json").read_text())
    if not verified["completed"] or not results["completed"] or contract["sdk_data_sha"] != digest(source_files["data"]):
        raise RuntimeError("verified completed reconstruction on this SDK grid required")
    model = onnx.load(source_files["model"])
    baseline = snapshot_ir(model)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    signature = [(node.name, node.op_type, tuple(node.input), tuple(node.output)) for node in model.graph.node]
    changes = []
    allowed = set()
    for index, record in enumerate(contract["blocks"]):
        path = training / f"block{index:02d}.npz"
        if digest(path) != results["blocks"][index]["artifact_sha"]:
            raise RuntimeError("reconstructed block artifact differs")
        convs = [node for node in model.graph.node if node.op_type == "Conv" and len(node.input) >= 3
            and node.input[1] == record["source_weight"]]
        if len(convs) != 1:
            raise RuntimeError("down-projection must map to one existing IR Conv")
        conv = convs[0]
        weight = numpy_helper.to_array(tensors[conv.input[1]])
        bias = numpy_helper.to_array(tensors[conv.input[2]])
        if weight.shape != (1024, 4096, 1, 1) or bias.shape != (1024,):
            raise RuntimeError("IR matrix orientation/shape differs")
        with np.load(path, allow_pickle=False) as values:
            if not np.array_equal(bias, values["original_bias"]):
                raise RuntimeError("original Torch/SDK down-projection bias differs")
            if args.profile != "bias_down":
                # Fixed scales from SDK .data remain intact, avoiding a fresh min/max rescaling here.
                new_weight = values["weight"].reshape(weight.shape).astype(weight.dtype)
                tensors[conv.input[1]].CopyFrom(numpy_helper.from_array(new_weight, name=conv.input[1]))
                allowed.add(conv.input[1])
            if args.profile != "adaround_down":
                new_bias = values["nearest_corrected_bias" if args.profile == "bias_down" else "bias"].astype(bias.dtype)
                tensors[conv.input[2]].CopyFrom(numpy_helper.from_array(new_bias, name=conv.input[2]))
                allowed.add(conv.input[2])
        changes.append({"block": index, "weight_changed": args.profile != "bias_down",
            "bias_changed": args.profile != "adaround_down", "weight_name": conv.input[1], "bias_name": conv.input[2]})
    if signature != [(node.name, node.op_type, tuple(node.input), tuple(node.output)) for node in model.graph.node]:
        raise RuntimeError("operator/input/output signature changed")
    validation = validate_ir(baseline, model, allowed)
    if not validation["changed_initializers"]:
        raise RuntimeError("reconstruction did not change any allowed parameter")
    destination = output / "reconstructed.model"
    onnx.save_model(model, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
        location="weights.bin", size_threshold=1024, convert_attribute=False)
    del model, tensors, weight, bias
    gc.collect()
    status("validate_saved_vendor_ir")
    saved_model = onnx.load(str(destination))
    saved_validation = validate_ir(baseline, saved_model, allowed)
    if saved_validation != validation:
        raise RuntimeError("saved external-data IR does not match validated in-memory candidate")
    (output / "ir_validation.json").write_text(json.dumps(saved_validation, indent=2))
    del saved_model
    gc.collect()
    data_file = output / "reconstructed.data"
    shutil.copyfile(source_files["data"], data_file)
    policy = root / "artifacts/sdpa_safe_full_20261010/sensitive_kl_except_sdpa/policy.cfg"
    policy_file = output / "policy.cfg"
    shutil.copyfile(policy, policy_file)
    (output / "provenance.json").write_text(json.dumps({"original_fingerprint": provenance,
        "changes": changes, "reconstruction_contract_sha": digest(sweep / "reconstruction_contract_kl120.json"),
        "training_provenance": json.loads((training / "provenance.json").read_text()),
        "copied_sdk_data_unchanged": digest(data_file) == digest(source_files["data"]),
        "copied_activation_precision_policy_unchanged": digest(policy_file) == digest(policy)}, indent=2))
    status("hybrid_step2", source_sha=digest(destination), external_weights_sha=digest(output / "weights.bin"),
        protected_tensor_count=126, sdpa_int8_count=24)
    context = RKNN(verbose=False)
    check(context.hybrid_quantization_step2(model_input=str(destination), data_input=str(data_file),
        model_quantization_cfg=str(policy_file)), "step2")
    artifact = output / "clip.rknn"
    check(context.export_rknn(str(artifact)), "export")
    status("simulator_calibration_sample", model_sha=digest(artifact), model_bytes=artifact.stat().st_size)
    check(context.init_runtime(), "simulator init")
    first = Path((root / "artifacts/calibration.txt").read_text().splitlines()[0])
    result = context.inference(inputs=[np.load(first)], data_format=["nchw"])
    if result is None or len(result) != 1 or result[0].size != 768 or not np.isfinite(result[0]).all():
        raise RuntimeError("invalid simulator calibration output")
    np.save(output / "simulator_calibration.npy", result[0])
    if fingerprint(root, "kl_divergence") != provenance:
        raise RuntimeError("original source changed during reconstruction build")
    status("completed", completed=True, policy_sha=digest(policy_file))
except Exception as error:
    status("failed", error=repr(error))
    raise
finally:
    if context is not None:
        context.release()
