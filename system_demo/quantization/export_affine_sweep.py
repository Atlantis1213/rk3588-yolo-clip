"""Fold measured LN channel ranges into existing ONNX affine parameters, then verify."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time
import numpy as np
import onnx
from onnx import numpy_helper
from affine_reparameterization import smooth_fold, affine_fold, channel_parameters
from record_ir_provenance import digest

PROFILES = {"smooth_ln2_a25": ("smooth", .25, ["ln2"]),
    "smooth_ln2_a50": ("smooth", .5, ["ln2"]), "smooth_ln2_a75": ("smooth", .75, ["ln2"]),
    "smooth_all_a50": ("smooth", .5, ["ln1", "ln2"]),
    "repq_ln2": ("affine", None, ["ln2"]), "repq_all": ("affine", None, ["ln1", "ln2"])}
parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=list(PROFILES), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / "artifacts/method_sweep_20261011/affine_sources" / args.profile
output.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
metadata = {"profile": args.profile, "completed": False, "phase": "transform", "extra_inference_nodes": 0}


def status(phase, **details):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print("AFFINE " + args.profile + " " + phase, flush=True)


def transform():
    source = root / "artifacts/clip_visual_reference_fp32.onnx"
    stats_root = root / "artifacts/method_sweep_20261011/calibration_stats120"
    if not json.loads((stats_root / "status.json").read_text())["completed"]:
        raise RuntimeError("full calibration statistics required")
    stats = np.load(stats_root / "channel_statistics.npz", allow_pickle=False)
    method, alpha, parts = PROFILES[args.profile]
    model = onnx.load(str(source))
    nodes = {node.name: node for node in model.graph.node}
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    consumers = Counter(name for node in model.graph.node for name in node.input)
    producers = {name: node for node in model.graph.node for name in node.output}
    signature = [(node.name, node.op_type, tuple(node.output)) for node in model.graph.node]
    def tensor(name):
        if name in tensors:
            return numpy_helper.to_array(tensors[name]).copy()
        if name in producers and producers[name].op_type == "Identity":
            return tensor(producers[name].input[0])
        raise RuntimeError("affine parameter is not an initializer/identity: " + name)
    def assign(node, index, value, label):
        old_name = node.input[index]
        if old_name in tensors and consumers[old_name] == 1:
            tensors[old_name].CopyFrom(numpy_helper.from_array(value, name=old_name))
        else:
            new_name = "sweep_" + args.profile + "_" + label
            if new_name in tensors:
                raise RuntimeError("duplicate folded initializer")
            new_tensor = numpy_helper.from_array(value, name=new_name)
            model.graph.initializer.append(new_tensor)
            tensors[new_name] = model.graph.initializer[-1]
            node.input[index] = new_name
    records = []
    for index in range(24):
        prefix = f"/transformer/resblocks/resblocks.{index}"
        for part in parts:
            suffix = "ln_1" if part == "ln1" else "ln_2"
            linear = "/attn" if part == "ln1" else "/mlp/c_fc"
            norm = nodes[prefix + "/" + suffix + "/LayerNormalization"]
            matmul = nodes[prefix + linear + "/MatMul"]
            addition = nodes[prefix + linear + "/Add"]
            bias_index = next(i for i, name in enumerate(addition.input) if name != matmul.output[0])
            gamma, beta, weight, bias = tensor(norm.input[1]), tensor(norm.input[2]), tensor(matmul.input[1]), tensor(addition.input[bias_index])
            key = f"block{index}.{part}"
            low, high = stats[key + ".min"], stats[key + ".max"]
            if method == "smooth":
                maxima = np.maximum(np.abs(low), np.abs(high)).clip(1e-5)
                weight_maxima = stats[key + ".weight_input_absmax"].clip(1e-5)
                scales = np.clip(maxima ** alpha / weight_maxima ** (1 - alpha), .01, 100)
                result = smooth_fold(gamma, beta, weight, bias, scales)
                records.append({"block": index, "part": part, "scale_min": float(scales.min()), "scale_max": float(scales.max())})
            else:
                scales, zero = channel_parameters(low, high)
                common_scale, common_zero = float(scales.mean()), int(np.rint(zero.mean()))
                result = affine_fold(gamma, beta, weight, bias, scales, zero, common_scale, common_zero)
                records.append({"block": index, "part": part, "common_scale": common_scale,
                    "common_zero": common_zero, "ratio_min": float((scales / common_scale).min()),
                    "ratio_max": float((scales / common_scale).max())})
            if not all(np.isfinite(value).all() and np.abs(value).max() < 65504 for value in result):
                raise RuntimeError("folded parameter is outside finite FP16 range")
            for node, slot, value, label in [(norm, 1, result[0], key + "_gamma"), (norm, 2, result[1], key + "_beta"),
                (matmul, 1, result[2], key + "_weight"), (addition, bias_index, result[3], key + "_bias")]:
                assign(node, slot, value, label)
    if signature != [(node.name, node.op_type, tuple(node.output)) for node in model.graph.node]:
        raise RuntimeError("inference operator signature changed")
    destination = output / "source.onnx"
    onnx.save_model(model, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
        location="weights.bin", size_threshold=1024, convert_attribute=False)
    onnx.checker.check_model(str(destination))
    return {"method": method, "alpha": alpha, "parts": parts, "folded_modules": len(records), "records": records,
        "original_source_sha": digest(source), "onnx_sha": digest(destination), "external_weights_sha": digest(output / "weights.bin"),
        "operator_signature_unchanged": True,
        "logical_parameter_count": json.loads((stats_root / "statistics.json").read_text())["model_parameter_count"]}


try:
    status("transform")
    details = transform()
    gc.collect()
    status("verify_float_equivalence", **details)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / "source.onnx"), options, providers=["CPUExecutionProvider"])
    reference = np.load(root / "artifacts/method_sweep_20261011/calibration_stats120/teacher_embeddings.npy")
    paths = [Path(line) for line in (root / "artifacts/calibration.txt").read_text().splitlines() if line]
    rows = []
    for index in [0, 20, 40, 60, 80, 100]:
        actual = session.run(None, {"pixel_values": np.load(paths[index])})[0].reshape(-1).astype(np.float64)
        expected = reference[index].astype(np.float64)
        cosine = float(actual @ expected / (np.linalg.norm(actual) * np.linalg.norm(expected)))
        maximum = float(np.max(np.abs(actual - expected)))
        if cosine < .99999 or maximum > .001:
            raise RuntimeError("float equivalence gate failed")
        rows.append({"calibration_input": paths[index].name, "cosine": cosine, "max_abs_error": maximum})
    status("completed", completed=True, float_equivalence=rows,
        note="LN affine reparameterization adaptation only; not full paper Softmax/log quantization implementation")
except Exception as error:
    status("failed", error=repr(error))
    raise
