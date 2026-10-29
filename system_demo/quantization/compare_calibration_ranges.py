"""Explain changed calibration ranges; one golden trace is diagnostic, not acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", choices=["kl_divergence"], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
fresh_dir = root / "artifacts/calibration_algorithms_20261010" / args.algorithm
if not json.loads((fresh_dir / "status.json").read_text())["completed"]:
    raise RuntimeError("fresh calibration has not completed")
original = yaml.safe_load((root / "artifacts/manual_hybrid_20261010/clip_visual_reference_fp32.quantization.cfg").read_text())
fresh = yaml.safe_load((fresh_dir / "clip_visual_reference_fp32.quantization.cfg").read_text())
normal_parameters, new_parameters = original["quantize_parameters"], fresh["quantize_parameters"]
if set(normal_parameters) != set(new_parameters):
    raise RuntimeError("calibration tensor names differ")
changed = [name for name in normal_parameters if normal_parameters[name] != new_parameters[name]]
snapshot = root / "artifacts/diagnosis_20261010/layer_analysis/snapshot"
mapping = {}
for line in (snapshot / "map_name_to_file.txt").read_text().splitlines():
    if not line.strip() or line.startswith("layer_name") or set(line.strip()) == {"-"}:
        continue
    fields = line.split()
    if len(fields) != 2 or not fields[1].endswith(".txt") or fields[0] in mapping:
        raise RuntimeError("unexpected or duplicate SDK mapping row: " + line)
    mapping[fields[0]] = fields[1]


def qdq_metrics(values, parameters):
    if parameters["qmethod"] != "layer" or parameters["dtype"] != "int8":
        raise RuntimeError("expected signed per-tensor activation quantization")
    scale, zero = float(parameters["scale"][0]), int(parameters["zero_point"][0])
    if scale <= 0 or not np.isfinite(scale) or not -128 <= zero <= 127:
        raise RuntimeError("invalid quantization parameters")
    unbounded = np.rint(values / scale) + zero
    quantized = np.clip(unbounded, -128, 127)
    reconstructed = (quantized - zero) * scale
    error = reconstructed - values
    norm = np.linalg.norm(values) * np.linalg.norm(reconstructed)
    return {"scale": scale, "zero_point": zero,
        "calibration_min": float(parameters["min"][0]), "calibration_max": float(parameters["max"][0]),
        "fraction_qdq_zero": float(np.mean(quantized == zero)),
        "fraction_saturated": float(np.mean((unbounded < -128) | (unbounded > 127))),
        "qdq_rmse": float(np.sqrt(np.mean(error ** 2))),
        "qdq_cosine": float(values @ reconstructed / norm) if norm else None}


rows = []
for block in range(24):
    name = f"/transformer/resblocks/resblocks.{block}/mlp/gelu/Mul_1_output_0_tp_rs"
    if name not in mapping or name not in normal_parameters:
        raise RuntimeError("missing GELU mapping: " + name)
    filename = Path(mapping[name]).with_suffix(".npy").name
    values = np.load(snapshot / "golden/npy" / filename, allow_pickle=False).astype(np.float64).ravel()
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite golden activation")
    before = qdq_metrics(values, normal_parameters[name])
    after = qdq_metrics(values, new_parameters[name])
    rows.append({"block": block, "tensor": name, "elements": len(values),
        "normal": before, args.algorithm: after, "scale_ratio": after["scale"] / before["scale"]})

destination = root / "artifacts/calibration_kl_ranges_20261010.json"
if destination.exists():
    raise RuntimeError("range report exists; preserve previous evidence")
sample = json.loads((root / "artifacts/diagnosis_20261010/layer_summary.json").read_text())["sample"]
report = {"scope": "same 120-image calibration, one saved validation FP32 golden trace for QDQ explanation only; "
          "not full calibration distribution, not bit-exact NPU arithmetic, not task accuracy",
    "normal_algorithm": "normal", "candidate_algorithm": args.algorithm,
    "sample": sample, "parameter_tensor_count": len(normal_parameters),
    "changed_parameter_tensor_count": len(changed), "changed_parameter_tensors": changed,
    "gelu_rows": rows}
destination.write_text(json.dumps(report, indent=2))
print(json.dumps({"parameter_tensor_count": len(normal_parameters), "changed_count": len(changed),
    "diagnostic_sample": sample, "block12": rows[12]}, indent=2), flush=True)
