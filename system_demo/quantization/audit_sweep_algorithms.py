"""Audit actual saved calibration/SDK metadata, not merely requested config labels."""
import json
from pathlib import Path
import pickle
from argparse import Namespace
import numpy as np
import yaml
from record_ir_provenance import digest

root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
profiles = ["normal30", "kl30", "mmse30"]
records = {}
data = {}
policies = {}
for profile in profiles:
    directory = sweep / profile
    status = json.loads((directory / "status.json").read_text())
    if not status["completed"]:
        raise RuntimeError("all three builds must complete first")
    provenance = json.loads((directory / "provenance.json").read_text())
    if digest(directory / "clip.rknn") != status["model_sha"]:
        raise RuntimeError("completed model hash changed")
    # Only load our own completed SDK artifacts, never arbitrary downloaded pickle input.
    with (directory / "clip_visual_reference_fp32.data").open("rb") as handle:
        data[profile] = pickle.load(handle)
    policies[profile] = yaml.safe_load((directory / "policy.cfg").read_text())
    def summary(value):
        if isinstance(value, dict):
            return {"type": "dict", "keys": list(map(str, value.keys()))[:60], "count": len(value)}
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "count": len(value)}
        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"type": type(value).__name__}
    records[profile] = {"requested_algorithm": provenance["algorithm"], "model_sha": status["model_sha"],
        "calibration_inputs": provenance["calibration_inputs"],
        "actual_saved_algorithm": data[profile]["quantizer.param"]["qalgorithm"],
        "ir_model_sha": digest(directory / "clip_visual_reference_fp32.model"),
        "ir_data_sha": digest(directory / "clip_visual_reference_fp32.data"),
        "policy_sha": digest(directory / "policy.cfg"),
        "metadata_top_level": {str(key): summary(value) for key, value in data[profile].items()}}


def differences(a, b, path=""):
    result = []
    if type(a) != type(b):
        return [path + ": type differs"]
    if isinstance(a, dict):
        if set(a) != set(b):
            result.append(path + ": keys differ")
        for key in a.keys() & b.keys():
            result += differences(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, Namespace):
        result += differences(vars(a), vars(b), path)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            result.append(path + ": length differs")
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                result += differences(x, y, path + "/" + str(index))
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
            result.append(path + ": array differs")
    elif isinstance(a, (str, int, float, bool)) or a is None:
        if a != b:
            result.append(path + ": " + repr(a)[:120] + " -> " + repr(b)[:120])
    else:
        result.append(path + ": opaque " + type(a).__name__ + " not compared")
    return result


pairs = {}
for first, second in [("normal30", "kl30"), ("kl30", "mmse30")]:
    changed = differences(data[first], data[second])
    pairs[first + "_vs_" + second] = {"policies_exactly_equal": policies[first] == policies[second],
        "same_30_calibration_inputs": records[first]["calibration_inputs"] == records[second]["calibration_inputs"],
        "quant_table_exactly_equal": not differences(data[first]["quant_tab"], data[second]["quant_tab"]),
        "metadata_diff_count": len(changed), "metadata_differences": changed[:100]}
report = {"variants": records, "pairs": pairs,
    "note": "identical editable activation cfg does not establish identical internal weight quantization or compiled execution"}
with (sweep / "algorithm_contract_audit_v2.json").open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2), flush=True)
