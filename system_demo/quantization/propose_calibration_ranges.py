"""Use previously frozen calibration histograms, never real evaluation activations."""
import json
from pathlib import Path

import numpy as np
import yaml

from histogram_range_search import affine_parameters, histogram_error, percentile_upper, mse_search
from record_ir_provenance import digest, fingerprint

root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
source = sweep / "calibration_stats120"
output = sweep / "histogram_ranges_20261011"
output.mkdir(exist_ok=False)
previous = json.loads((root / "artifacts/manual_hybrid_20261010/ir_provenance.json").read_text())
if fingerprint(root) != previous or not json.loads((source / "status.json").read_text())["completed"]:
    raise RuntimeError("completed unmodified calibration statistics required")
summary = json.loads((source / "statistics.json").read_text())
modules = {row["module"]: row for row in summary["modules"]}
policy = root / "artifacts/sdpa_safe_full_20261010/sensitive_kl_except_sdpa/policy.cfg"
parameters = yaml.safe_load(policy.read_text())["quantize_parameters"]
rows = []
with np.load(source / "channel_statistics.npz", allow_pickle=False) as arrays:
    for index in range(24):
        key = f"block{index}.gelu"
        metadata = modules[key]
        counts, bounds = arrays[key + ".histogram"], arrays[key + ".histogram_bounds"]
        if metadata["out_of_histogram"] or int(counts.sum()) != metadata["rows"] * metadata["width"]:
            raise RuntimeError("histogram missing activation mass: " + key)
        tensor = f"/transformer/resblocks/resblocks.{index}/mlp/gelu/Mul_1_output_0_tp_rs"
        original = parameters[tensor]
        baseline = {name: original[name][0] for name in ["min", "max", "scale", "zero_point"]}
        percentile = affine_parameters(metadata["min"], percentile_upper(counts, bounds, .9999))
        rows.append({"block": index, "tensor": tensor, "histogram_bin_count": len(counts),
            "kl_baseline": dict(baseline, **histogram_error(counts, bounds, baseline)),
            "percentile_9999": dict(percentile, **histogram_error(counts, bounds, percentile)),
            "mse_search": mse_search(counts, bounds, metadata["min"], metadata["max"], baseline)})
report = {"completed": True, "calibration_count": 120, "rows": rows,
    "statistics_sha": digest(source / "channel_statistics.npz"), "reference_policy_sha": digest(policy),
    "source_fingerprint": previous, "scope": "float-reference aggregate GELU histograms only; no validation/test activation fitting",
    "backend_parameters_used_verified": False, "board_evaluated": False,
    "approximation": "bin-center QDQ uses nearest-even affine rounding; not bit-exact fused INT8 SDPA/NPU arithmetic"}
(output / "proposals.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"blocks": len(rows), "baseline_mean_hist_mse": float(np.mean([r['kl_baseline']['histogram_qdq_mse'] for r in rows])),
    "percentile_mean_hist_mse": float(np.mean([r['percentile_9999']['histogram_qdq_mse'] for r in rows])),
    "mse_search_mean_hist_mse": float(np.mean([r['mse_search']['histogram_qdq_mse'] for r in rows]))}), flush=True)
