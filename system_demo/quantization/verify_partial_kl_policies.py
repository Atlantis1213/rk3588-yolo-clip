"""Verify every partial variant changes only eight baseline precision choices."""
import json
from pathlib import Path
import yaml
from partial_mlp_policy import PARTIAL_KL_DOWN, select_down_tensors
from record_ir_provenance import fingerprint

root = Path(__file__).resolve().parent
builds = root / "artifacts/sdpa_safe_full_20261010"
reference = yaml.safe_load((builds / "sensitive_kl_except_sdpa/policy.cfg").read_text())
reference_status = json.loads((builds / "sensitive_kl_except_sdpa/status.json").read_text())
provenance = json.loads((root / "artifacts/calibration_algorithms_20261010/kl_divergence/ir_provenance.json").read_text())
if fingerprint(root, algorithm="kl_divergence") != provenance:
    raise RuntimeError("source/calibration/SDK changed since KL step1")
layers = json.loads((root / "artifacts/diagnosis_20261010/layer_summary.json").read_text())["layers"]
records = []
for variant, blocks in PARTIAL_KL_DOWN.items():
    status = json.loads((builds / variant / "status.json").read_text())
    if not status.get("completed") or status["calibration_algorithm"] != "kl_divergence":
        raise RuntimeError("partial variant has not completed with KL: " + variant)
    candidate = yaml.safe_load((builds / variant / "policy.cfg").read_text())
    selected = select_down_tensors(layers, blocks)
    expected = dict(reference["custom_quantize_layers"], **{name: "float16" for name in selected})
    if candidate["quantize_parameters"] != reference["quantize_parameters"]:
        raise RuntimeError("KL quantization parameters changed: " + variant)
    if candidate["custom_quantize_layers"] != expected or len(expected) != 134:
        raise RuntimeError("changes are not exactly eight down-projection choices: " + variant)
    if status["source_sha"] != reference_status["source_sha"] or status["protected_matrix_blocks"] != list(blocks):
        raise RuntimeError("source or block metadata differs: " + variant)
    records.append({"variant": variant, "blocks": list(blocks), "same_kl_quantization_parameters": True,
        "only_eight_down_choices_added": True, "protected_tensor_count": 134})
destination = root / "artifacts/partial_kl_policy_equality_20261010.json"
if destination.exists():
    raise RuntimeError("existing policy equality evidence; do not overwrite")
destination.write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2), flush=True)
