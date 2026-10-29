"""Supplement error measurement with the actual recovered project floating source."""
import json
from pathlib import Path
import numpy as np
from evaluate_real_200 import load
from numerical_fidelity import summarize, unit_normalize

root = Path(__file__).resolve().parents[1]
data = root / "results/real_eval_200_20261011"
selected = json.loads((data / "dataset/manifest.json").read_text(encoding="utf-8"))
names = [row["image"] for row in selected["samples"]]
original_dir = data / "recovered_project_fp32_verified"
status = json.loads((original_dir / "status.json").read_text())
if not status["completed"] or status["source_sha"] != "a51e3e827dcdcd733fdc20a94726a97828a1dd18521b6034ce9d07b58627fa6b":
    raise RuntimeError("complete verified original reference required")
original = load(original_dir, names)
deployed = load(data / "deployed_historical", names)
replacement = load(data / "onnx_fp32", names)
report = {"count": 200, "inputs": "exact same board-preprocessed FP16 values cast to FP32",
    "original_source_sha": status["source_sha"], "deployed_sha": "898069fbe1bf71a760f551f0b6bc36c889a5dbedcb15d427070b8923ba1be177",
    "original_fp32_to_actual_deployed": summarize(original, deployed),
    "original_fp32_to_downloaded_reference_fp32": summarize(original, replacement),
    "unit_contract_original_to_downloaded_fp32": summarize(unit_normalize(original), unit_normalize(replacement)),
    "unit_contract_original_to_deployed": summarize(unit_normalize(original), unit_normalize(deployed)),
    "output_norms": {name: {"mean": float(np.linalg.norm(values, axis=1).mean()),
        "min": float(np.linalg.norm(values, axis=1).min()), "max": float(np.linalg.norm(values, axis=1).max())}
        for name, values in [("original_fp32", original), ("downloaded_fp32", replacement), ("historical_deployed", deployed)]},
    "interpretation": "original and historical output unit embeddings, downloaded FP32 outputs raw embeddings; compare directions or explicitly match unit-output contracts. Raw cross-contract RMSE/SQNR are NOT quantization loss. Floating directions match on these 200 inputs, not proof of byte-identical weights. Historical directional error remains unexplained by this normalization difference."}
runtime230 = data / "historical_runtime230_probe"
if (runtime230 / "manifest.json").exists():
    old = load(runtime230, names)
    report["runtime230_to_runtime232_historical"] = summarize(old, deployed)
    report["original_fp32_to_runtime230_historical"] = summarize(original, old)
with (data / "recovered_original_fidelity_unit_contract_v2.json").open("x") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({name: {key: value for key, value in report[name].items() if key != "per_sample"}
    for name in ["original_fp32_to_actual_deployed", "unit_contract_original_to_downloaded_fp32",
        "unit_contract_original_to_deployed"]}, indent=2), flush=True)
