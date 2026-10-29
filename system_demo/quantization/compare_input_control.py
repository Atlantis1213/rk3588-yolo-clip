"""Only input boundary is quantized; keep the full deployed reference graph FP16."""
import json
from pathlib import Path
import numpy as np

diagnosis = Path(__file__).resolve().parent / "artifacts/diagnosis_20261010"
folder = diagnosis / "quant_input_control_20261010"
manifest = json.loads((folder / "manifest.json").read_text())
rows = []
for name in sorted({r["image"] for r in manifest["rows"]}):
    selected = {r["mode"]: r for r in manifest["rows"] if r["image"] == name}
    original = np.fromfile(folder / selected["original"]["file"], np.float32).astype(np.float64)
    reconstructed = np.fromfile(folder / selected["input_qdq"]["file"], np.float32).astype(np.float64)
    rows.append({"image": name, "input_qdq_rmse": selected["input_qdq"]["input_qdq_rmse"],
                 "fp16_original_to_input_qdq_cosine": float(original @ reconstructed / (np.linalg.norm(original) * np.linalg.norm(reconstructed))),
                 "embedding_rmse": float(np.sqrt(np.mean((original-reconstructed)**2)))})
report = {"note": "FP16 full graph; only input is quantized/dequantized with normal INT8 model scale/zp", "rows": rows}
(diagnosis / "input_qdq_control.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
