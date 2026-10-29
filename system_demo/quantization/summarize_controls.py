"""Six frozen validation samples; summarize simulated precision controls without test leakage."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "artifacts"
diagnosis = root / "diagnosis_20261010"
reference = np.load(diagnosis / "torch.f32.npy").astype(np.float64)
names = [r["image"] for r in json.loads((diagnosis / "reference_comparison.json").read_text())["rows"]]
rows = []
for mode in ("all_fp16", "sensitive_gelu"):
    location = root / "precision_controls_20261010" / mode
    metadata = json.loads((location / "status.json").read_text())
    if not metadata["completed"]:
        raise RuntimeError(mode + " has not completed")
    values = np.load(location / "simulator.f32.npy").astype(np.float64)
    if values.shape != reference.shape or not np.isfinite(values).all():
        raise RuntimeError("invalid control outputs")
    cosine = (reference * values).sum(axis=1) / (np.linalg.norm(reference, axis=1) * np.linalg.norm(values, axis=1))
    row = {"mode": mode, "model_bytes": metadata["model_bytes"], "protected_count": metadata["selected_tensor_count"],
           "cosine_mean": float(cosine.mean()), "cosine_min": float(cosine.min()),
           "samples": [{"image": name, "cosine_to_torch": float(value)} for name, value in zip(names, cosine)]}
    rows.append(row)
(root / "precision_controls_20261010/summary_simulator.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
