"""Compare diagnostic host/native submission with prior board and FP32 references."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "artifacts"
diagnosis = root / "diagnosis_20261010"
probe = diagnosis / "quant_io_probe_20261010"
metadata = json.loads((probe / "manifest.json").read_text())
reference = np.load(diagnosis / "torch.f32.npy")
names = [r["image"] for r in json.loads((diagnosis / "reference_comparison.json").read_text())["rows"]]
existing = root / "quant_public_int8_normal_ieee"
prior_manifest = json.loads((existing / "manifest.json").read_text())
prior_embeddings = np.fromfile(existing / "embeddings.f32", np.float32).reshape(-1, 768)
prior = {s["image"]: e for s, e in zip(prior_manifest["samples"], prior_embeddings)}


def metric(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("nonfinite probe output")
    return {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "max_abs_error": float(np.abs(a-b).max())}


rows = []
for name, golden in zip(names, reference):
    selected = [r for r in metadata["rows"] if r["image"] == name + ".bin"]
    modes = {r["mode"]: np.fromfile(probe / r["file"], np.float32) for r in selected}
    if set(modes) != {"fp16_host", "fp32_host", "int8_host", "int8_passthrough"}:
        raise RuntimeError("incomplete probe modes")
    rows.append({"image": name, "float16_to_float32": metric(modes["fp16_host"], modes["fp32_host"]),
                 "host_float16_to_prior": metric(modes["fp16_host"], prior[name]),
                 "manual_int8_host_to_passthrough": metric(modes["int8_host"], modes["int8_passthrough"]),
                 "manual_int8_to_host_float16": metric(modes["int8_host"], modes["fp16_host"]),
                 "torch_to_manual_int8": metric(golden, modes["int8_host"]),
                 "max_output_dequant_error": max(r["output_dequant_max_abs_error"] for r in selected)})
report = {"input_scale": metadata["input_scale"], "input_zp": metadata["input_zp"],
          "output_scale": metadata["output_scale"], "output_zp": metadata["output_zp"], "rows": rows}
(diagnosis / "io_contract_comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
