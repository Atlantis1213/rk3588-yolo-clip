"""Verify/extract only the completed GPU pilot's fixed list of artifact files."""
import json
from pathlib import Path
import zipfile
import numpy as np
from record_ir_provenance import digest

root = Path(__file__).resolve().parent
sweep = root / "artifacts/method_sweep_20261011"
directory = sweep / "reconstruction_gpu"
archive = directory / "reconstructed_weights.zip"
if digest(archive) != "962dd38a93f15c1b1730b85187b6b11efed341318dd63d2c08212f6fbd1315b4":
    raise RuntimeError("remote archive SHA mismatch")
output = directory / "verified"
output.mkdir(exist_ok=False)
expected = {"status.json", "provenance.json"} | {f"block{index:02d}.npz" for index in range(24)}
with zipfile.ZipFile(archive) as bundle:
    if set(bundle.namelist()) != expected or len(bundle.namelist()) != len(expected):
        raise RuntimeError("unexpected or duplicate archive entries")
    for name in sorted(expected):
        with (output / name).open("xb") as handle:
            handle.write(bundle.read(name))
status = json.loads((output / "status.json").read_text())
contract = json.loads((sweep / "reconstruction_contract_kl120.json").read_text())
if not status["completed"] or len(status["blocks"]) != 24:
    raise RuntimeError("GPU pilot incomplete")
grid_errors = []
for index, (record, scales) in enumerate(zip(status["blocks"], contract["blocks"])):
    artifact = output / f"block{index:02d}.npz"
    if record["block"] != index or digest(artifact) != record["artifact_sha"]:
        raise RuntimeError("block identity/SHA differs")
    with np.load(artifact, allow_pickle=False) as values:
        if set(values.files) != {"weight", "bias", "nearest_corrected_bias", "original_bias"}:
            raise RuntimeError("artifact key set differs")
        if values["weight"].shape != (1024, 4096) or any(values[name].shape != (1024,)
            for name in ["bias", "nearest_corrected_bias", "original_bias"]):
            raise RuntimeError("matrix shape differs")
        if not all(np.isfinite(values[name]).all() for name in values.files):
            raise RuntimeError("nonfinite reconstructed parameter")
        scale = np.asarray(scales["weight"]["scale"], np.float32)[:, None]
        zero = np.asarray(scales["weight"]["zero_point"], np.float32)[:, None]
        grid = values["weight"] / scale + zero
        error = float(np.max(np.abs(grid - np.rint(grid))))
        if error > 1e-4 or grid.min() < -128.0001 or grid.max() > 127.0001:
            raise RuntimeError("learned weights are not on fixed signed-INT8 channel grid")
        grid_errors.append(error)
report = {"completed": True, "archive_sha": digest(archive), "blocks": 24,
    "max_grid_integer_error": max(grid_errors), "all_block_hashes_valid": True,
    "note": "fixed SDK weight grid verified; SDK re-quantization and full board accuracy still require acceptance"}
(directory / "verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
