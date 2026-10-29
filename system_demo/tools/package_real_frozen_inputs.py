"""Small exact-input bundle so the recovered remote model need not be downloaded."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
base = root / "results/real_eval_200_20261011"
frozen = base / "frozen"
manifest = json.loads((frozen / "manifest.json").read_text())
hashes = json.loads((frozen / "input_sha256.json").read_text())
output = base / "exact_frozen_inputs.zip"
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
    bundle.write(frozen / "manifest.json", "manifest.json")
    bundle.write(frozen / "input_sha256.json", "input_sha256.json")
    for row in manifest["samples"]:
        name = row["image"]
        path = frozen / "inputs_fp16" / (name + ".bin")
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise RuntimeError("frozen input changed before packaging")
        bundle.write(path, "inputs_fp16/" + name + ".bin")
print(json.dumps({"bundle_bytes": output.stat().st_size,
    "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}), flush=True)
