"""Validate the fixed, locally prepared image bundle without arbitrary ZIP paths."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
output = root / "results/real_eval_200_20261011/dataset"
output.mkdir(parents=True, exist_ok=False)
archive = root / "results/real_eval_200_20261011/real_eval_200.zip"
summary = json.loads((root / "results/real_eval_200_20261011/selection_summary.json").read_text(encoding="utf-8"))
if hashlib.sha256(archive.read_bytes()).hexdigest() != summary["bundle_sha256"]:
    raise RuntimeError("selected image bundle SHA differs")
with zipfile.ZipFile(archive) as bundle:
    manifest = json.loads(bundle.read("manifest.json"))
    rows = manifest["samples"]
    names = {"manifest.json", "samples.tsv"} | {"images/" + row["image"] for row in rows}
    if manifest["count"] != 200 or len(rows) != 200 or len(names) != 202 or set(bundle.namelist()) != names:
        raise RuntimeError("unexpected bundle sample membership")
    if len(bundle.namelist()) != 202 or any(Path(row["image"]).name != row["image"] for row in rows):
        raise RuntimeError("duplicate or unsafe archive path")
    (output / "images").mkdir()
    for row in rows:
        payload = bundle.read("images/" + row["image"])
        if hashlib.sha256(payload).hexdigest() != row["file_sha256"]:
            raise RuntimeError("image SHA differs")
        with (output / "images" / row["image"]).open("xb") as handle:
            handle.write(payload)
    for name in ["manifest.json", "samples.tsv"]:
        with (output / name).open("xb") as handle:
            handle.write(bundle.read(name))
print("Verified/extracted 200 real images; original photos are unchanged", flush=True)
