"""Create an untouched third split; do not rewrite active calibration files."""
import json
from pathlib import Path
import random
import zipfile
import cv2
import numpy as np

root = Path(__file__).resolve().parent / "artifacts"
existing = json.loads((root / "dataset_provenance.json").read_text())
validation = json.loads((root / "eval_labels.json").read_text())
used = set(existing["calibration_sources"]) | {s["source"] for s in validation["samples"]}
destination = root / "final_test_png"; destination.mkdir(exist_ok=True)
randomizer = random.Random(20261009)
samples = []
with zipfile.ZipFile(root.parent / "assets/dataset-resized.zip") as archive:
    for label in validation["classes"]:
        names = sorted(n for n in archive.namelist() if n.endswith(".jpg") and f"/{label}/" in n)
        randomizer.shuffle(names)
        for index, source in enumerate(names[40:60], start=40):
            if source in used: raise RuntimeError("final-test overlap")
            image = cv2.imdecode(np.frombuffer(archive.read(source), np.uint8), cv2.IMREAD_COLOR)
            filename = f"{label}_{index:03d}.png"
            cv2.imwrite(str(destination / filename), cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR))
            samples.append({"image": filename, "class": label, "source": source})
(root / "final_test_labels.json").write_text(json.dumps({"classes": validation["classes"], "samples": samples,
    "note": "third file-disjoint split, not used to choose scales or precision policy"}, indent=2))
print(f"Prepared {len(samples)} independent final-test samples", flush=True)
