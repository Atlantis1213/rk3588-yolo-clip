"""New development/locked splits; preserve every previously evaluated image and input."""
import hashlib
import json
from pathlib import Path
import random
import zipfile
import cv2
import numpy as np

root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
output = artifacts / "method_sweep_20261011/protocol_v2"
if output.exists():
    raise RuntimeError("protocol exists; preserve its split rather than resampling")
original = json.loads((artifacts / "dataset_provenance.json").read_text())
validation = json.loads((artifacts / "eval_labels.json").read_text())
final = json.loads((artifacts / "final_test_labels.json").read_text())
used = set(original["calibration_sources"]) | {s["source"] for s in validation["samples"] + final["samples"]}
archive = root / "assets/dataset-resized.zip"
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
if archive_sha != original["sha256"]:
    raise RuntimeError("dataset archive differs from the recorded source")
generator = random.Random(20261011)
records = {name: [] for name in ["extra_calibration", "development", "locked_test"]}
with zipfile.ZipFile(archive) as dataset:
    available = {}
    for label in validation["classes"]:
        candidates = sorted(name for name in dataset.namelist()
            if name.startswith("dataset-resized/") and name.lower().endswith(".jpg")
            and not Path(name).name.startswith("._") and f"/{label}/" in name and name not in used)
        generator.shuffle(candidates)
        if len(candidates) < 68:
            raise RuntimeError("need 20 extra calibration + 24 development + 24 locked images: " + label)
        available[label] = candidates
    output.mkdir(parents=True, exist_ok=False)
    for split in records:
        (output / split).mkdir()
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    extra_paths = []
    for label, candidates in available.items():
        for split, sources in [("extra_calibration", candidates[:20]),
                               ("development", candidates[20:44]), ("locked_test", candidates[44:68])]:
            for index, source in enumerate(sources):
                image = cv2.imdecode(np.frombuffer(dataset.read(source), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cannot decode " + source)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
                identifier = f"v2_{label}_{index:03d}"
                if split == "extra_calibration":
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    value = ((rgb.astype(np.float32) / 255 - mean) / std).astype(np.float16).astype(np.float32)
                    path = output / split / (identifier + ".npy")
                    np.save(path, value.transpose(2, 0, 1)[None].copy())
                    extra_paths.append(str(path))
                else:
                    path = output / split / (identifier + ".png")
                    if not cv2.imwrite(str(path), image):
                        raise RuntimeError("cannot save " + str(path))
                records[split].append({"image": path.name, "class": label, "source": source,
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    previous_paths = [line for line in (artifacts / "calibration.txt").read_text().splitlines() if line]
    if len(previous_paths) != 120:
        raise RuntimeError("expected the unchanged 120-image calibration")
    (output / "calibration_augmented_240.txt").write_text("\n".join(previous_paths + extra_paths) + "\n")
    for split, rows in records.items():
        (output / (split + "_labels.json")).write_text(json.dumps({"classes": validation["classes"], "samples": rows}, indent=2))
    all_sources = [row["source"] for rows in records.values() for row in rows]
    if len(set(all_sources)) != len(all_sources) or set(all_sources) & used:
        raise RuntimeError("new split overlap")
    (output / "protocol.json").write_text(json.dumps({"seed": 20261011, "archive_sha256": archive_sha,
        "excluded_previous_sources": sorted(used), "counts": {name: len(rows) for name, rows in records.items()},
        "augmented_calibration_count": 240, "locked_test_sealed_until_strategy_freeze": True,
        "note": "no old calibration/validation/final source reused in new development or locked test; "
                "future QAT must exclude both new development and locked sources"}, indent=2))
print(json.dumps({"output": str(output), "counts": {name: len(rows) for name, rows in records.items()},
                  "augmented_calibration_count": 240}, indent=2))
