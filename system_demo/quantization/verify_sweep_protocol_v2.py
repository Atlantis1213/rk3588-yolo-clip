"""Audit new split identities, files and exact resized-image duplicates."""
import hashlib
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np

root = Path(__file__).resolve().parent
protocol = root / "artifacts/method_sweep_20261011/protocol_v2"
record = json.loads((protocol / "protocol.json").read_text())
old_sources = set(record["excluded_previous_sources"])
new_sources = set()
owners = {}
duplicates = []
cv2.setNumThreads(1)
with zipfile.ZipFile(root / "assets/dataset-resized.zip") as archive:
    def pixel_sha(source):
        image = cv2.imdecode(np.frombuffer(archive.read(source), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("invalid source image")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        return hashlib.sha256(image.tobytes()).hexdigest()
    for source in sorted(old_sources):
        owners.setdefault(pixel_sha(source), ("previous", source))
    for split, expected_count in [("extra_calibration", 120), ("development", 144), ("locked_test", 144)]:
        labels = json.loads((protocol / (split + "_labels.json")).read_text())
        if len(labels["samples"]) != expected_count:
            raise RuntimeError("split count differs")
        for row in labels["samples"]:
            if row["source"] in old_sources or row["source"] in new_sources:
                raise RuntimeError("source split overlap")
            new_sources.add(row["source"])
            file = protocol / split / row["image"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != row["file_sha256"]:
                raise RuntimeError("generated split file changed")
            digest = pixel_sha(row["source"])
            if digest in owners:
                duplicates.append({"split": split, "source": row["source"],
                    "previous_split": owners[digest][0], "previous_source": owners[digest][1]})
            else:
                owners[digest] = (split, row["source"])
result = {"source_split_disjoint": True, "generated_file_hashes_valid": True,
    "new_sources": len(new_sources), "exact_resized_pixel_duplicate_count": len(duplicates),
    "duplicates": duplicates, "scope": "exact 224x224 pixel hashes, not near-duplicate or semantic deduplication"}
destination = protocol / "verification.json"
if destination.exists():
    raise RuntimeError("verification exists; preserve evidence")
destination.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != "duplicates"}, indent=2))
if duplicates:
    raise RuntimeError("new protocol contains exact pixel duplicates; repair before using it for selection")
