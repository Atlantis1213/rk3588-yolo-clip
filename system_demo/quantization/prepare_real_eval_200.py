"""Select a frozen, balanced real-scene evaluation set; original images are untouched."""
import hashlib
import json
from pathlib import Path
import random
import zipfile
import cv2
import numpy as np

root = Path(__file__).resolve().parent
source = Path(r"H:\Race\25工创\wastesorting_data\2025_7_25")
output = root / "real_eval_200_20261011"
output.mkdir(exist_ok=False)
(output / "images").mkdir()
categories = {"cycle": "可回收物", "harmful": "有害垃圾", "kitchen": "厨余垃圾",
    "others": "其他垃圾", "empty": "unknown"}
generator = random.Random(20261011)
rows, inventory = [], {}
file_hashes, pixel_hashes = set(), set()
for category in categories:
    candidates = sorted(path for path in (source / category).rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    inventory[category] = len(candidates)
    generator.shuffle(candidates)
    chosen = 0
    for path in candidates:
        payload = path.read_bytes()
        file_sha = hashlib.sha256(payload).hexdigest()
        if file_sha in file_hashes:
            continue
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("undecodable image: " + str(path))
        pixel_sha = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if pixel_sha in pixel_hashes:
            continue
        name = f"real_{category}_{chosen:03d}" + path.suffix.lower()
        destination = output / "images" / name
        with destination.open("xb") as handle:
            handle.write(payload)
        rows.append({"image": name, "class": category, "mapped_category": categories[category],
            "source": str(path.relative_to(source)).replace("\\", "/"), "file_sha256": file_sha,
            "original_pixels_sha256": pixel_sha, "width": image.shape[1], "height": image.shape[0],
            "skip_yolo": category == "empty"})
        file_hashes.add(file_sha)
        pixel_hashes.add(pixel_sha)
        chosen += 1
        if chosen == 40:
            break
    if chosen != 40:
        raise RuntimeError("fewer than 40 decodable unique images: " + category)
manifest = {"source_root": str(source), "seed": 20261011, "count": 200,
    "source_inventory": inventory, "classes": list(categories), "per_class_count": 40,
    "samples": rows, "selected_images_not_for_calibration_or_training": True,
    "empty_policy": "ground-truth empty bypasses YOLO only; CLIP still predicts from full image; never directly assign empty",
    "sampling_note": "balanced classes, exact-file/pixel deduplication; no claim of instance/near-duplicate independence"}
(output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
with (output / "samples.tsv").open("x", encoding="utf-8", newline="\n") as handle:
    for row in rows:
        handle.write(row["image"] + "\t" + row["class"] + "\n")
bundle = output / "real_eval_200.zip"
with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for name in ["manifest.json", "samples.tsv"]:
        archive.write(output / name, name)
    for row in rows:
        archive.write(output / "images" / row["image"], "images/" + row["image"])
summary = {"count": len(rows), "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    "bundle_bytes": bundle.stat().st_size, "source_inventory": inventory, "source_root": str(source)}
(output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
