"""Contact sheet for inspecting scene geometry before fixing a non-YOLO crop rule."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
data = root / "results/real_eval_200_20261011/dataset"
output = root / "results/real_eval_200_20261011/nonyolo_source_preview.jpg"
if output.exists():
    raise RuntimeError("preserve previous preview")
manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
rows = [row for category in manifest["classes"] for row in manifest["samples"] if row["class"] == category][:0]
for category in manifest["classes"]:
    rows += [row for row in manifest["samples"] if row["class"] == category][:5]
canvas = np.full((5 * 204, 5 * 320, 3), 245, np.uint8)
for i, row in enumerate(rows):
    image = cv2.imread(str(data / "images" / row["image"]))
    if image is None:
        raise RuntimeError("missing selected image")
    thumbnail = cv2.resize(image, (320, 180))
    x, y = i % 5 * 320, i // 5 * 204
    canvas[y:y + 180, x:x + 320] = thumbnail
    cv2.putText(canvas, row["image"], (x + 3, y + 197), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
cv2.imwrite(str(output), canvas)
print(output)
