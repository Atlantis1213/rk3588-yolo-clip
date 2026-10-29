"""Freeze traditional-vision crops before viewing any CLIP scores; preserve every sample."""
import hashlib
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from green_foreground_crop import foreground_crop

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["original", "saturated_green"], default="original")
parser.add_argument("--review", type=Path)
parser.add_argument("--conservative-review", action="store_true", help="use one wide fixed central ROI for all reviewed failures")
args = parser.parse_args()
if args.conservative_review and args.review is None:
    parser.error("conservative review requires an explicit visual-review manifest")
source = root / "results/real_eval_200_20261011/dataset"
output = root / ("results/real_eval_200_20261011/nonyolo_reviewed_crop_v2" if args.conservative_review else
    "results/real_eval_200_20261011/nonyolo_reviewed_crop" if args.review else
    "results/real_eval_200_20261011/nonyolo_green_crop" if args.mode == "original"
    else "results/real_eval_200_20261011/nonyolo_saturated_green_crop")
output.mkdir(exist_ok=False)
(output / "dataset").mkdir()
(output / "crops").mkdir()
(output / "contact_sheets").mkdir()
selected = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
review = json.loads(args.review.read_text(encoding="utf-8")) if args.review else None
overrides = review["boxes"] if review else {}
if review and not set(overrides) <= {row["image"] for row in selected["samples"] if row["class"] != "empty"}:
    raise ValueError("review boxes must refer to existing nonempty samples only")
if len(selected["samples"]) != 200:
    raise ValueError("preserve original 200-image membership")
(output / "dataset/manifest.json").write_bytes((source / "manifest.json").read_bytes())
(output / "dataset/samples.tsv").write_bytes((source / "samples.tsv").read_bytes())
for name in ["board_labels_vitl14.tsv", "board_text_features_vitl14.f32"]:
    (output / name).write_bytes((source.parent / name).read_bytes())
rows = []
sheet = None
for index, sample in enumerate(selected["samples"]):
    payload = (source / "images" / sample["image"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sample["file_sha256"]:
        raise RuntimeError("original image SHA changed")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    box, status = foreground_crop(image, empty=sample["class"] == "empty", mode=args.mode)
    if sample["image"] in overrides:
        canvas_width, canvas_height = review["coordinate_canvas"]
        x0, y0, x1, y1 = overrides[sample["image"]]
        if args.conservative_review:
            # Single geometry-only safeguard for visually incomplete/ambiguous
            # segmentation, not a per-class or classifier-score-derived crop.
            x0, y0, x1, y1 = [75, 5, 245, 175]
        if not (0 <= x0 < x1 <= canvas_width and 0 <= y0 < y1 <= canvas_height):
            raise ValueError("review box is outside source preview")
        x0, x1 = [int(round(value * image.shape[1] / canvas_width)) for value in [x0, x1]]
        y0, y1 = [int(round(value * image.shape[0] / canvas_height)) for value in [y0, y1]]
        box = [x0, y0, x1 - x0, y1 - y0]
        status = {"method": "visual_review_conservative_center_roi" if args.conservative_review
            else "visual_review_corrected_roi", "foreground_found": True, "fallback": False}
    x, y, width, height = box
    crop = image[y:y + height, x:x + width]
    path = output / "crops" / (sample["image"] + ".png")
    if not cv2.imwrite(str(path), crop):
        raise RuntimeError("crop save failed")
    rows.append(dict(image=sample["image"], roi=box, crop_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        crop_area_ratio=width * height / float(image.shape[0] * image.shape[1]), **status))
    if index % 20 == 0:
        sheet = np.full((5 * 214, 4 * 640, 3), 245, np.uint8)
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + width - 1, y + height - 1), (0, 0, 255), 4)
    pair = np.concatenate([cv2.resize(annotated, (320, 180)), cv2.resize(crop, (320, 180))], axis=1)
    sx, sy = index % 4 * 640, index % 20 // 4 * 214
    sheet[sy:sy + 180, sx:sx + 640] = pair
    cv2.putText(sheet, sample["image"] + (" FALLBACK" if status["fallback"] else ""),
        (sx + 3, sy + 200), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
    if index % 20 == 19:
        cv2.imwrite(str(output / "contact_sheets" / f"sheet_{index // 20:02d}.jpg"), sheet)
report = {"count": len(rows), "samples": rows, "yolo_used": False,
    "fixed_rule": {"mode": args.mode, "hsv_lower": [35, 35 if args.mode == "original" else 180, 35], "hsv_upper": [95, 255, 255], "kernel": [5, 5],
        "min_area_ratio": .002, "padding_ratio": .12, "minimum_padding_pixels": 8},
    "reference": "original G:/OrangePi/Yolo_clip/src/auto_label/foreground.py; fixed padding added",
    "empty_policy": "known empty stays whole image; classifier must still predict",
    "no_predictions_used_for_cropping": True, "sample_exclusion_count": 0,
    "visual_review_correction_count": len(overrides),
    "visual_review_note": review["reviewer"] if review else None,
    "review_manifest_sha": hashlib.sha256(args.review.read_bytes()).hexdigest() if review else None,
    "conservative_review_canvas_roi": [75, 5, 245, 175] if args.conservative_review else None,
    "nonempty_foreground_found": sum(row["foreground_found"] for row in rows),
    "nonempty_fallback_count": sum(row["fallback"] for row in rows)}
(output / "crop_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
