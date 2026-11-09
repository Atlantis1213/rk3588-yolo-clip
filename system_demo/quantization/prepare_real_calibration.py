# 真实场景校准样本选择：从五类原始数据中排除已有 200 张评估样本及精确重复图。
# select_samples 只选择并返回清单；main 才复制图像并打包，不会在选择函数中执行 YOLO。

"""Select disjoint real calibration images, never recycle the 200 evaluation images."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile

import cv2
import numpy as np


# 返回记录包含来源、文件哈希、解码像素哈希和原图尺寸；默认每类 48 张，共 240 张。
def select_samples(source, evaluation, per_class=48, seed=20261012):
    if per_class <= 0 or len(evaluation["samples"]) != 200:
        raise ValueError(
            "positive per-class count and original 200-image exclusion manifest required"
        )
    if (
        str(source).replace("\\", "/").lower()
        != evaluation["source_root"].replace("\\", "/").lower()
    ):
        raise ValueError("calibration source must match the evaluation dataset root")
    classes = evaluation["classes"]
    if set(classes) != {"cycle", "harmful", "kitchen", "others", "empty"}:
        raise ValueError("expected five fixed real dataset classes")
    # 三个层次去重：原始路径、文件字节、解码像素。
    # 像素哈希能识别重新封装但画面完全相同的图片，仍不证明不同对象实例或近重复图已独立。
    excluded_sources = {row["source"].replace("\\", "/") for row in evaluation["samples"]}
    files = {row["file_sha256"] for row in evaluation["samples"]}
    pixels = {row["original_pixels_sha256"] for row in evaluation["samples"]}
    generator = random.Random(seed)
    selected = []
    for category in classes:
        candidates = sorted(
            path
            for path in (source / category).rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        # 先排序后用固定 seed 打乱；类别遍历顺序来自已冻结的评估类别清单。
        generator.shuffle(candidates)
        chosen = 0
        for path in candidates:
            relative = path.relative_to(source).as_posix()
            if relative in excluded_sources:
                continue
            payload = path.read_bytes()
            file_sha = hashlib.sha256(payload).hexdigest()
            if file_sha in files:
                continue
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("undecodable calibration source: " + relative)
            pixel_sha = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if pixel_sha in pixels:
                continue
            selected.append(
                {
                    "image": f"calibration_{category}_{chosen:03d}" + path.suffix.lower(),
                    "class": category,
                    "source": relative,
                    "file_sha256": file_sha,
                    "original_pixels_sha256": pixel_sha,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "skip_yolo": category == "empty",
                }
            )
            # 选中后立即加入排除集合，使后续类别/候选也不能再次选中这张图的精确副本。
            files.add(file_sha)
            pixels.add(pixel_sha)
            chosen += 1
            if chosen == per_class:
                break
        # 不足时直接失败，不默默缩小某一类样本数，保持校准类别均衡。
        if chosen != per_class:
            raise ValueError("insufficient disjoint unique samples: " + category)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--exclude-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=48)
    args = parser.parse_args()
    source, exclusion = Path(args.source), Path(args.exclude_manifest)
    evaluation = json.loads(exclusion.read_text(encoding="utf-8"))
    rows = select_samples(source, evaluation, args.per_class)
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    (output / "images").mkdir()
    for row in rows:
        payload = (source / row["source"]).read_bytes()
        # 选择与复制分两步，复制前再次校验，防止两步之间源图片被更改。
        if hashlib.sha256(payload).hexdigest() != row["file_sha256"]:
            raise RuntimeError("source changed after selection")
        with (output / "images" / row["image"]).open("xb") as handle:
            handle.write(payload)
    manifest = {
        "purpose": "calibration only, not additional validation/test",
        "source_root": str(source),
        "count": len(rows),
        "classes": evaluation["classes"],
        "per_class_count": args.per_class,
        "seed": 20261012,
        "samples": rows,
        "excluded_eval_manifest_sha256": hashlib.sha256(exclusion.read_bytes()).hexdigest(),
        "eval_source_file_and_pixel_disjoint": True,
        "instance_or_near_duplicate_independence_proven": False,
        "preprocessing_pending": "YOLO frozen top1 proposal for nonempty; empty full image; same CLIP normalization",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "samples.tsv").open("x", encoding="utf-8") as handle:
        handle.writelines(row["image"] + "\t" + row["class"] + "\n" for row in rows)
    with zipfile.ZipFile(
        output / "real_calibration.zip", "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        for name in ["manifest.json", "samples.tsv"]:
            archive.write(output / name, name)
        for row in rows:
            archive.write(output / "images" / row["image"], "images/" + row["image"])
    print(json.dumps({"count": len(rows), "eval_source_file_and_pixel_disjoint": True}), flush=True)


if __name__ == "__main__":
    main()
