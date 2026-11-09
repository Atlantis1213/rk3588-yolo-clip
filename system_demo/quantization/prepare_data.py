# 公开数据准备入口：六类各取 40 张，前 20 张用于校准，后 20 张用于验证。
# 输出 calibration.txt、标准化 NCHW 张量、验证 PNG 和来源清单；执行本脚本会下载/写入数据。

"""Public TrashNet calibration/eval split; no test images are used for calibration."""
import hashlib
import json
from pathlib import Path
import random
import subprocess
import urllib.request
import zipfile
import cv2
import numpy as np

root = Path(__file__).resolve().parent
assets = root / "assets"
assets.mkdir(exist_ok=True)
archive = assets / "dataset-resized.zip"
url = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"
if not archive.exists() or archive.stat().st_size != 42834870:
    print("Downloading public TrashNet resized archive", flush=True)
    temporary = archive.with_suffix(".download")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--retry-all-errors",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    if temporary.stat().st_size != 42834870:
        raise RuntimeError("incomplete TrashNet download")
    temporary.replace(archive)
if archive.stat().st_size != 42834870:
    raise RuntimeError("unexpected TrashNet archive size")

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
artifacts = root / "artifacts"
calibration = artifacts / "calibration_npy"
calibration.mkdir(parents=True, exist_ok=True)
evaluation = artifacts / "eval_png"
evaluation.mkdir(exist_ok=True)
# 局部随机数生成器不受其他代码的随机调用影响；固定 seed 和排序共同保证划分可复现。
randomizer = random.Random(20261009)
rows, cal_paths, cal_sources = [], [], []
with zipfile.ZipFile(archive) as dataset:
    members = dataset.namelist()
    for label in classes:
        # 先筛选当前类别的原始 JPG，再按固定随机顺序抽样，避免不同类别样本混入同一切片。
        names = sorted(n for n in members if n.lower().endswith(".jpg") and f"/{label}/" in n)
        randomizer.shuffle(names)
        if len(names) < 40:
            raise RuntimeError("insufficient class images: " + label)
        for index, name in enumerate(names[:40]):
            bgr = cv2.imdecode(np.frombuffer(dataset.read(name), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cannot decode " + name)
            resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
            identifier = f"{label}_{index:03d}"
            # 每类独立切分；全局共 120 张校准 + 120 张验证，不能把验证图再用于调整量化范围。
            if index < 20:
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                # 顺序必须与板端一致：resize224 → RGB → 标准化 → FP16 舍入 → 转回 FP32。
                # 最后转回 FP32 只满足工具输入类型要求，不会恢复被半精度舍入丢掉的信息。
                values = (
                    ((rgb.astype(np.float32) / 255 - mean) / std)
                    .astype(np.float16)
                    .astype(np.float32)
                )
                path = calibration / (identifier + ".npy")
                # HWC 转 CHW 后补 batch 维，最终 [1,3,224,224]；copy 生成连续内存。
                np.save(path, values.transpose(2, 0, 1)[None].copy())
                cal_paths.append(str(path))
                cal_sources.append(name)
            else:
                filename = identifier + ".png"
                cv2.imwrite(str(evaluation / filename), resized)
                rows.append({"image": filename, "class": label, "source": name})
# 按原始归档成员名核对划分互斥，再保存来源与归档 SHA，便于追溯实验输入。
if set(cal_sources) & {r["source"] for r in rows}:
    raise RuntimeError("calibration/evaluation overlap")
(artifacts / "calibration.txt").write_text("\n".join(cal_paths) + "\n")
(artifacts / "eval_labels.json").write_text(
    json.dumps({"classes": classes, "samples": rows}, indent=2)
)
(artifacts / "dataset_provenance.json").write_text(
    json.dumps(
        {
            "source": url,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "seed": 20261009,
            "calibration_count": len(cal_paths),
            "eval_count": len(rows),
            "calibration_sources": cal_sources,
            "note": "Public six-class classification; not the original four-category plus unknown deployment dataset.",
        },
        indent=2,
    )
)
print(
    f"Prepared {len(cal_paths)} calibration and {len(rows)} disjoint evaluation samples", flush=True
)
