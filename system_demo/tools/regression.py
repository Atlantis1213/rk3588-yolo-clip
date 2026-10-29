#!/usr/bin/env python3
"""Check refactor/LUT/native input against the original five-image model outputs."""
import json
from pathlib import Path
import subprocess


def report(binary, options):
    common = ["--yolo", "yolo_waste_single_class.rknn", "--clip", "clip_image_encoder_vitl14.rknn",
              "--label-tsv", "board_labels_vitl14.tsv", "--text-features", "board_text_features_vitl14.f32",
              "--images", "images", "--duration", "2", "--fps", "2.5", "--fallback-full-image"]
    run = subprocess.run([binary] + common + options, text=True, capture_output=True, timeout=40)
    if run.returncode: raise RuntimeError(run.stdout + run.stderr)
    return json.loads(run.stdout[run.stdout.find("{"):])


legacy_baseline = report("../yolo_clip_bench_optimized/yolo_clip_pipeline_bench_camera",
                  ["--yolo-threads", "1", "--clip-threads", "1", "--clip-core-mode", "all"])
baseline = report("./build/yolo_clip_system_demo", ["--normalization", "scalar", "--yolo-threads", "1",
                                                  "--clip-threads", "1", "--clip-core-mode", "all"])
results = []
for name, options in [
    ("scalar_host", ["--normalization", "scalar"]),
    ("lut_host", ["--normalization", "lut"]),
    ("scalar_dma", ["--normalization", "scalar", "--zero-copy-input"]),
    ("lut_dma", ["--normalization", "lut", "--zero-copy-input"]),
    ("lut_multicontext", ["--normalization", "lut", "--zero-copy-input", "--yolo-threads", "2", "--clip-threads", "4"]),
    ("lut_serial", ["--normalization", "lut", "--zero-copy-input", "--serial"]),
    ("legacy_lut", ["--normalization", "lut_legacy"]),
    ("legacy_scalar", ["--normalization", "scalar_legacy"]),
]:
    configuration = ["--yolo-threads", "1", "--clip-threads", "1", "--clip-core-mode", "all"] + options
    actual = report("./build/yolo_clip_system_demo", configuration)
    expected = legacy_baseline if name.startswith("legacy_") else baseline
    for key in ("completed", "detected", "checksum_xor", "sample_predictions"):
        if actual[key] != expected[key]: raise RuntimeError(f"{name}: mismatch in {key}")
    print(f"PASS {name}: predictions and checksum match corresponding IEEE/legacy reference", flush=True)
    results.append({"name": name, "report": actual})
Path("results").mkdir(exist_ok=True)
Path("results/regression.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
