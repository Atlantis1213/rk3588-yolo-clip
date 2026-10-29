#!/usr/bin/env python3
"""Measure CLIP-only capacity, then screen/confirm unchanged YOLO->CLIP schedules."""
import argparse
import json
from pathlib import Path
import subprocess


def execute(command, output, name):
    print("START " + name, flush=True)
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    item = {"command": command, "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode == 0:
        item["report"] = report = json.loads(result.stdout[result.stdout.find("{"):])
        print(f"DONE {name}: FPS={report.get('fps', report.get('completed_fps')):.3f}", flush=True)
    (output / f"{name}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
    if result.returncode: raise RuntimeError(name + ": " + result.stderr)
    return item


def pipeline_command(config, duration):
    return ["./build/yolo_clip_system_demo", "--yolo", "yolo_waste_single_class.rknn",
            "--clip", "clip_image_encoder_vitl14.rknn", "--label-tsv", "board_labels_vitl14.tsv",
            "--text-features", "board_text_features_vitl14.f32", "--images", "images",
            "--duration", str(duration), "--fps", "8", "--normalization", "lut", "--zero-copy-input",
            "--opencv-threads", "1", "--frame-queue", "4", "--clip-queue", "8", "--clip-buffers", "8",
            "--fallback-full-image", "--yolo-threads", str(config[1]), "--yolo-core-mode", config[2],
            "--clip-threads", str(config[3]), "--clip-core-mode", config[4], "--npu-gate", config[5]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/npu_scheduling_20261009")
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    micro = []
    for workers, mode, gate in [(1, "core0", "none"), (1, "all", "none"), (3, "spread", "none"),
                                (4, "round_robin", "none"), (3, "spread", "per_core"), (4, "round_robin", "per_core")]:
        name = f"cliponly_{workers}_{mode}_{gate}"
        command = ["./build/bench_clip_concurrency", "clip_image_encoder_vitl14.rknn",
                   "../yolo_clip_bench/sample_cycle00001_pixel_values_nhwc_fp16.bin",
                   str(workers), mode, "10", gate, "dma"]
        micro.append(execute(command, output, name))
    configs = [
        ("baseline_2y4c", 2, "legacy", 4, "round_robin", "none"),
        ("spread_1y3c", 1, "core0", 3, "spread", "none"),
        ("spread_2y3c", 2, "legacy", 3, "spread", "none"),
        ("gated_1y3c", 1, "core0", 3, "spread", "per_core"),
        ("gated_2y3c", 2, "legacy", 3, "spread", "per_core"),
        ("dedicated_1y2c", 1, "core0", 2, "split12", "none"),
        ("gated_baseline", 2, "legacy", 4, "round_robin", "per_core"),
        ("all_1y1c", 1, "core0", 1, "all", "none"),
    ]
    screened = [(config, execute(pipeline_command(config, 12), output, "screen_" + config[0])) for config in configs]
    ranked = sorted(screened, key=lambda pair: pair[1]["report"]["completed_fps"], reverse=True)
    selected = [configs[0]] + [pair[0] for pair in ranked if pair[0][0] != configs[0][0]][:2]
    confirmed = []
    for repeat in range(3):
        order = selected[repeat:] + selected[:repeat]
        for config in order:
            confirmed.append({"config": config, "run": execute(pipeline_command(config, 20), output,
                                 f"confirm_{repeat}_{config[0]}")})
    summary = []
    for config in selected:
        reports = [item["run"]["report"] for item in confirmed if item["config"][0] == config[0]]
        summary.append({"config": config, "runs": len(reports), "means": {
            key: sum(r[key] for r in reports) / len(reports) for key in
            ("completed_fps", "latency_ms_p95", "clip_run_ms_mean", "yolo_run_ms_mean",
             "clip_queue_wait_ms_mean", "clip_npu_gate_wait_ms_mean", "yolo_npu_gate_wait_ms_mean")}})
    result = {"clip_only": [item["report"] for item in micro], "screening": [item[1]["report"] for item in screened],
              "confirmed": summary}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
