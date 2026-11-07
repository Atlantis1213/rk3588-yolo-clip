#!/usr/bin/env python3
# 板端摄像头 A/B/C 对照：CPU+scalar、DMA+scalar、DMA+LUT，固定模型与调度设置。
# 每轮旋转顺序，保留原始输出和资源采样；实景帧不逐像素相同，需保持场景稳定。

"""Balanced short A/B/C runs with identical V4L2/model/scheduling settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def sample(pid):
    root = Path(f"/proc/{pid}")
    # /proc/stat 用最后一个右括号分离进程名，后续字段取 utime/stime 累计 CPU tick。
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])
    rss = next(int(s.split()[1]) for s in (root / "status").read_text().splitlines() if s.startswith("VmRSS:"))
    return ticks, rss, len(list((root / "fd").iterdir()))


def run(args, path, norm, index, output):
    command = [args.binary, "--yolo", "yolo_waste_single_class.rknn", "--clip", "clip_image_encoder_vitl14.rknn",
               "--label-tsv", "board_labels_vitl14.tsv", "--text-features", "board_text_features_vitl14.f32",
               "--camera", args.camera, "--camera-path", path, "--normalization", norm,
               "--camera-width", "640", "--camera-height", "480", "--camera-format", "YUYV",
               "--camera-buffers", "12", "--duration", str(args.duration), "--fps", "30",
               "--yolo-threads", "2", "--clip-threads", "4", "--frame-queue", "4",
               "--clip-queue", "8", "--clip-buffers", "8", "--clip-core-mode", "round_robin",
               "--affinity-profile", "none", "--opencv-threads", "1", "--fallback-full-image"]
    name = f"{index:02d}_{path}_{norm}"
    print("START " + name, flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start, previous, samples = time.monotonic(), None, []
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    timeout = False
    while process.poll() is None:
        now = time.monotonic()
        if now - start > args.duration + 40:
            timeout = True
            process.terminate()
            break
        try:
            ticks, rss, fds = sample(process.pid)
            # 以系统 tick 频率换算进程 CPU 时间；100% 表示占满一个核，多线程可超过 100%。
            cpu = None if previous is None else (ticks - previous[0]) / ticks_per_second / (now - previous[1]) * 100
            samples.append({"sec": now - start, "cpu_percent": cpu, "rss_kib": rss, "fds": fds})
            previous = ticks, now
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            pass
        time.sleep(0.25)
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        timeout = True
        process.kill()
        stdout, stderr = process.communicate()
    result = {"name": name, "command": command, "returncode": process.returncode, "timeout": timeout,
              "stdout": stdout, "stderr": stderr, "resources": samples}
    if process.returncode == 0 and not timeout:
        result["report"] = report = json.loads(stdout[stdout.find("{"):])
        # 排除前 5 秒预热，只用持续输入期间的样本计算 CPU 均值。
        stable = [s for s in samples if 5 <= s["sec"] <= args.duration and s["cpu_percent"] is not None]
        result["cpu_percent_mean"] = sum(s["cpu_percent"] for s in stable) / len(stable) if stable else None
        if not report["completed"] or report["skipped_no_object"]:
            raise RuntimeError("not a complete pipeline test")
        # 每个生成帧应归入完成或某类丢弃；对账失败说明运行不能作为有效对照结果。
        if report["generated_frames"] != sum(report[k] for k in ("completed", "frame_queue_dropped", "clip_queue_dropped", "clip_slot_dropped")):
            raise RuntimeError("frame accounting mismatch")
        print(f"DONE {name}: FPS={report['completed_fps']:.3f}, "
              f"CPU={result['cpu_percent_mean']:.2f}%, YOLOprep={report['yolo_preprocess_ms_mean']:.2f}ms, "
              f"CLIPprep={report['clip_preprocess_ms_mean']:.2f}ms", flush=True)
    (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if process.returncode != 0 or timeout:
        raise RuntimeError(f"{name}: " + stderr)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/yolo_clip_system_demo")
    parser.add_argument("--camera", default="/dev/v4l/by-path/platform-xhci-hcd.12.auto-usb-0:1.4:1.0-video-index0")
    parser.add_argument("--duration", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", default="results/camera_comparison")
    args = parser.parse_args()
    if args.duration <= 5 or args.rounds <= 0:
        raise ValueError("duration must exceed 5s; rounds must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    variants = [("cpu", "scalar"), ("dma", "scalar"), ("dma", "lut")]
    results = []
    for round_index in range(args.rounds):
        # ABC, BCA, CAB reduces fixed-order bias without changing model scheduling.
        order = variants[round_index % 3:] + variants[:round_index % 3]
        for path, norm in order:
            results.append(run(args, path, norm, len(results) + 1, output))
    summary = []
    for path, norm in variants:
        selected = [r for r in results if r["report"]["camera_path"] == path and r["report"]["normalization_mode"] == norm]
        metrics = {key: sum(r["report"][key] for r in selected) / len(selected) for key in
                   ("completed_fps", "latency_ms_p95", "yolo_preprocess_ms_mean", "clip_preprocess_ms_mean")}
        metrics["cpu_percent_mean"] = sum(r["cpu_percent_mean"] for r in selected) / len(selected)
        metrics["explicit_copy_bytes_per_run"] = sum(r["report"]["explicit_cpu_copy_bytes"] for r in selected) / len(selected)
        summary.append({"camera_path": path, "normalization": norm, "runs": len(selected), "means": metrics})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
