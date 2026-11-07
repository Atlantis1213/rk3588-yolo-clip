#!/usr/bin/env python3
# 真实 USB 摄像头的板端运行与资源采样脚本，需要连接设备和已部署模型。
# 通过 /proc 采样进程 CPU、内存与 FD，保存报告；它不是无需硬件的单元测试。

"""Run repeatable live-camera inference tests and sample process resources on Linux."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def sample_process(pid):
    root = Path(f"/proc/{pid}")
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])  # Original stat fields 14/15.
    status = (root / "status").read_text().splitlines()
    rss_kib = next((int(row.split()[1]) for row in status if row.startswith("VmRSS:")), 0)
    return ticks, rss_kib, len(list((root / "fd").iterdir()))


def benchmark(duration, fps, name, output, camera):
    command = ["./yolo_clip_pipeline_bench_camera",
               "--yolo", "yolo_waste_single_class.rknn",
               "--clip", "clip_image_encoder_vitl14.rknn",
               "--label-tsv", "board_labels_vitl14.tsv",
               "--text-features", "board_text_features_vitl14.f32",
               "--camera", camera, "--camera-format", "YUYV",
               "--camera-width", "640", "--camera-height", "480",
               "--camera-buffers", "12", "--fps", str(fps), "--duration", str(duration),
               "--frame-queue", "4", "--yolo-threads", "2", "--clip-threads", "4",
               "--clip-core-mode", "round_robin", "--fallback-full-image"]
    print(f"START {name}: {duration}s, requested camera FPS={fps}", flush=True)
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    previous = None
    tick_rate = os.sysconf("SC_CLK_TCK")
    next_progress = 20
    timed_out = False
    while process.poll() is None:
        now = time.monotonic()
        elapsed = now - start
        if elapsed > duration + 45:
            timed_out = True
            process.terminate()
            break
        try:
            ticks, rss_kib, fd_count = sample_process(process.pid)
            cpu = None if previous is None else 100 * (ticks - previous[0]) / tick_rate / (now - previous[1])
            previous = ticks, now
            samples.append({"elapsed_sec": round(elapsed, 3), "cpu_percent": cpu,
                            "rss_kib": rss_kib, "fd_count": fd_count})
        except (FileNotFoundError, ProcessLookupError):
            pass
        if elapsed >= next_progress:
            recent = samples[-1] if samples else {}
            print(f"PROGRESS {name}: {elapsed:.0f}s RSS={recent.get('rss_kib')}KiB "
                  f"FDs={recent.get('fd_count')}", flush=True)
            next_progress += 20
        time.sleep(1)
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        timed_out = True
    result = {"name": name, "command": command, "returncode": process.returncode,
              "timed_out": timed_out, "stdout": stdout, "stderr": stderr, "resources": samples}
    if process.returncode == 0 and not timed_out:
        report = json.loads(stdout[stdout.find("{"):])
        result["report"] = report
        stable = [s for s in samples if 10 <= s["elapsed_sec"] <= duration and s["rss_kib"] > 0]
        if stable:
            cpu = [s["cpu_percent"] for s in stable if s["cpu_percent"] is not None]
            result["resource_summary"] = {
                "cpu_percent_mean": sum(cpu) / len(cpu) if cpu else None,
                "rss_mib_first": stable[0]["rss_kib"] / 1024,
                "rss_mib_last": stable[-1]["rss_kib"] / 1024,
                "rss_mib_min": min(s["rss_kib"] for s in stable) / 1024,
                "rss_mib_max": max(s["rss_kib"] for s in stable) / 1024,
                "fd_count_first": stable[0]["fd_count"],
                "fd_count_last": stable[-1]["fd_count"],
                "fd_count_max": max(s["fd_count"] for s in stable),
            }
        if report["completed"] <= 0 or report["skipped_no_object"] != 0:
            raise RuntimeError("full pipeline did not execute as expected")
        accounted = (report["completed"] + report["frame_queue_dropped"] +
                     report["clip_queue_dropped"] + report["clip_slot_dropped"])
        if accounted != report["generated_frames"]:
            raise RuntimeError("frame accounting mismatch")
        print(f"DONE {name}: {report['completed_fps']:.3f} FPS, "
              f"P95={report['latency_ms_p95']:.1f}ms, "
              f"captured={report['generated_frames']}, completed={report['completed']}", flush=True)
    # Generated measurement artifact, retained even when a camera/RGA/RKNN error occurs.
    (output / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if process.returncode != 0 or timed_out:
        raise RuntimeError(f"{name} failed: {stderr}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="usb_camera_results")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stability-duration", type=int, default=180)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--camera", default="/dev/video0")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = [benchmark(args.duration, args.fps, f"repeat_{i+1}", output, args.camera)
               for i in range(args.repeats)]
    if args.stability_duration:
        results.append(benchmark(args.stability_duration, args.fps, "stability", output, args.camera))
    summary = [{"name": item["name"], "report": item["report"],
                "resource_summary": item.get("resource_summary", {})} for item in results]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("ALL PASS: " + str(output / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
