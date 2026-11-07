#!/usr/bin/env python3
# 旧摄像头 demo 的板端集成验证：自检真实 RGA/DMA、比较宿主与 DMA 输入并检查失败退出。
# 文件输入可验证模型链路，但不能代替 USB 摄像头的采集/EXPBUF/归还测试。

"""Board-side integration checks using real RGA and RKNN, without a USB camera."""
import json
import subprocess
import sys


def run(arguments, success=True):
    result = subprocess.run(arguments, text=True, capture_output=True, timeout=90)
    if success and result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)
    if not success and result.returncode != 1:
        raise RuntimeError(f"expected graceful exit 1, got {result.returncode}: {result.stderr}")
    return result


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "./yolo_clip_pipeline_bench_camera"
    run([binary, "--dma-self-test"])
    print("PASS: real DMA/RGA YUYV, NV12, resize, letterbox and odd-coordinate ROI")
    common = [binary, "--yolo", "yolo_waste_single_class.rknn",
              "--clip", "clip_image_encoder_vitl14.rknn",
              "--label-tsv", "board_labels_vitl14.tsv",
              "--text-features", "board_text_features_vitl14.f32",
              "--duration", "2", "--fps", "2.5", "--fallback-full-image"]
    pool = ["--images", "images", "--yolo-threads", "1", "--clip-threads", "1",
            "--clip-core-mode", "all"]
    # 以低输入速率跑五帧，减少性能丢帧影响；之后按校验和与分类结果逐配置对照。
    baseline = json.loads(run(common + pool).stdout)
    if baseline["completed"] != 5:
        raise RuntimeError("baseline did not complete all five frames")
    configurations = [
        ("native DMA input", pool + ["--zero-copy-input"]),
        ("multiple contexts", ["--images", "images", "--yolo-threads", "2",
                               "--clip-threads", "4", "--zero-copy-input"]),
        ("serial DMA input", ["--images", "images", "--serial", "--zero-copy-input"]),
    ]
    for name, options in configurations:
        report = json.loads(run(common + options).stdout)
        for key in ("completed", "detected", "checksum_xor", "sample_predictions"):
            if report[key] != baseline[key]:
                raise RuntimeError(f"{name}: output mismatch for {key}")
        if not report["zero_copy_input"]:
            raise RuntimeError("DMA input flag missing")
        print(f"PASS: {name}, all five outputs match host input exactly")
    # 主动注入输入失败，验证生产者异常能唤醒并回收 worker，而不是挂起进程。
    failed = run(common + ["--video", "/nonexistent/camera_test.mp4",
                           "--yolo-threads", "1", "--clip-threads", "1"], success=False)
    if "failed to open video" not in failed.stderr:
        raise RuntimeError("unexpected producer failure: " + failed.stderr)
    print("PASS: producer error joins worker threads and exits gracefully")
    run(common + ["--images", "images", "--camera", "/dev/video0"], success=False)
    print("PASS: mutually exclusive input validation")
    print("USB V4L2 capture/EXPBUF/requeue require a real connected camera; not covered here.")


if __name__ == "__main__":
    main()
