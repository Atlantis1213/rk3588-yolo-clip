# USB camera system test — 2026-10-09

## Outcome

Real V4L2 -> exported DMA-BUF -> RGA -> native RKNN FD input -> YOLO/CLIP inference
completed successfully. One full 60-second inference test completed without process errors.
The repeated-run and 180-second stability plan did not finish: the USB device disconnected
and re-enumerated repeatedly. A capture-only V4L2 control test also disconnected, without
RGA, RKNN or NPU inference running.

This establishes functional DMA inference on the actual camera, but not long-run stability,
classification accuracy, or a speedup relative to a same-camera CPU-copy baseline.

## Hardware and configuration

- Board: Orange Pi 5 Plus, RK3588, SSH 192.168.123.99.
- Camera: USB FHD Camesa, USB ID 1bf2:b550, uvcvideo driver 6.1.43.
- Raw format: progressive YUYV 640x480, bytesperline 1280, sizeimage 614400.
- Requested and driver-reported frame interval: 30 FPS; measured delivery approximately 10 FPS.
- V4L2 exported buffers: 12; driver EXPBUF and RGA FD import both worked.
- Pipeline: YOLO workers 2, CLIP workers 4, round_robin NPU allocation, frame queue 4,
  CLIP queue/pool 8, CPU affinity none, system DMA heap.
- `--fallback-full-image` enabled. No frame skipped CLIP because of absent detection.
- Stable node used on retry:
  `/dev/v4l/by-path/platform-xhci-hcd.12.auto-usb-0:1.4:1.0-video-index0`.
  This resolves the current node at open time; it does not reconnect an already open capture FD.

## Completed measurements

| Metric | 10-second smoke test | 60-second valid run |
| --- | ---: | ---: |
| Captured frames | 100 | 598 |
| Full YOLO/CLIP completions | 44 | 231 |
| Completed FPS, including drain time | 3.533 | 3.696 |
| Input queue drops | 56 | 367 |
| CLIP queue / tensor-pool drops | 0 / 0 | 0 / 0 |
| Latency P50 | 1920.863 ms | 1941.553 ms |
| Latency P95 | 2543.984 ms | 2606.735 ms |
| YOLO preprocessing mean | 24.503 ms | 20.787 ms |
| YOLO run mean | 475.066 ms | 501.684 ms |
| CLIP preprocessing mean | 4.643 ms | 4.156 ms |
| CLIP run mean | 1024.052 ms | 1030.400 ms |
| Frames with a detected ROI | 0 | 0 |

Both completed runs therefore used full-image CLIP fallback. Real camera ROI detection remains
to be exercised with an object recognized by the detector; DMA ROI processing has separate
synthetic hardware self-test coverage. No accuracy claim is possible without labeled scenes.

The 60-second run used 62.494 seconds including pipeline draining. CPU resource sampling
from process seconds 10 to 60 measured 15.40% mean CPU, where 100% denotes one CPU core,
not the entire eight-core board. RSS stayed at 1372.77 MiB and open FD count stayed at 58
throughout that sampled interval. No growing RSS/FD trend appeared within this one run.

Frame dropping is the configured oldest-pending-frame policy under overload. Frame accounting
matched exactly: 598 = 231 completions + 367 input queue drops.

Timing begins at successful V4L2 dequeue, and excludes sensor exposure and USB transmission.
The run timer excludes model initialization. RGA/RKNN success is evidence of correct operation;
no controlled CPU-copy camera comparison has been measured yet.

## Interrupted tests and source-only control

- Initial 60-second run: interrupted around process second 40 by camera loss.
- Retry first 60-second run: passed; raw record is `usb_camera_results_20261009_retry1/repeat_1.json`.
- Retry second run: interrupted around process second 12 by camera loss.
- Capture-only command below: returned exit code 1 before all 600 frames were captured;
  the kernel logged another USB disconnect and URB error -19.

```sh
timeout -k 3 75 v4l2-ctl \
  -d /dev/v4l/by-path/platform-xhci-hcd.12.auto-usb-0:1.4:1.0-video-index0 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --set-parm=30 --stream-mmap=12 --stream-count=600 --stream-to=/dev/null
```

Capture-only output settled around 9.98 FPS, despite the reported 30 FPS setting. Camera
automatic exposure with dynamic frame rate was enabled, but the cause of reduced delivery
rate was not isolated. No exposure, USB power, governor or driver settings were altered.

Relevant kernel events (timestamps are seconds since boot):

```text
[10301.197191] usb 1-1.4: USB disconnect, device number 5
[10301.199778] uvcvideo 1-1.4:1.1: Failed to resubmit video URB (-19).
[10556.764279] usb 1-1.4: USB disconnect, device number 6
[10570.588336] usb 1-1.4: USB disconnect, device number 7
[10747.116217] usb 1-1.4: USB disconnect, device number 8
[10747.117707] uvcvideo 1-1.4:1.1: Failed to resubmit video URB (-19).
[10747.408557] usb 1-1.4: new high-speed USB device number 9 using xhci-hcd
```

These logs establish USB device loss, not its physical cause. Whether the camera was manually
unplugged remains unconfirmed. Cable/contact, hub/port/power and driver/device behavior are
possible causes; inference is not necessary to trigger the failure, as shown by the source-only test.
The application exited with a handled error instead of hanging on the lost device.

## Next validation

Keep the connection fixed, preferably directly connected to a board USB port, and first complete
a source-only capture run. Then repeat full inference using `test_usb_camera.py`:

```sh
cd /home/orangepi/yolo_clip_bench_optimized
python3 -u test_usb_camera.py \
  --camera /dev/v4l/by-path/platform-xhci-hcd.12.auto-usb-0:1.4:1.0-video-index0 \
  --output usb_camera_results_next --duration 60 --repeats 3 --stability-duration 180 --fps 30
```

The original and retry measurements were saved separately on the board and downloaded locally.
No camera disconnects were suppressed or retried inside a measurement interval.
