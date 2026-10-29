# V4L2 USB camera + DMA-BUF + RGA + RKNN

## Files and deployment

- `yolo_clip_pipeline_bench.optimized.cpp`: existing benchmark extended with camera input and native DMA tensor input.
- `camera_dma.hpp`: V4L2 buffer ownership, DMA heap allocation, cache synchronization and FD-based RGA preprocessing.
- `build_camera.sh`: build a separate `yolo_clip_pipeline_bench_camera` executable.
- `verify_camera_pipeline.py`: board-side integration checks with real RGA/RKNN.

The camera version is deployed under `/home/orangepi/yolo_clip_bench_optimized/` on `192.168.123.99`.
Its source is named `yolo_clip_pipeline_bench.camera.cpp` on the board. The original project
`/home/orangepi/yolo_clip_bench` and the previous optimized benchmark source/executable remain intact.

Build on the board:

```sh
cd /home/orangepi/yolo_clip_bench_optimized
sh build_camera.sh
```

For a local Linux checkout where the source retains its local filename:

```sh
sh build_camera.sh yolo_clip_pipeline_bench.optimized.cpp
```

## Actual data path

```text
USB camera (progressive YUYV or single-plane NV12)
  -> V4L2 REQBUFS(MMAP) / EXPBUF / DQBUF
  -> exported camera DMA-BUF FD, imported into RGA once per buffer
  -> RGA converts into a worker-private BGR DMA image
  -> RGA copies into a reusable square letterbox DMA image
  -> RGA resizes and converts to RGB 640x640
  -> CPU normalizes directly into a DMA-backed FP16 tensor
  -> RKNN create_mem_from_fd + set_io_mem(native NHWC, pass_through=1)
  -> YOLO NPU inference and CPU box selection
  -> RGA crops/resizes the BGR DMA image into RGB 224x224
  -> CPU applies CLIP normalization directly into a DMA tensor pool slot
  -> CLIP queue carries slot_id
  -> CLIP worker binds the slot FD and executes inference
  -> text feature matching, statistics and slot recycling
```

Camera frames never become `cv::Mat`; there is no FFmpeg, OpenCV capture, `clone()` or
CPU pixel upload on the camera path. RGA still reads/writes image data in memory: sharing
FDs avoids CPU copies, not hardware memory traffic. RGB-to-FP16 normalization remains CPU work,
and writes its result directly into the input DMA buffer. The final tensor therefore has a
different FD from the image buffer. This is not a raw camera FD passed unchanged to the NPU.

The board's RGA driver rejected color-fill jobs during testing. Constant letterbox padding
is initialized to 114 by the CPU once per source geometry, flushed to the device, and reused.
Per-frame image placement and scaling remain on RGA.

## Ownership and synchronization

- Each dequeued V4L2 frame owns a reference to the capture object. Its destructor requeues
  the buffer with QBUF, including when a pending frame is dropped from the input queue.
- RGA operations are synchronous. Buffers are not requeued while an RGA operation is in flight.
- Each YOLO worker owns its RGA output images and preprocessing state.
- Camera buffers are returned after ROI preprocessing; CLIP queue tasks retain only tensor slots.
- The capture driver must provide at least `frame_queue + YOLO_workers + 2` buffers, to keep
  frames available for the device while pending/processing frames are held by software.
- CPU accesses to DMA memory use DMA_BUF_IOCTL_SYNC START/END with READ/WRITE directions.
- Each RKNN context imports pool-slot FDs lazily and retains both the RKNN memory wrapper and
  the underlying DMA buffer. Wrappers are destroyed before context and FD teardown.
- Producer/worker exceptions close the queues, join threads, and report an error in main.

## Camera prerequisites

```sh
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

Select the actual image capture node, rather than a metadata node. This implementation supports
progressive single-plane V4L2 capture with `YUYV` or contiguous `NV12`, including row padding.
The driver must support `VIDIOC_EXPBUF`, and RGA must be able to import the exported memory.
Unsupported formats or FD imports fail explicitly; there is no hidden CPU-copy fallback.
BT.601 limited/full and BT.709 limited color conversion are supported; BT.709 full-range
is rejected because the installed RGA runtime's full-CSC path failed hardware testing.

MJPEG/H.264-only camera modes need a compressed-stream decoder and are not implemented here.
Raw YUYV/NV12 capture does not need MPP. Request a mode that the USB camera actually supports.

RKNN direct input currently requires packed native FP16 NHWC tensors (batch 1, channels 3).
Native stride/layout is checked before accepting zero-copy mode. Models requiring another
native layout fail rather than silently reinterpret data. RGB normalization semantics are unchanged.

## Run with a USB camera

```sh
cd /home/orangepi/yolo_clip_bench_optimized
./yolo_clip_pipeline_bench_camera \
  --yolo yolo_waste_single_class.rknn \
  --clip clip_image_encoder_vitl14.rknn \
  --label-tsv board_labels_vitl14.tsv \
  --text-features board_text_features_vitl14.f32 \
  --camera /dev/video0 \
  --camera-format YUYV --camera-width 640 --camera-height 480 \
  --camera-buffers 12 --fps 30 --duration 10 \
  --frame-queue 4 --yolo-threads 2 --clip-threads 4 \
  --clip-core-mode round_robin --affinity-profile none
```

Camera mode automatically enables RGA and native DMA input. `--fps` requests a camera frame
interval via V4L2; a driver may negotiate a different FPS. Acquisition consumes available frames
continuously, with oldest-pending-frame dropping when the queue is full. The initial V4L2 log
shows the accepted image dimensions, format and buffer count. Use `--fallback-full-image` to
force CLIP processing even when YOLO detects no target; default behavior skips those frames.

`--dma-heap PATH` overrides `/dev/dma_heap/system`. The system heap was tested on this board;
`/dev/dma_heap/reserved` was not accessible to the orangepi user and is not required.

## Checks without a camera

```sh
./yolo_clip_pipeline_bench_camera --dma-self-test
python3 verify_camera_pipeline.py
```

The self-test allocates actual DMA-BUFs and exercises YUYV/NV12 conversion, resize,
letterbox, odd-coordinate RGB ROI crop, and cache synchronization on the real RGA.
Integration checks compare five image outputs and checksums between host input and native
DMA input in one-context, multi-context and serial modes, and check graceful producer errors.
These are correctness tests, not evidence of a performance gain.

All these checks passed on the board on 2026-10-09. Each inference mode completed all five
test images with identical sample predictions and `checksum_xor=18313580699023835136`.
The build completed with `-Wall -Wextra` without warnings.

Initially on 2026-10-09 no USB capture node was present. After camera connection, real YUYV
capture, EXPBUF, RGA FD import and native RKNN input passed a 10-second smoke test and a
60-second complete pipeline test. USB disconnect/re-enumeration interrupted further runs,
including a capture-only V4L2 test, so long-run stability is not established. See
`USB_CAMERA_TEST_20261009.md` for measurements and limitations. Missing or disconnected
devices produce handled error exits.

Existing `--images` and `--video` modes remain available. Add `--zero-copy-input` to test their
native DMA tensor input independently; their original OpenCV decoding/preprocessing path remains.
Latency starts at successful dequeue/enqueue, not at camera exposure or USB transfer. Reported
FPS uses wall time including queue draining; do not treat it as the device's capture FPS.
