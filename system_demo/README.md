# YOLO-CLIP system optimization demo on RK3588

This is an isolated continuation of the verified camera demo. The previous demo and the
original model files are retained. Model structure, image normalization formulas, and the
YOLO -> ROI -> CLIP -> text matching sequence are preserved.

## Structure

```text
include/
  v4l2_capture.hpp       camera negotiation, exported FDs, frame ownership
  dma_buffer.hpp         DMA heap and CPU/device cache ownership
  rga_preprocess.hpp     FD-based image preprocessing and hardware self-test
  normalization.hpp      scalar reference and exact byte-to-FP16 LUT
  model_runner.hpp       RKNN context, native input import, detection/encoding
  buffer_pool.hpp        move-only RAII input slot leases
  pipeline_queue.hpp     bounded queue, oldest-pending drop, close/wakeup
  scheduler.hpp          CPU affinity and worker exception coordination
  npu_arbiter.hpp        optional per-core admission and adaptive YOLO placement
  metrics.hpp            phase timing, latency and explicit copy reporting
  input_io.hpp / text_bank.hpp / types.hpp
src/
  main.cpp               CLI entry
  pipeline.cpp           worker orchestration and input dispatch
tests/                   portable correctness/concurrency/normalization tests
tools/                   short camera A/B/C, regression, model inventory
docs/                    measurements and interview/debug notes
```

## Build and test on the board

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O3 -g -DNDEBUG'
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

RKNN headers are located with `RKNN_INCLUDE_DIR` (default search includes
`/userdata/yolo_clip_bench`), RGA headers with `RGA_INCLUDE_DIR`. Runtime libraries and
OpenCV must be installed on the target. This is native ARM compilation, not cross compilation.
For a cross build, supply an actual toolchain file and a matching target sysroot/libraries;
the project does not pretend the host's OpenCV/RGA libraries are ARM dependencies.

Portable tests can be built on Linux without RKNN/RGA/OpenCV:

```sh
cmake -S . -B build-sanitize -DBUILD_HARDWARE_DEMO=OFF \
  -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize -j2
ctest --test-dir build-sanitize --output-on-failure
```

Tests cover normalization bits for all 256 input values/channel, padded rows/tails,
invalid inputs, close/wakeup, dropped/closed tasks, producer/consumer exceptions,
4,000 concurrent tasks, and input leases that outlive the pool wrapper. The hardware
libraries are intentionally outside sanitizer instrumentation.

## Controlled camera comparison

All variants use the same V4L2 raw format, size, queue capacity, model files, NPU masks,
worker counts, OpenCV thread setting and complete-pipeline fallback behavior.

| Variant | Frame preprocessing | Normalization | RKNN input |
| --- | --- | --- | --- |
| CPU + scalar | mapped capture buffer -> explicit raw copy -> OpenCV | original scalar formula | host input submission |
| DMA/RGA + scalar | capture FD -> RGA DMA images | original scalar formula | external FD/native binding |
| DMA/RGA + LUT | capture FD -> RGA DMA images | exact precomputed FP16 values | external FD/native binding |

```sh
python3 -u tools/compare_camera.py --duration 12 --rounds 3 \
  --output results/camera_comparison_20261009
```

The order rotates ABC/BCA/CAB. CPU percentage is process CPU, with 100% representing
one core. The script retains each raw report and resource samples. Live frames are not
bit-identical across runs; keep the scene fixed and use these as short, repeated system
comparisons, not a labeled accuracy experiment. Interpolation/color conversion can also
differ slightly between OpenCV and RGA. Scalar vs LUT is tested independently for exact bits.

`explicit_cpu_copy_bytes` counts only instrumented application copies: capture raw memcpy,
CPU letterbox copy, and the old optional input-memory memcpy. It excludes OpenCV's internal
work, kernel USB handling and proprietary SDK internal transfers. Normalization output
writes are reported separately and are not called copies. FD sharing still incurs hardware
memory traffic. Source dequeue time excludes exposure and USB transfer.

Example full camera command:

```sh
./build/yolo_clip_system_demo \
  --yolo yolo_waste_single_class.rknn --clip clip_image_encoder_vitl14.rknn \
  --label-tsv board_labels_vitl14.tsv --text-features board_text_features_vitl14.f32 \
  --camera /dev/v4l/by-path/platform-xhci-hcd.12.auto-usb-0:1.4:1.0-video-index0 \
  --camera-path dma --normalization lut --duration 12 --fps 30 \
  --camera-width 640 --camera-height 480 --camera-buffers 12 \
  --yolo-threads 2 --clip-threads 4 --clip-core-mode round_robin \
  --opencv-threads 1 --fallback-full-image
```

`--camera-path cpu` selects the comparison path; camera DMA remains the default.
`--normalization scalar|lut|scalar_legacy|lut_legacy` works for camera and image/video
modes; `lut` is the default. Current scalar/LUT use IEEE binary16 round-to-nearest
ties-to-even. The legacy options intentionally retain the historical exponent-carry bug
for reproducing old measurements, not for deployment. Both formula sets together occupy
6 KiB (3 KiB for current tables and 3 KiB for legacy tables). No fast-math is used.
See `docs/OPTIMIZATION_20261010.md` for the correction and independent reference tests.

## NPU scheduling experiments

`--clip-core-mode all|round_robin|auto|spread|split12|core0|core1|core2`
and `--yolo-core-mode legacy|auto|all|adaptive|core0|core1|core2` control RKNN
core masks, not CPU affinity. `spread` assigns CLIP workers to cores 0/1/2;
`split12` alternates cores 1/2. `round_robin` preserves the old 2/0/1/2 assignment.
`--npu-gate none|per_core` optionally prevents this process from issuing overlapping
requests on the same cores. It is application admission control, not a kernel scheduler.

Adaptive YOLO requires `--npu-gate per_core`: it acquires one available core and changes
the context mask before inference. Pending adaptive requests get admission precedence
over static requests; this reduces a source of starvation but is not a real-time guarantee.
Reported model `run_ms` includes gate waiting; gate wait is also reported separately.
The old worker counts/masks and `--npu-gate none` remain defaults.

```sh
python3 -u tools/confirm_adaptive_npu.py \
  --output results/npu_adaptive_ieee_20261010 --duration 12 --rounds 3
```

The three complete-pipeline configurations are compared on fixed images in rotating
order. This is not a new live-camera comparison. CLIP-only capacity can be measured
separately with `tools/sweep_npu_scheduling.py`; that script's full sweep also runs
pipeline screening/confirmation. Do not interpret aggregate output interval as single
request latency, or assume that three contexts deliver ideal linear scaling.

## Isolated model quantization

The historical checkpoint/ONNX was initially unavailable on the board. The original
project ONNX was subsequently recovered on the remote Windows machine. On 200 fixed
inputs, its normalized floating-point outputs match the downloaded official ViT-L/14
reference to numerical precision; their raw output norms differ because the original
export includes output normalization. See `docs/MODEL_PROVENANCE.md` and
`docs/REAL_DATA_EVALUATION_20261011.md` for fingerprints and matched-contract comparisons.
Toolkit/Runtime 2.3.2 candidates are tested with a process-local Runtime, leaving the
installed 2.3.0 library and original model files intact. No INT8 candidate is enabled
by default; speed alone is insufficient when accuracy collapses.

Results: `docs/OPTIMIZATION_20261010.md`. Artifacts and reproduction commands:
`quantization/README.md`.
Follow-up layer diagnosis, user-provided article review and rejected manual hybrid candidate:
`docs/QUANT_DIAGNOSIS_20261010.md`. Its 7.3 FPS result is CLIP-only and fails accuracy,
not a supported full-pipeline performance claim.

## CPU profiling and regression

```sh
./build/bench_normalization scalar 500
./build/bench_normalization lut 500
python3 -u tools/regression.py
```

The regression script checks current scalar/LUT, host/native input, multiple contexts
and serial inference against the corrected scalar reference. Legacy variants are checked
separately against the retained old executable. Current and legacy checksums need not match.

The board has gprof but no installed perf executable. For an isolated CPU profile:

```sh
cmake -S . -B build-profile -DBUILD_HARDWARE_DEMO=OFF \
  -DBUILD_TESTING=OFF -DENABLE_GPROF=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-profile -j2
cd build-profile
./bench_normalization scalar 2000
gprof ./bench_normalization gmon.out
```

Profiled timings are not used as uninstrumented benchmark timings. Isolated normalization
CPU time is not NPU time or complete pipeline speedup. Named scalar/LUT functions are
kept visible for profiling. Detailed results are in `docs/RESULTS.md` after measurement.

## Model provenance

`model_inventory` queries actual logical/native tensor layouts and Runtime/driver versions.
Deployment fingerprints and unknown historical conversion fields are documented in
`docs/MODEL_PROVENANCE.md`. Missing original training/export/conversion artifacts cannot
be reconstructed from `.rknn` alone; they are not fabricated as completed work.
