# Three optimization priorities: measured results (2026-10-09)

All code and measurements are in the isolated `/home/orangepi/yolo_clip_bench_systemopt`
deployment. The original project and the previously verified camera executable remain intact.
Only short correctness and comparison runs were performed; no long stability campaign was added.

Version note (2026-10-10): the following camera/CPU measurements were taken before the
FP16 exponent-carry correction, using the historical converter. They remain historical
evidence, not fresh benchmarks of the corrected default. Current scalar/LUT use IEEE
rounding; `scalar_legacy`/`lut_legacy` reproduce the old behavior. Old scalar/LUT equality
proved mutual consistency, not IEEE correctness. Independent IEEE tests now cover it.
Current scheduling and quantization results are in `OPTIMIZATION_20261010.md`.

## 1. Controlled camera CPU / DMA-RGA / LUT comparison

USB FHD Camesa, YUYV 640x480, driver-requested 30 FPS, actual delivery approximately 10 FPS.
All configurations use the same model files, YOLO workers 2, CLIP workers 4, round_robin
NPU masks, frame queue 4, CLIP queue/pool 8, camera buffers 12, affinity none, OpenCV threads 1,
and full-image fallback so every completed frame executes both models. Each variant was run
three times for 12 seconds with order ABC/BCA/CAB. All nine runs completed successfully.

| Mean across three runs | CPU + scalar | DMA/RGA + scalar | DMA/RGA + LUT |
| --- | ---: | ---: | ---: |
| CPU (%; 100% = one core) | 32.07 | 16.52 | 11.90 |
| YOLO preprocessing (ms) | 19.02 | 19.49 | 10.56 |
| CLIP preprocessing (ms) | 7.12 | 7.87 | 2.02 |
| Completed throughput (FPS, includes drain) | 3.465 | 3.424 | 3.425 |
| Mean of per-run P95 latency (ms) | 2693.77 | 2598.10 | 2601.89 |
| Explicit application copy bytes per run | 77,312,000 | 0 | 0 |

Observed CPU reduction: DMA/RGA vs CPU approximately 48.5%; DMA/RGA+LUT vs CPU
approximately 62.9%; LUT vs the same DMA/RGA scalar path approximately 28.0%.
Within the DMA path, YOLO preprocessing decreased approximately 45.8% and CLIP preprocessing
approximately 74.3%. These are this short experiment's sample means, not universal guarantees.

Throughput did not improve materially. It stayed around 3.4 FPS because NPU model execution
and shared-core scheduling dominate. DMA/RGA scalar preprocessing was not faster than CPU
preprocessing in this experiment; its measured benefit was reduced CPU occupancy/copying.
Do not write a claimed FPS acceleration or an NPU speedup based on these data.

The copy ledger counts capture raw memcpy and CPU letterbox copy, plus optional explicit
input memcpy. It does not count kernel USB traffic, OpenCV internal work, SDK-internal
transfers or hardware memory traffic. FP16 normalization writes remain and are reported
separately. A zero copy-ledger value does not mean zero memory writes or zero CPU use.

Live frames differ across runs, and no labeled accuracy set was used. In these runs YOLO
did not report a target, so the actual camera path used full-image CLIP fallback. Synthetic
RGA ROI self-tests and the existing image regression provide separate ROI coverage.
Latency starts after V4L2 dequeue, excluding exposure/USB transfer; model loading is excluded.
Reported P95 column is the arithmetic mean of three per-run P95 values, not pooled P95.

Raw reports and 250ms process resource samples:
`results/camera_comparison_20261009/01_cpu_scalar.json` through `09_dma_scalar.json`.
Summary: `results/camera_comparison_20261009/summary.json`.

## 2. CPU normalization profiling and exact LUT optimization

The original per-pixel division, per-channel normalization and FP32-to-FP16 bit conversion
were extracted into an independently testable scalar reference. LUT precomputes the exact
results for 256 byte inputs per channel. The measured legacy tables occupied 3 KiB total.
The current implementation retains both corrected and legacy tables, totaling 6 KiB.
No new model, quantization or fast-math was used in this historical experiment; the later
IEEE correction is described separately and is not retroactively claimed in these timings.

Uninstrumented benchmark: GCC 11.4, -O3, 20 warmup iterations, 500 measured iterations,
three repeats per implementation/core with order alternation, no concurrent compile/inference.
CPU0 is Cortex-A55; CPU4 is Cortex-A76. Governor remained ondemand; no system governor was changed.

| Core | Operation | Scalar (ms) | LUT (ms) | Kernel speed ratio |
| --- | --- | ---: | ---: | ---: |
| A55 / CPU0 | YOLO RGB640 -> FP16 | 21.746 | 2.991 | 7.27x |
| A55 / CPU0 | CLIP RGB224 -> normalized FP16 | 4.068 | 0.372 | 10.93x |
| A76 / CPU4 | YOLO RGB640 -> FP16 | 4.380 | 0.932 | 4.70x |
| A76 / CPU4 | CLIP RGB224 -> normalized FP16 | 0.690 | 0.111 | 6.19x |

These ratios apply to the isolated normalization kernel, not NPU or complete pipeline inference.
Checksums matched for every scalar/LUT/core run. Tests cover all 256 values and all channels,
both normalization formulas, padded rows and output tails.

The board lacked a perf executable, so gprof was used for the isolated CPU-only workload.
Its instrumented scalar profile recorded 10.25 seconds in `normalization::Scalar`, 4040 calls,
including warmups. Instrumented timing is not used in the table. This profile does not describe
the proprietary RKNN kernels or attribute the entire system's CPU time.

Raw evidence: `results/debug_profiles/normalization.json`, `gprof_scalar.txt`, and the per-core
benchmark text files. `--normalization scalar` remains available; verified LUT is the new default.

## 3. Modules, ownership fix, regression and debugging

Code was separated into V4L2 capture, DMA memory, RGA processing, normalization, model runners,
text bank, queues/pool, scheduling and metrics, with a CMake build and CTest.

The original queue returned immediately when closed. A producer transferred only an integer
slot id and depended on explicit release callbacks, so a rejected enqueue could lose the slot's
return obligation. A preprocessing exception before enqueue had the same ownership weakness.
This is a logical pool-slot leak on error/shutdown paths, not evidence of a long-running heap leak.

The new `BufferPool::Lease` is move-only and holds shared pool state. Scope exit, rejected enqueue,
oldest-task drop, producer/consumer exception, and queue destruction all return the slot once.
Drop callbacks and destruction of replaced tasks are kept outside the queue lock, avoiding
camera QBUF operations under that lock. Normal shutdown verifies all CLIP slots were returned.

Verification completed:

- CMake RelWithDebInfo build, -O3 -g, -Wall -Wextra: passed without warnings.
- CTest normal build: both original test programs passed at the time of this experiment.
- CTest Debug + ASan/UBSan: both original programs passed, including 4,000 concurrent tasks.
- Exact five-image comparison against the retained original camera executable: scalar host,
  LUT host, scalar DMA, LUT DMA, LUT multi-context and LUT serial all matched predictions and
  `checksum_xor=18313580699023835136`.
- Real camera comparison: all nine short runs passed and every run's frame accounting matched.
- GDB on isolated original queue reproduction: exception stack led to the rejected closed enqueue;
  `free_slots` was 0 when it should be 1. This is an intentional regression reproduction, not
  a claim of a production segmentation fault.
- GDB on fixed `TestClosedQueueOwnership`: stopped at the regression seam and subsequently
  completed the queue/pool suite normally.

GDB transcripts: `results/debug_profiles/gdb_legacy.txt`, `gdb_fixed.txt`.
`results/regression.json` now contains the updated eight-variant IEEE/legacy regression,
not the old six-variant report. The historical checksum above applies only to legacy inputs.
The current normal and ASan/UBSan CTest suites have three passing programs, including
the new NPU arbiter tests.

Deployment/model inventory is in `docs/MODEL_PROVENANCE.md` and
`results/debug_profiles/model_inventory.json`. Historical training/export/conversion scripts
were not present, so those fields remain explicitly unknown rather than invented.

## Resume statement supported by these measurements

Implemented V4L2 camera capture and DMA-BUF sharing with RGA/RKNN on RK3588; built a
multi-context YOLO-CLIP pipeline with bounded queues and RAII buffer leases. In a repeated
same-camera short comparison of the historical input converter, DMA/RGA plus a bit-matched
normalization LUT reduced mean
process CPU usage from 32.1% to 11.9%, while complete inference throughput remained about
3.4 FPS. Added CMake/CTest, sanitizer checks and GDB-backed shutdown ownership regression.
