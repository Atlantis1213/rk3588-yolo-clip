# Explainable implementation and debugging evidence

## Closed-queue ownership case

Original behavior: input slot acquired -> preprocessing -> integer slot id submitted to queue.
When the queue had already closed, Push returned without on_drop, so explicit Release was missed.
The old acquisition contract also needed manual cleanup if preprocessing threw.

Reproduction and real GDB commands:

```sh
./build-sanitize/legacy_slot_leak_repro  # intentional exit 1
gdb -q -batch -ex 'catch throw' -ex run -ex 'bt 6' \
  -ex 'frame 1' -ex 'print free_slots' --args ./build-sanitize/legacy_slot_leak_repro
```

Observed stack: __cxa_throw -> ReproduceOriginalClosedQueueLeak -> main; free_slots=0.
The reproduction copies the original queue implementation and the old manual-id ownership
pattern. It tests the real rejection seam, but is not a production crash or heap-leak report.

Fix: represent slot ownership with a move-only Lease. Move transfers responsibility;
destruction returns the slot; shared state keeps slots/mutex alive even if the pool wrapper
is destroyed earlier. The free-index vector reserves its full capacity once so releasing
a valid lease does not allocate. Queue callbacks and old-task destruction occur outside
the queue mutex. Close wakes waiting consumers, while already queued tasks drain normally.

```sh
gdb -q -batch -ex 'break TestClosedQueueOwnership' -ex run -ex 'bt 4' \
  -ex continue --args ./build/test_queue_pool
```

The fixed suite passed, including concurrent producers/consumers and exception ownership cases.
ASan/UBSan passed separately outside GDB. Shared pool state does not form a shared_ptr cycle:
state owns slots; leases own state; state does not own leases.

## LUT case

The inputs to normalization are RGB bytes. Each channel has exactly 256 possible values.
Precompute the normalization formula and FP16 bit representation for each value, then replace
per-pixel arithmetic and bit conversion with three indexed loads and output stores.
All 256 values/channel and padded-stride/tail cases are checked. Corrected YOLO/CLIP tables
are 3 KiB; retaining the legacy tables makes the current total 6 KiB. NEON was not added;
this optimization is LUT based.

Important regression: the original FP16 converter mishandled mantissa overflow. CLIP's
green=250 gives approximately 1.9998449, but the old converter produced 0x3c00 (1.0),
instead of 0x4000 (2.0). A LUT that agrees with a faulty scalar implementation preserves
the bug. The current converter implements IEEE round-to-nearest ties-to-even and is
checked against compiler `__fp16`/`_Float16`, including exponent carry, ties, subnormals,
overflow and signed zero. Legacy modes exist only for historical comparisons.

Explain CPU microbenchmark vs pipeline results separately: a normalization kernel can become
4.7x faster on A76 without changing NPU-bound complete inference FPS. RGA and DMA primarily
reduce CPU work/copies here. Avoid describing zero application memcpy as zero memory traffic.

## Data ownership to draw in an interview

```text
V4L2 DQBUF -> Frame retains capture object -> synchronous RGA use -> Frame destruction/QBUF
CPU normalization -> DMA tensor Lease -> move into ClipTask -> queue -> CLIP worker -> Lease reset
```

The camera and tensor buffers use different FDs. Native FP16 NHWC input is checked before
pass-through binding. No CPU pixel upload is needed on the DMA camera path; normalization
still writes a new tensor. CPU baseline maps the same capture buffers, copies raw bytes,
uses OpenCV and host input submission. Native import wrappers retain their DMA backing
buffers until after inference and are destroyed before the RKNN context.

## Technical boundaries

- Worker pools are fixed stage workers, not work-stealing or RTOS priority scheduling.
- CPU affinity is distinct from NPU core masks and thread scheduling priority.
- `round_robin` and `spread` are fixed per-worker assignments; adaptive YOLO changes its
  mask after acquiring a core through the application arbiter. This is not NPU preemption.
- Camera dequeue latency excludes exposure and USB transfer.
- The build on Orange Pi uses native g++, not a completed cross-compile deployment.
- Gprof evidence is an isolated CPU kernel profile, not proprietary NPU profiling.
- Historical model conversion/training provenance is incomplete and recorded as unknown.
- Real-camera comparisons used full-image fallback; no labeled accuracy claim is supported.
- Public TrashNet six-class quantization accuracy is distinct from original scene accuracy.
- Quantization candidates currently fail accuracy acceptance; neither INT8 nor batch=3
  is integrated into the default zero-copy camera pipeline.
- Native INT8 DMA binding is diagnostic only and has not demonstrated consistent output
  equivalence to the SDK host path. The default pipeline rejects this input type.
