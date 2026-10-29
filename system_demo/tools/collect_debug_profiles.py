#!/usr/bin/env python3
"""Generate CPU-only profiles, real GDB traces, and queried model fingerprints."""
import hashlib
import json
from pathlib import Path
import subprocess

output = Path("results/debug_profiles")
output.mkdir(parents=True, exist_ok=True)


def run(command, name, expected=0):
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    (output / name).write_text("COMMAND: " + " ".join(command) + "\nEXIT: " + str(result.returncode) +
                               "\n" + result.stdout + result.stderr)
    if result.returncode != expected:
        raise RuntimeError(f"{name}: expected {expected}, got {result.returncode}: {result.stderr}")
    return result.stdout


run(["./build-sanitize/legacy_slot_leak_repro"], "legacy_reproduction.txt", expected=1)
run(["gdb", "-q", "-batch", "-ex", "catch throw", "-ex", "run", "-ex", "bt 6",
     "-ex", "frame 1", "-ex", "print free_slots", "--args", "./build-sanitize/legacy_slot_leak_repro"],
    "gdb_legacy.txt")
run(["gdb", "-q", "-batch", "-ex", "break TestClosedQueueOwnership", "-ex", "run",
     "-ex", "bt 4", "-ex", "continue", "--args", "./build/test_queue_pool"], "gdb_fixed.txt")

benchmarks = []
for core in (0, 4):
    for repeat in range(3):
        order = ("scalar", "lut") if repeat % 2 == 0 else ("lut", "scalar")
        for mode in order:
            values = json.loads(run(["taskset", "-c", str(core), "./build/bench_normalization", mode, "500"],
                                    f"normalization_core{core}_{repeat}_{mode}.txt"))
            for value in values:
                value.update(core=core, repeat=repeat)
                benchmarks.append(value)
(output / "normalization.json").write_text(json.dumps(benchmarks, indent=2))

run(["cmake", "-S", ".", "-B", "build-profile", "-DBUILD_HARDWARE_DEMO=OFF",
     "-DBUILD_TESTING=OFF", "-DENABLE_GPROF=ON", "-DCMAKE_BUILD_TYPE=Release"], "configure_profile.txt")
run(["cmake", "--build", "build-profile", "-j2"], "build_profile.txt")
profile = subprocess.run(["taskset", "-c", "4", "./bench_normalization", "scalar", "2000"],
                         cwd="build-profile", text=True, capture_output=True, timeout=120)
if profile.returncode: raise RuntimeError(profile.stderr)
(output / "profile_run.txt").write_text(profile.stdout + profile.stderr)
run(["gprof", "-b", "./build-profile/bench_normalization", "./build-profile/gmon.out"], "gprof_scalar.txt")

inventory = json.loads(run(["./build/model_inventory", "yolo_waste_single_class.rknn",
                           "clip_image_encoder_vitl14.rknn"], "model_inventory_raw.txt"))
for item, filename in zip(inventory, ("yolo_waste_single_class.rknn", "clip_image_encoder_vitl14.rknn")):
    digest = hashlib.sha256()
    with open(filename, "rb") as model:
        for block in iter(lambda: model.read(1024 * 1024), b""): digest.update(block)
    item.update(filename=filename, sha256=digest.hexdigest(), original_onnx=None,
                training_version=None, historical_conversion_config=None)
(output / "model_inventory.json").write_text(json.dumps(inventory, indent=2))
print("PASS: GDB legacy/fixed traces, pinned-core benchmarks, gprof, model inventory", flush=True)
