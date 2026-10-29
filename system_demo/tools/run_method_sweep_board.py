"""Serial, hash-verified development evaluation; the locked test is never read."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def operator_signature(detail):
    return Counter((parts[1], parts[2], parts[3], parts[-1])
        for line in detail.splitlines()
        if re.match(r"^\s*\d+\s+\w+\s+\w+\s+(CPU|NPU|GPU)\s", line)
        for parts in [line.split()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "speed"], required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--speed-label")
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    allowed = {"fp16", "kl120", "normal30", "kl30", "mmse30", "repq_ln2", "repq_all",
        "smooth_ln2_a25", "smooth_ln2_a50", "smooth_ln2_a75", "smooth_all_a50",
        "bias_down", "adaround_down", "adaround_bias_down"}
    if not set(args.variants) <= allowed or len(set(args.variants)) != len(args.variants):
        raise ValueError("unknown or duplicate variant")
    if args.seconds <= 0 or args.repeats <= 0:
        raise ValueError("positive duration/repeats required")
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "quantization/artifacts/method_sweep_20261011"
    results = root / "results/method_sweep_20261011"
    environment = dict(os.environ, LD_LIBRARY_PATH=str(root / "quant_runtime232"))
    lock = (root / ".method_sweep_npu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Other historical runners do not share this lock. Reject their active binaries too.
    processes = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True, check=True).stdout
    if any(name.startswith(("dump_clip", "dump_prefix", "bench_clip", "yolo_clip_syste"))
           for name in processes.splitlines()):
        raise RuntimeError("another NPU workload is active; keep measurements serial")

    def model_info(name):
        if name == "fp16":
            model = root / "clip_reference_fp16_232.rknn"
            info = {"model_sha": ""}
        else:
            directory = artifacts / ("kl120_reference" if name == "kl120" else name)
            if name.startswith(("repq_", "smooth_")):
                directory = artifacts / "affine_rknn" / name
            elif name in {"bias_down", "adaround_down", "adaround_bias_down"}:
                directory = artifacts / "reconstruction_rknn" / name
            model = directory / "clip.rknn"
            info = json.loads((directory / "status.json").read_text())
            if not info.get("completed") or info.get("phase") != "completed":
                raise RuntimeError("model build incomplete: " + name)
        actual = digest(model)
        if info["model_sha"] and info["model_sha"] != actual:
            raise RuntimeError("deployed model SHA mismatch: " + name)
        return model, actual

    def run(command, log, timeout):
        result = subprocess.run(list(map(str, command)), cwd=root, env=environment,
            capture_output=True, text=True, timeout=timeout)
        log.write_text(result.stdout + "\n" + result.stderr)
        result.check_returncode()
        return result

    if args.phase == "validate":
        protocol = artifacts / "protocol_v2"
        verification = json.loads((protocol / "verification.json").read_text())
        if (not verification["source_split_disjoint"] or not verification["generated_file_hashes_valid"]
            or verification["exact_resized_pixel_duplicate_count"] != 0):
            raise RuntimeError("data protocol verification gate failed")
        labels = json.loads((protocol / "development_labels.json").read_text())
        samples = labels["samples"]
        if len(samples) != 144 or Counter(row["class"] for row in samples) != dict.fromkeys(labels["classes"], 24):
            raise RuntimeError("expected frozen balanced development split")
        images = protocol / "development"
        if {path.name for path in images.glob("*.png")} != {row["image"] for row in samples}:
            raise RuntimeError("development sample membership differs")
        for row in samples:
            if digest(images / row["image"]) != row["file_sha256"]:
                raise RuntimeError("development image SHA differs: " + row["image"])
        input_file = root / "results/quant_public_fp16_ieee/inputs_fp16/cardboard_020.png.bin"
        for name in args.variants:
            model, model_sha = model_info(name)
            output = results / "development" / name
            output.mkdir(parents=True, exist_ok=False)
            status = {"phase": "started", "completed": False, "model_sha": model_sha,
                "variant": name, "count": 144, "locked_test_evaluated": False,
                "labels_sha": digest(protocol / "development_labels.json")}
            def update(phase, **details):
                status.update(phase=phase, **details)
                (output / "status.json").write_text(json.dumps(status, indent=2))
                print(name + " " + phase, flush=True)
            try:
                update("inventory")
                run([root / "build/dump_prefix_outputs", model, output / "inventory", "--perf", input_file],
                    output / "inventory.log", 90)
                detail = (output / "inventory/perf_detail.txt").read_text()
                attention = [line for line in detail.splitlines()
                    if re.match(r"^\s*\d+\s+exSDPAttention\s", line)]
                dtype = "FLOAT16" if name == "fp16" else "INT8"
                if len(attention) != 24 or not all(re.search(r"exSDPAttention\s+" + dtype + r"\s+NPU", line)
                                                  for line in attention):
                    raise RuntimeError("actual SDPA precision/target differs from safe route")
                reference = results / "development/kl120/inventory/perf_detail.txt"
                precision_equal = None
                if name not in {"fp16", "kl120"}:
                    if not reference.exists():
                        raise RuntimeError("validate KL120 first for compiled precision comparison")
                    actual, expected = operator_signature(detail), operator_signature(reference.read_text())
                    if not actual or actual != expected:
                        raise RuntimeError("compiled precision/operator graph differs: " + str({
                            "added": list((actual - expected).items())[:8],
                            "removed": list((expected - actual).items())[:8]}))
                    precision_equal = True
                update("inference", sdpa_npu_count=24, sdpa_dtype=dtype,
                    compiled_precision_equal_to_kl120=precision_equal)
                run([root / "build/dump_clip_embeddings", model, images, output / "dump", "host"],
                    output / "inference.log", 480)
                manifest = json.loads((output / "dump/manifest.json").read_text())
                names = [row["image"] for row in manifest["samples"]]
                if manifest["embedding_dim"] != 768 or len(names) != 144 or set(names) != {row["image"] for row in samples}:
                    raise RuntimeError("incomplete or mismatched development outputs")
                if name != "fp16":
                    reference_dump = results / "development/fp16/dump"
                    reference_manifest = json.loads((reference_dump / "manifest.json").read_text())
                    if names != [row["image"] for row in reference_manifest["samples"]]:
                        raise RuntimeError("development sample order differs from FP16")
                    comparison = subprocess.run(["diff", "-qr", str(reference_dump / "inputs_fp16"),
                        str(output / "dump/inputs_fp16")], capture_output=True, text=True, timeout=60)
                    (output / "input_diff.log").write_text(comparison.stdout + comparison.stderr)
                    comparison.check_returncode()
                if digest(model) != model_sha:
                    raise RuntimeError("model changed during evaluation")
                update("completed", completed=True, inputs_bitwise_equal_to_fp16=name != "fp16")
            except Exception as error:
                update("failed", error=repr(error))
                raise
    else:
        if not args.speed_label or not re.fullmatch(r"[a-z0-9_]+", args.speed_label):
            raise ValueError("new safe speed label required")
        output = results / "speed" / args.speed_label
        output.mkdir(parents=True, exist_ok=False)
        configs = []
        for name in args.variants:
            model, model_sha = model_info(name)
            validation = json.loads((results / "development" / name / "status.json").read_text())
            if not validation["completed"] or validation["model_sha"] != model_sha:
                raise RuntimeError("timing model must match completed validation")
            configs += [(name, model, model_sha, 1, "all"), (name, model, model_sha, 3, "spread")]
        input_file = results / "development/fp16/dump/inputs_fp16/v2_cardboard_000.png.bin"
        rows = []
        for repeat in range(args.repeats):
            for name, model, model_sha, workers, mode in (configs if repeat % 2 == 0 else list(reversed(configs))):
                print(f"SPEED {name} {workers} {repeat}", flush=True)
                result = run([root / "build/bench_clip_concurrency", model, input_file, workers, mode,
                    args.seconds, "none", "host"], output / f"{repeat}_{name}_{workers}.log", 90)
                report = json.loads(result.stdout[result.stdout.index("{"):])
                if digest(model) != model_sha:
                    raise RuntimeError("model changed during timing")
                rows.append({"variant": name, "repeat": repeat, "model_sha": model_sha, "report": report})
                (output / "speed_summary.json").write_text(json.dumps({"scope": "CLIP-only aggregate throughput; "
                    "not single-image latency or end-to-end YOLO/RGA/DMA throughput", "rows": rows}, indent=2))
                print(f"DONE {name} {workers}: {report['fps']:.3f} FPS", flush=True)


if __name__ == "__main__":
    main()
