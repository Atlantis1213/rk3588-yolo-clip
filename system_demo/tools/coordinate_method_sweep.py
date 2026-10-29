"""Local board coordinator: one candidate slot, complete metrics, then retirement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import re

root = Path(__file__).resolve().parents[1]
quant = root / "quantization"
sys.path.insert(0, str(quant))
from evaluate_protocol_v2 import load_dump, compare
import numpy as np

REMOTE = "orangepi@192.168.123.99"
BOARD = "/home/orangepi/yolo_clip_bench_systemopt"
SWEEP = "quantization/artifacts/method_sweep_20261011"
RESULTS = "results/method_sweep_20261011"
PROFILES = ["normal30", "kl30", "repq_ln2", "smooth_ln2_a25", "smooth_ln2_a50",
    "smooth_ln2_a75", "repq_all", "smooth_all_a50"]
RECONSTRUCTION = ["bias_down", "adaround_down", "adaround_bias_down"]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def command(args, timeout=600):
    result = subprocess.run(list(map(str, args)), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        result.check_returncode()
    return result


def remote(script, timeout=600):
    return command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=4", REMOTE, script], timeout)


def source_dir(profile):
    if profile in RECONSTRUCTION:
        return quant / "artifacts/method_sweep_20261011/reconstruction_rknn" / profile
    return quant / "artifacts/method_sweep_20261011" / ("affine_rknn/" + profile
        if profile.startswith(("repq_", "smooth_")) else profile)


def retire(profile):
    source = source_dir(profile)
    build = json.loads((source / "status.json").read_text())
    if not build["completed"] or sha(source / "clip.rknn") != build["model_sha"]:
        raise RuntimeError("full local model must be intact before retiring board copy")
    local = root / RESULTS / "development" / profile
    validation = json.loads((local / "status.json").read_text())
    if not validation["completed"] or validation["model_sha"] != build["model_sha"]:
        raise RuntimeError("local completed validation evidence must be retained")
    remote(f"cd {BOARD} && python3 tools/retire_sweep_model.py --variant {profile} --sha256 {build['model_sha']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retire-mmse30-first", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES + RECONSTRUCTION, default=PROFILES)
    parser.add_argument("--run-directory", default="coordinator_20261011")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.run_directory):
        raise ValueError("safe new coordinator directory required")
    output = root / RESULTS / args.run_directory
    output.mkdir(parents=True, exist_ok=False)
    info = {"phase": "started", "completed": False, "jobs": [], "locked_test_evaluated": False}
    def status(phase, **values):
        info.update(phase=phase, **values)
        (output / "status.json").write_text(json.dumps(info, indent=2))
        print("BOARD COORDINATOR " + phase, flush=True)
    labels = json.loads((quant / SWEEP.split("quantization/")[1] / "protocol_v2/development_labels.json").read_text())
    old = json.loads((quant / "artifacts/eval_labels.json").read_text())
    if labels["classes"] != old["classes"]:
        raise RuntimeError("class order differs from text bank")
    texts = np.load(quant / "artifacts/public_text_features.npy").astype(np.float64)
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    base = load_dump(root / RESULTS / "development/fp16/dump", labels, texts)
    kl = compare(base, load_dump(root / RESULTS / "development/kl120/dump", labels, texts))
    try:
        command(["scp", "-q", root / "tools/retire_sweep_model.py", root / "tools/run_method_sweep_board.py", f"{REMOTE}:{BOARD}/tools/"])
        if args.retire_mmse30_first:
            if not (root / RESULTS / "speed/mmse30_round1/speed_summary.json").exists():
                raise RuntimeError("MMSE timing evidence must be fetched before retirement")
            status("retire_mmse30")
            retire("mmse30")
        for profile in args.profiles:
            source = source_dir(profile)
            while not (source / "status.json").exists() or not json.loads((source / "status.json").read_text()).get("completed"):
                build_status = json.loads((source / "status.json").read_text()) if (source / "status.json").exists() else {}
                cpu_status = json.loads((quant / "artifacts/method_sweep_20261011/cpu_queue_20261011/status.json").read_text())
                if build_status.get("phase") == "failed" or cpu_status.get("phase") == "failed":
                    raise RuntimeError("CPU queue/build needs inspection: " + profile)
                status("waiting_" + profile)
                time.sleep(30)
            build = json.loads((source / "status.json").read_text())
            if sha(source / "clip.rknn") != build["model_sha"]:
                raise RuntimeError("local model SHA differs from completed build")
            relative = str(source.relative_to(root)).replace("\\", "/")
            status("deploy_" + profile)
            remote(f"test ! -e {BOARD}/{relative} && mkdir -p {BOARD}/{relative}")
            command(["scp", "-q", source / "clip.rknn", source / "status.json", f"{REMOTE}:{BOARD}/{relative}/"])
            status("validate_" + profile)
            remote(f"cd {BOARD} && python3 -u tools/run_method_sweep_board.py --phase validate --variants {profile}")
            command(["scp", "-q", "-r", f"{REMOTE}:{BOARD}/{RESULTS}/development/{profile}",
                root / RESULTS / "development/"])
            candidate = load_dump(root / RESULTS / "development" / profile / "dump", labels, texts)
            metrics = compare(base, candidate)
            metrics["model_sha"] = build["model_sha"]
            metrics["variant"] = profile
            # Always time algorithm controls. Float folds are timed only if classification is no worse than KL.
            timed = profile in {"normal30", "kl30"} or metrics["correct"] >= kl["correct"]
            metrics["speed_measured"] = timed
            metric_file = quant / "artifacts/method_sweep_20261011" / ("accuracy_" + profile + "_v2.json")
            with metric_file.open("x") as handle:
                json.dump({"dataset": "TrashNet development v2, 144 images", "classes": labels["classes"],
                    "locked_test_evaluated": False, "variants": [base[6], kl, metrics]}, handle, indent=2)
            if timed:
                status("speed_" + profile)
                remote(f"cd {BOARD} && python3 -u tools/run_method_sweep_board.py --phase speed "
                    f"--variants fp16 kl120 {profile} --speed-label {profile}_round1")
                command(["scp", "-q", "-r", f"{REMOTE}:{BOARD}/{RESULTS}/speed/{profile}_round1",
                    root / RESULTS / "speed/"])
            status("retire_" + profile)
            retire(profile)
            info["jobs"].append({"variant": profile, "correct": metrics["correct"],
                "accuracy": metrics["accuracy"], "speed_measured": timed, "model_sha": build["model_sha"]})
            status("done_" + profile)
        status("completed", completed=True)
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
