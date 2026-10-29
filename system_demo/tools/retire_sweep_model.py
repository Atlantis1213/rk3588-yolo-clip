"""Remove only a verified temporary candidate copy, leaving evidence/defaults intact."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_method_sweep_board import digest

parser = argparse.ArgumentParser()
parser.add_argument("--variant", required=True)
parser.add_argument("--sha256", required=True)
args = parser.parse_args()
allowed = {"normal30", "kl30", "mmse30", "repq_ln2", "repq_all",
    "smooth_ln2_a25", "smooth_ln2_a50", "smooth_ln2_a75", "smooth_all_a50",
    "bias_down", "adaround_down", "adaround_bias_down"}
if args.variant not in allowed:
    raise ValueError("only this sweep's candidate copies may be retired")
root = Path(__file__).resolve().parents[1]
lock = (root / ".method_sweep_npu.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
processes = subprocess.run(["ps", "-eo", "comm"], capture_output=True, text=True, check=True).stdout
if any(name.startswith(("dump_clip", "dump_prefix", "bench_clip", "yolo_clip_syste")) for name in processes.splitlines()):
    raise RuntimeError("NPU workload still active; keep its model")
directory = root / "quantization/artifacts/method_sweep_20261011"
if args.variant.startswith(("repq_", "smooth_")):
    directory /= "affine_rknn"
elif args.variant in {"bias_down", "adaround_down", "adaround_bias_down"}:
    directory /= "reconstruction_rknn"
directory /= args.variant
model = (directory / "clip.rknn").resolve(strict=True)
if model.parent != directory.resolve(strict=True) or model.is_symlink():
    raise RuntimeError("candidate path is not the expected regular temporary file")
validation = json.loads((root / "results/method_sweep_20261011/development" / args.variant / "status.json").read_text())
build = json.loads((directory / "status.json").read_text())
if (not validation["completed"] or not build["completed"]
    or validation["model_sha"] != args.sha256 or build["model_sha"] != args.sha256
    or digest(model) != args.sha256):
    raise RuntimeError("candidate SHA/validation gate failed; refuse removal")
model.unlink()
(directory / "temporary_copy_retired.json").write_text(json.dumps({"model_sha": args.sha256,
    "removed": str(model), "local_full_model_verified_by_caller": True,
    "evidence_and_build_status_preserved": True}, indent=2))
print("Retired verified temporary candidate " + args.variant, flush=True)
