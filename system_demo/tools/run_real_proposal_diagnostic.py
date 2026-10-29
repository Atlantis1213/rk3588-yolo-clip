"""Auxiliary known-presence ROI classification; never relabel it as online detection accuracy."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_method_sweep_board import digest

root = Path(__file__).resolve().parents[1]
base = root / "results/real_eval_200_20261011"
output = base / "proposal_diagnostic"
output.mkdir(exist_ok=False)
(output / "dataset").mkdir()
(output / "dataset/manifest.json").write_bytes((base / "dataset/manifest.json").read_bytes())
environment = dict(os.environ, LD_LIBRARY_PATH=str(root / "quant_runtime232"))
lock = (root / ".method_sweep_npu.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
models = {"deployed_historical": root / "clip_image_encoder_vitl14.rknn",
    "official_fp16": root / "clip_reference_fp16_232.rknn",
    "official_kl120": root / "quantization/artifacts/method_sweep_20261011/kl120_reference/clip.rknn"}
hashes = {name: digest(path) for name, path in models.items()}
info = {"completed": False, "model_sha": hashes, "variants": [], "confidence_threshold": 0.0,
    "protocol": "auxiliary YOLO top1 proposal/CLIP classification, known empty bypass; not online system accuracy",
    "primary_005_threshold_evaluation_preserved": True}


def status(phase, **details):
    info.update(phase=phase, **details)
    (output / "board_status.json").write_text(json.dumps(info, indent=2))
    print("PROPOSAL DIAGNOSTIC " + phase, flush=True)


def run(command, log, timeout):
    result = subprocess.run(list(map(str, command)), env=environment, cwd=root,
        capture_output=True, text=True, timeout=timeout)
    log.write_text(result.stdout + result.stderr)
    result.check_returncode()
    print(result.stdout, flush=True)


try:
    status("freeze_top1_proposals")
    run([root / "build/prepare_real_clip_inputs", root / "yolo_waste_single_class.rknn", base / "dataset",
        output / "frozen", 0.0], output / "prepare.log", 180)
    manifest = json.loads((output / "frozen/manifest.json").read_text())
    (output / "frozen/input_sha256.json").write_text(json.dumps({row["image"]:
        digest(output / "frozen/inputs_fp16" / (row["image"] + ".bin")) for row in manifest["samples"]}, indent=2))
    info["valid_nonempty_proposal_count"] = manifest["detected_nonempty"]
    info["nonempty_proposals_above_original_005"] = sum(row["detected"] and not row["skip_yolo"]
        and row["max_confidence"] >= .05 for row in manifest["samples"])
    for name, model in models.items():
        status("inference_" + name)
        run([root / "build/dump_frozen_clip_inputs", model, base / "dataset", output / "frozen", output / name],
            output / (name + ".log"), 600)
        if digest(model) != hashes[name]:
            raise RuntimeError("model changed during diagnostic")
        info["variants"].append(name)
    status("completed", completed=True)
except Exception as error:
    status("failed", error=repr(error))
    raise
