"""Evaluate only the frozen strategy and baselines on a file-disjoint split."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "quantization"))
from partial_mlp_policy import PARTIAL_KL_DOWN

selection = json.loads((root / "quantization/artifacts/partial_kl_selected_20261011.json").read_text())
variant = selection["selected"]["variant"]
if variant not in PARTIAL_KL_DOWN or selection["final_test_used_for_selection"]:
    raise RuntimeError("invalid or non-frozen candidate selection")
validation = json.loads((root / "results/sdpa_safe_full_20261010" / (variant + "_validation_status.json")).read_text())
build = json.loads((root / "quantization/artifacts/sdpa_safe_full_20261010" / variant / "status.json").read_text())
if validation["phase"] != "completed" or validation["model_sha"] != selection["model_sha"] or not build["completed"]:
    raise RuntimeError("chosen strategy is not the validated candidate")
if validation["protected_matrix_blocks"] != list(PARTIAL_KL_DOWN[variant]):
    raise RuntimeError("chosen blocks differ from validated blocks")
models = {"fp16": root / "clip_reference_fp16_232.rknn",
    "kl": root / "quantization/artifacts/partial_kl_20261010/reference/clip.rknn",
    "selected": root / "quantization/artifacts/group_ptq_20261010/active/clip.rknn"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


hashes = {name: sha(path) for name, path in models.items()}
kl_status = json.loads((root / "results/sdpa_safe_full_20261010/sensitive_kl_except_sdpa_validation_status.json").read_text())
if hashes["selected"] != selection["model_sha"] or hashes["selected"] != build["model_sha"]:
    raise RuntimeError("active model differs from frozen candidate")
if hashes["kl"] != kl_status["model_sha"] or kl_status["phase"] != "completed":
    raise RuntimeError("temporary KL reference changed")
images = root / "final_test_png"
if len(list(images.glob("*.png"))) != 120:
    raise RuntimeError("expected exactly 120 final images")
output = root / "results/partial_kl_final_20261011"
output.mkdir(exist_ok=False)
env = dict(os.environ, LD_LIBRARY_PATH=str(root / "quant_runtime232"))
status = {"phase": "started", "completed": False, "variant": variant,
    "model_sha": hashes, "selection_rule": selection["selection_rule"],
    "retune_after_final_test": False}


def write_status(phase):
    status["phase"] = phase
    (output / "status.json").write_text(json.dumps(status, indent=2))


try:
    write_status("started")
    names = None
    for name, model in models.items():
        print("START independent final " + name, flush=True)
        write_status("inference_" + name)
        result = subprocess.run([str(root / "build/dump_clip_embeddings"), str(model), str(images),
            str(output / name), "host"], cwd=root, env=env, capture_output=True, text=True, timeout=360)
        (output / (name + ".log")).write_text(result.stdout + "\n" + result.stderr)
        result.check_returncode()
        manifest = json.loads((output / name / "manifest.json").read_text())
        if manifest["embedding_dim"] != 768 or len(manifest["samples"]) != 120:
            raise RuntimeError("incomplete final dump: " + name)
        current_names = [row["image"] for row in manifest["samples"]]
        if names is None:
            names = current_names
        elif current_names != names:
            raise RuntimeError("final sample order differs: " + name)
        if name != "fp16":
            comparison = subprocess.run(["diff", "-qr", str(output / "fp16/inputs_fp16"),
                str(output / name / "inputs_fp16")], capture_output=True, text=True, timeout=60)
            (output / (name + "_input_diff.log")).write_text(comparison.stdout + "\n" + comparison.stderr)
            comparison.check_returncode()
        if sha(model) != hashes[name]:
            raise RuntimeError("model file changed during final evaluation")
        print("DONE independent final " + name, flush=True)
    status.update(completed=True, count=120, embedding_dim=768, inputs_bitwise_equal=True)
    write_status("completed")
except Exception as error:
    status["error"] = repr(error)
    write_status("failed")
    raise
