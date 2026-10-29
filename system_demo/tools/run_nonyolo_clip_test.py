"""CLIP-only evaluation on frozen visually reviewed, non-YOLO crops."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile

import numpy as np
import paramiko

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/real_eval_200_20261011/nonyolo_reviewed_crop_v2"
BOARD = "/home/orangepi/yolo_clip_bench_systemopt"
RELATIVE = DATA.relative_to(ROOT).as_posix()
REMOTE_DATA = BOARD + "/" + RELATIVE
sys.path.insert(0, str(ROOT / "quantization"))
from evaluate_real_200 import load, classify, task_metrics
from numerical_fidelity import summarize, unit_normalize


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    info = {"completed": False, "phase": "start", "yolo_used": False,
        "pid": __import__("os").getpid(), "model_outputs": [], "no_parameter_or_threshold_tuning": True}
    output = DATA / "test_status.json"
    if output.exists():
        raise RuntimeError("existing test; inspect rather than overwrite/restart")

    def status(phase, **details):
        info.update(phase=phase, **details)
        output.write_text(json.dumps(info, indent=2))
        print("CLIP ONLY " + phase, flush=True)

    def remote(script, timeout=600):
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "orangepi@192.168.123.99", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode:
            raise RuntimeError("board command failed: " + (result.stdout + result.stderr)[-4000:])
        return result.stdout + result.stderr

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    config = paramiko.SSHConfig()
    path = Path.home() / ".ssh/config"
    if path.exists():
        with path.open() as handle:
            config.parse(handle)
    keys = [str(Path(value).expanduser()) for value in config.lookup("192.168.123.99").get("identityfile", [])]
    reference = None
    reference_log = None
    try:
        status("verify_crops")
        selected = json.loads((DATA / "dataset/manifest.json").read_text(encoding="utf-8"))
        crops = json.loads((DATA / "crop_manifest.json").read_text())
        names = [row["image"] for row in selected["samples"]]
        if len(names) != 200 or names != [row["image"] for row in crops["samples"]] or crops["yolo_used"]:
            raise RuntimeError("fixed non-YOLO crop protocol required")
        for row in crops["samples"]:
            if digest(DATA / "crops" / (row["image"] + ".png")) != row["crop_sha256"]:
                raise RuntimeError("crop changed before freezing")
        bundle = DATA / "crops_bundle.zip"
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in ["dataset/manifest.json", "dataset/samples.tsv", "crop_manifest.json"]:
                archive.write(DATA / name, name)
            for name in names:
                archive.write(DATA / "crops" / (name + ".png"), "crops/" + name + ".png")
        info["crop_bundle_sha"] = digest(bundle)
        client.connect("192.168.123.99", username="orangepi", key_filename=keys or None,
            timeout=10, banner_timeout=10, auth_timeout=10)
        client.get_transport().set_keepalive(15)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        remote("test ! -e " + shlex.quote(REMOTE_DATA) + " && mkdir " + shlex.quote(REMOTE_DATA))
        status("upload_crops")
        sftp.put(str(bundle), REMOTE_DATA + "/crops_bundle.zip", confirm=True)
        code = "import pathlib,hashlib,json,zipfile; p=pathlib.Path(" + repr(REMOTE_DATA) + "); b=p/'crops_bundle.zip'; assert hashlib.sha256(b.read_bytes()).hexdigest()==" + repr(info["crop_bundle_sha"]) + "; zipfile.ZipFile(b).extractall(p); m=json.loads((p/'crop_manifest.json').read_text()); assert all(hashlib.sha256((p/'crops'/(r['image']+'.png')).read_bytes()).hexdigest()==r['crop_sha256'] for r in m['samples'])"
        remote("python3 -c " + shlex.quote(code))
        status("freeze_exact_board_inputs")
        (DATA / "freeze.log").write_text(remote("cd " + shlex.quote(BOARD) + " && ./build/freeze_nonyolo_clip_inputs " + shlex.quote(REMOTE_DATA)))
        code = "import pathlib,hashlib,json; p=pathlib.Path(" + repr(REMOTE_DATA) + "); d=p/'frozen/inputs_fp16'; (p/'frozen/input_sha256.json').write_text(json.dumps({f.name[:-4]:hashlib.sha256(f.read_bytes()).hexdigest() for f in d.glob('*.bin')}))"
        remote("python3 -c " + shlex.quote(code))
        (DATA / "frozen/inputs_fp16").mkdir(parents=True)
        for name in ["manifest.json", "input_sha256.json"]:
            sftp.get(REMOTE_DATA + "/frozen/" + name, str(DATA / "frozen" / name))
        hashes = json.loads((DATA / "frozen/input_sha256.json").read_text())
        for name in names:
            target = DATA / "frozen/inputs_fp16" / (name + ".bin")
            sftp.get(REMOTE_DATA + "/frozen/inputs_fp16/" + name + ".bin", str(target))
            if digest(target) != hashes[name]:
                raise RuntimeError("retrieved input changed")
        status("start_fp32_reference_and_board_runs")
        reference_log = (DATA / "fp32_reference.log").open("x", encoding="utf-8")
        reference = subprocess.Popen(["wsl.exe", "-e", "/home/atlantis/yolo-clip-quant-venv/bin/python",
            "-u", "/mnt/f/YOLO-CLIP/system_demo/quantization/run_real_fp32_reference.py", "--data", RELATIVE],
            stdout=reference_log, stderr=subprocess.STDOUT)
        models = {"deployed_historical": BOARD + "/clip_image_encoder_vitl14.rknn",
            "official_fp16": BOARD + "/clip_reference_fp16_232.rknn",
            "official_kl120": BOARD + "/quantization/artifacts/method_sweep_20261011/kl120_reference/clip.rknn"}
        for variant, model in models.items():
            status("inference_" + variant)
            info.setdefault("model_sha", {})[variant] = remote("sha256sum " + shlex.quote(model)).split()[0]
            script = "cd " + shlex.quote(BOARD) + " && flock -n .method_sweep_npu.lock env LD_LIBRARY_PATH=" + shlex.quote(BOARD + "/quant_runtime232")
            script += " ./build/dump_frozen_clip_inputs " + shlex.quote(model) + " " + shlex.quote(REMOTE_DATA + "/dataset") + " " + shlex.quote(REMOTE_DATA + "/frozen") + " " + shlex.quote(REMOTE_DATA + "/" + variant)
            destination = DATA / variant
            destination.mkdir()
            (destination / "board.log").write_text(remote(script), encoding="utf-8")
            for name in ["manifest.json", "embeddings.f32"]:
                sftp.get(REMOTE_DATA + "/" + variant + "/" + name, str(destination / name))
            load(destination, names)
            info["model_outputs"].append(variant)
        source = ROOT / "quantization/artifacts/method_sweep_20261011/affine_rknn/repq_all"
        build = json.loads((source / "status.json").read_text())
        model = source / "clip.rknn"
        if not build["completed"] or digest(model) != build["model_sha"]:
            raise RuntimeError("completed RepQ source required")
        temporary = REMOTE_DATA + "/repq_all_temporary.rknn"
        status("upload_repq_all")
        sftp.put(str(model), temporary, confirm=True)
        if remote("sha256sum " + shlex.quote(temporary)).split()[0] != build["model_sha"]:
            raise RuntimeError("RepQ board SHA mismatch")
        variant = "repq_all"
        status("inference_repq_all")
        script = "cd " + shlex.quote(BOARD) + " && flock -n .method_sweep_npu.lock env LD_LIBRARY_PATH=" + shlex.quote(BOARD + "/quant_runtime232")
        script += " ./build/dump_frozen_clip_inputs " + shlex.quote(temporary) + " " + shlex.quote(REMOTE_DATA + "/dataset") + " " + shlex.quote(REMOTE_DATA + "/frozen") + " " + shlex.quote(REMOTE_DATA + "/" + variant)
        destination = DATA / variant
        destination.mkdir()
        (destination / "board.log").write_text(remote(script), encoding="utf-8")
        for name in ["manifest.json", "embeddings.f32"]:
            sftp.get(REMOTE_DATA + "/" + variant + "/" + name, str(destination / name))
        load(destination, names)
        if digest(model) != build["model_sha"] or remote("sha256sum " + shlex.quote(temporary)).split()[0] != build["model_sha"]:
            raise RuntimeError("source changed; preserve temporary copy")
        sftp.remove(temporary)
        info["model_outputs"].append(variant)
        info["model_sha"][variant] = build["model_sha"]
        status("waiting_fp32_reference")
        if reference.wait(timeout=1200):
            raise RuntimeError("FP32 reference failed; inspect preserved log")
        reference_log.close()
        reference_log = None
        status("evaluate")
        subprocess.run([sys.executable, str(ROOT / "tools/evaluate_nonyolo_clip.py")], check=True, cwd=ROOT)
        status("completed", completed=True, temporary_repq_copy_removed=True, default_models_unchanged=True)
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        if reference_log is not None:
            reference_log.close()
        client.close()


if __name__ == "__main__":
    main()
