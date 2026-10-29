"""One board model slot, exact-input fidelity, preserved evidence, no promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shlex
import subprocess
import sys
import time

import paramiko
from evaluate_real_candidate import evaluate

ROOT = Path(__file__).resolve().parents[1]
BOARD = "/home/orangepi/yolo_clip_bench_systemopt"
BASE = ROOT / "results/real_eval_200_20261011/proposal_diagnostic"
ARTIFACTS = ROOT / "quantization/artifacts/method_sweep_20261011"
AFFINE = ["repq_ln2", "smooth_ln2_a25", "smooth_ln2_a50", "smooth_ln2_a75", "repq_all", "smooth_all_a50"]
RECONSTRUCTION = ["bias_down", "adaround_down", "adaround_bias_down"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=AFFINE + RECONSTRUCTION, default=AFFINE)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reconstruction-attempt", default="irfix_20261011")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[a-zA-Z0-9_-]+", value) for value in [args.run, args.reconstruction_attempt]):
        raise ValueError("safe run and attempt directory names required")
    if len(set(args.profiles)) != len(args.profiles):
        raise ValueError("duplicate profiles")
    output = BASE / args.run
    output.mkdir(exist_ok=False)
    info = {"completed": False, "profiles_predeclared": args.profiles, "jobs": [],
        "automatic_promotion": False, "calibration_training_on_this_set": False, "pid": __import__("os").getpid()}

    def status(phase, **values):
        info.update(phase=phase, **values)
        (output / "status.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        print("REAL CANDIDATE " + phase, flush=True)

    def command(script, timeout=600):
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            "orangepi@192.168.123.99", script], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"board command exit {result.returncode}: " + (result.stdout + result.stderr)[-5000:])
        return result.stdout + result.stderr

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    config = paramiko.SSHConfig()
    ssh_config = Path.home() / ".ssh/config"
    if ssh_config.exists():
        with ssh_config.open() as handle:
            config.parse(handle)
    keys = [str(Path(path).expanduser()) for path in config.lookup("192.168.123.99").get("identityfile", [])]
    remote_slot = BOARD + "/quantization/artifacts/real_candidate_" + args.run
    remote_output = BOARD + "/results/real_eval_200_20261011/proposal_diagnostic/" + args.run
    try:
        status("verify_frozen_inputs")
        expected = json.loads((BASE / "frozen/input_sha256.json").read_text())
        probe = "import hashlib,json,pathlib; p=pathlib.Path(" + repr(BOARD + "/results/real_eval_200_20261011/proposal_diagnostic/frozen/inputs_fp16") + "); print(json.dumps({f.name[:-4]:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.glob('*.bin')}))"
        if json.loads(command("python3 -c " + shlex.quote(probe))) != expected:
            raise RuntimeError("board frozen input SHA mismatch")
        command("test ! -e " + shlex.quote(remote_slot) + " && test ! -e " + shlex.quote(remote_output)
            + " && mkdir " + shlex.quote(remote_slot) + " " + shlex.quote(remote_output))
        client.connect("192.168.123.99", username="orangepi", key_filename=keys or None,
            timeout=10, banner_timeout=10, auth_timeout=10)
        client.get_transport().set_keepalive(15)
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(60)
        for profile in args.profiles:
            source = ARTIFACTS / ("affine_rknn/" + profile if profile in AFFINE
                else "reconstruction_rknn/" + args.reconstruction_attempt + "/" + profile)
            build = json.loads((source / "status.json").read_text())
            model = source / "clip.rknn"
            if not build.get("completed") or digest(model) != build["model_sha"]:
                raise RuntimeError("completed SHA-verified model required: " + profile)
            candidate = output / profile
            candidate.mkdir()
            (candidate / "source_build_status.json").write_bytes((source / "status.json").read_bytes())
            destination = remote_slot + "/clip.rknn"
            remote_candidate = remote_output + "/" + profile
            status("upload_" + profile)
            sftp.put(str(model), destination, confirm=True)
            remote_sha = command("sha256sum " + shlex.quote(destination)).split()[0]
            if remote_sha != build["model_sha"]:
                raise RuntimeError("board model SHA mismatch")
            status("inference_" + profile)
            script = "cd " + shlex.quote(BOARD) + " && flock -n .method_sweep_npu.lock env LD_LIBRARY_PATH=" + shlex.quote(BOARD + "/quant_runtime232")
            script += " ./build/dump_frozen_clip_inputs " + shlex.quote(destination)
            script += " ./results/real_eval_200_20261011/dataset ./results/real_eval_200_20261011/proposal_diagnostic/frozen " + shlex.quote(remote_candidate)
            (candidate / "board.log").write_text(command(script), encoding="utf-8")
            for name in ["manifest.json", "embeddings.f32"]:
                sftp.get(remote_candidate + "/" + name, str(candidate / name))
            report = evaluate(BASE, candidate)
            if digest(model) != build["model_sha"]:
                raise RuntimeError("local source changed during inference")
            # Remove exactly our disposable uploaded file, after local source and
            # complete output validation. Do not touch any historical model/IR.
            if posixpath.dirname(destination) != remote_slot or sftp.lstat(destination).st_mode & 0o170000 != 0o100000:
                raise RuntimeError("unexpected temporary model target")
            if command("sha256sum " + shlex.quote(destination)).split()[0] != build["model_sha"]:
                raise RuntimeError("temporary model changed; do not remove")
            sftp.remove(destination)
            job = {"profile": profile, "model_sha": build["model_sha"], "correct": report["task"]["correct"],
                "accuracy": report["task"]["accuracy"],
                "cosine_fp32": report["fidelity"]["onnx_fp32"]["raw_output_contract"]["cosine_mean"],
                "single_context_mean_run_ms": report["single_context_run_ms"]["mean"],
                "temporary_board_copy_removed": True, "full_local_model_preserved": True}
            info["jobs"].append(job)
            status("done_" + profile)
            print(json.dumps(job), flush=True)
        status("completed", completed=True)
    except Exception as error:
        status("failed", error=repr(error))
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
