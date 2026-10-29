"""One persistent SSH/SFTP session; resume owned partial file and verify SHA before promotion."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import paramiko

root = Path(__file__).resolve().parents[1]
output = root / "quantization/artifacts/recovered_project_20261011"
status_path = output / "transfer_status.json"
expected = "a51e3e827dcdcd733fdc20a94726a97828a1dd18521b6034ce9d07b58627fa6b"
source_path = "G:/OrangePi/Yolo_clip/models/clip_image_encoder_vitl14.onnx"
destination = output / "clip_image_encoder_vitl14.onnx"
partial = output / "clip_image_encoder_vitl14.onnx.partial"
info = {"completed": False, "source": source_path, "expected_sha": expected, "bytes_received": 0}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def update(phase, **details):
    info.update(phase=phase, **details)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, status_path)
    print("RECOVERED ASSET " + phase + " " + str(info["bytes_received"]), flush=True)


client = paramiko.SSHClient()
try:
    if destination.exists():
        raise RuntimeError("completed-name asset exists; verify it rather than overwriting")
    ssh_root = Path.home() / ".ssh"
    known_hosts = ssh_root / "known_hosts"
    if not known_hosts.exists():
        raise RuntimeError("existing SSH host-key trust required")
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    config = paramiko.SSHConfig()
    if (ssh_root / "config").exists():
        with (ssh_root / "config").open() as handle:
            config.parse(handle)
    configured = config.lookup("127.0.0.1")
    identities = [str(Path(path).expanduser()) for path in configured.get("identityfile", [])]
    update("connecting")
    client.connect("127.0.0.1", port=2222, username="atlantis", key_filename=identities or None,
        timeout=10, banner_timeout=15, auth_timeout=15, allow_agent=True, look_for_keys=True)
    transport = client.get_transport()
    transport.set_keepalive(10)
    sftp = paramiko.SFTPClient.from_transport(transport, window_size=16 * 1024 * 1024, max_packet_size=32768)
    sftp.get_channel().settimeout(30)
    remote_size = sftp.stat(source_path).st_size
    if remote_size != 1216143767:
        raise RuntimeError("original ONNX source size changed")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > remote_size:
        raise RuntimeError("partial file is larger than source")
    with sftp.open(source_path, "rb") as remote:
        if offset:
            with partial.open("rb") as local:
                for position in [0, max(0, offset - 32768)]:
                    local.seek(position); remote.seek(position)
                    if local.read(min(32768, offset - position)) != remote.read(min(32768, offset - position)):
                        raise RuntimeError("partial prefix/tail does not match original source; retain it for inspection")
        update("resuming", bytes_received=offset, source_bytes=remote_size)
        last_report = time.monotonic()
        with partial.open("ab") as local:
            while offset < remote_size:
                batch_end = min(remote_size, offset + 2 * 1024 * 1024)
                chunks = [(position, min(32768, batch_end - position)) for position in range(offset, batch_end, 32768)]
                kwargs = {"max_concurrent_prefetch_requests": 16} if "max_concurrent_prefetch_requests" in inspect.signature(remote.readv).parameters else {}
                for (position, length), payload in zip(chunks, remote.readv(chunks, **kwargs)):
                    if position != offset or len(payload) != length:
                        raise RuntimeError("noncontiguous or short SFTP chunk")
                    local.write(payload)
                    offset += length
                local.flush()
                if time.monotonic() - last_report > 10:
                    update("resuming", bytes_received=offset)
                    last_report = time.monotonic()
    update("verify_sha", bytes_received=offset)
    if digest(partial) != expected:
        raise RuntimeError("full source SHA mismatch; do not promote partial artifact")
    if destination.exists():
        raise RuntimeError("destination appeared during download; preserve both files")
    partial.rename(destination)
    extras = {
        "best.onnx": "G:/OrangePi/Yolo_clip/outputs/train_single_class/yolo11n_waste_single_class2/weights/best.onnx",
        "best.pt": "G:/OrangePi/Yolo_clip/outputs/train_single_class/yolo11n_waste_single_class2/weights/best.pt",
        "args.yaml": "G:/OrangePi/Yolo_clip/outputs/train_single_class/yolo11n_waste_single_class2/args.yaml",
        "results.csv": "G:/OrangePi/Yolo_clip/outputs/train_single_class/yolo11n_waste_single_class2/results.csv",
        "app.yaml": "G:/OrangePi/Yolo_clip/configs/app.yaml",
        "rknn_utils.py": "G:/OrangePi/Yolo_clip/src/rknn_utils.py",
        "original_inference.py": "G:/OrangePi/Yolo_clip/src/pipeline/inference.py",
        "hf_clip.py": "G:/OrangePi/Yolo_clip/src/clip_classifier/hf_clip.py",
        "original_clip_export.py": "G:/OrangePi/Yolo_clip/src/clip_classifier/export.py",
        "yolo_float_contract.json": "D:/YOLO-CLIP-QAT-20261010/real_eval_200_20261011/yolo_float_contract.json"}
    retrieved = {}
    for name, remote_path in extras.items():
        target = output / name
        if target.exists():
            raise RuntimeError("extra asset exists; do not overwrite: " + name)
        sftp.get(remote_path, str(target))
        retrieved[name] = {"sha256": digest(target), "bytes": target.stat().st_size, "source": remote_path}
    (output / "retrieved_files.json").write_text(json.dumps(retrieved, indent=2))
    update("completed", completed=True)
except Exception as error:
    update("failed", error=repr(error), bytes_received=partial.stat().st_size if partial.exists() else info["bytes_received"])
    raise
finally:
    client.close()
