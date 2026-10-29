"""Original recovered ONNX, exact board inputs, independent of classifier/prompt accuracy."""
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parent
base = root / "real_eval_200_20261011"
output = base / "recovered_project_fp32"
output.mkdir(exist_ok=False)
source = Path("G:/OrangePi/Yolo_clip/models/clip_image_encoder_vitl14.onnx")
expected = "a51e3e827dcdcd733fdc20a94726a97828a1dd18521b6034ce9d07b58627fa6b"
started = time.monotonic()
info = {"completed": False, "pid": os.getpid(), "source_sha": expected, "count_completed": 0,
    "input_policy": "bit-identical original board FP16 tensors cast to FP32", "backend": "ONNX Runtime CPU FP32"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def update(phase, **details):
    info.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, output / "status.json")
    print("ORIGINAL FP32 " + phase + " " + str(info["count_completed"]), flush=True)


try:
    update("verify_source")
    if sha(source) != expected:
        raise RuntimeError("original recovered source SHA differs")
    with zipfile.ZipFile(base / "exact_frozen_inputs.zip") as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        hashes = json.loads(bundle.read("input_sha256.json"))
        names = [row["image"] for row in manifest["samples"]]
        expected_names = {"manifest.json", "input_sha256.json"} | {"inputs_fp16/" + name + ".bin" for name in names}
        if len(names) != 200 or len(set(names)) != 200 or set(bundle.namelist()) != expected_names:
            raise RuntimeError("frozen bundle membership differs")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or session.get_inputs()[0].shape != [1, 3, 224, 224]:
            raise RuntimeError("original source input shape differs")
        input_name = session.get_inputs()[0].name
        outputs, rows = [], []
        for index, name in enumerate(names):
            payload = bundle.read("inputs_fp16/" + name + ".bin")
            if hashlib.sha256(payload).hexdigest() != hashes[name]:
                raise RuntimeError("board input bytes differ")
            tensor = np.frombuffer(payload, np.float16).reshape(1, 224, 224, 3).astype(np.float32).transpose(0, 3, 1, 2).copy()
            tick = time.monotonic()
            result = session.run(None, {input_name: tensor})[0].reshape(-1)
            if result.shape != (768,) or not np.isfinite(result).all():
                raise RuntimeError("invalid original-source embedding")
            outputs.append(result.astype(np.float32))
            rows.append({"image": name, "run_ms": (time.monotonic() - tick) * 1000, "input_sha256": hashes[name]})
            if (index + 1) % 20 == 0:
                update("inference", count_completed=index + 1)
    np.asarray(outputs, np.float32).tofile(output / "embeddings.f32")
    (output / "manifest.json").write_text(json.dumps({"embedding_dim": 768, "samples": rows}, indent=2))
    if sha(source) != expected:
        raise RuntimeError("source changed during reference run")
    update("completed", completed=True, count_completed=200)
except Exception as error:
    update("failed", error=repr(error))
    raise
