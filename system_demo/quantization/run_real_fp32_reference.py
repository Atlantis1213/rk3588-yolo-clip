"""FP32 ONNX reference on exactly the same saved, IEEE-FP16-rounded ROI inputs."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="results/real_eval_200_20261011")
parser.add_argument("--source", default="quantization/artifacts/clip_visual_reference_fp32.onnx")
parser.add_argument("--source-sha", default="c49a301aadd75c022c80c13c30c2ac911fa5b2e10a03583102c2d609178a631d")
parser.add_argument("--output-directory", default="onnx_fp32")
args = parser.parse_args()
data = root / args.data
output = data / args.output_directory
output.mkdir(exist_ok=False)
source = root / args.source
expected = args.source_sha


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


started = time.monotonic()
info = {"phase": "load", "completed": False, "pid": os.getpid(), "source_sha": expected,
    "reference_dtype": "FP32 arithmetic", "input_values": "exact board-preprocessed IEEE-FP16 values cast to FP32",
    "count_completed": 0}


def update(phase, **details):
    info.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    temporary = output / "status.tmp"
    temporary.write_text(json.dumps(info, indent=2))
    os.replace(temporary, output / "status.json")
    print("FP32 reference " + phase + " " + str(info["count_completed"]), flush=True)


try:
    update("verify")
    if sha(source) != expected:
        raise RuntimeError("original floating source SHA changed")
    selected = json.loads((data / "dataset/manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((data / "frozen/manifest.json").read_text())
    hashes = json.loads((data / "frozen/input_sha256.json").read_text())
    names = [row["image"] for row in selected["samples"]]
    if len(names) != 200 or names != [row["image"] for row in frozen["samples"]] or set(names) != set(hashes):
        raise RuntimeError("frozen 200 sample membership/order differs")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].shape != [1, 3, 224, 224]:
        raise RuntimeError("floating source input contract differs")
    input_name = session.get_inputs()[0].name
    update("inference")
    values = []
    rows = []
    for index, name in enumerate(names):
        path = data / "frozen/inputs_fp16" / (name + ".bin")
        if sha(path) != hashes[name]:
            raise RuntimeError("board-preprocessed input SHA differs")
        value = np.fromfile(path, np.float16)
        if value.size != 224 * 224 * 3:
            raise RuntimeError("unexpected input shape")
        tensor = value.reshape(1, 224, 224, 3).astype(np.float32).transpose(0, 3, 1, 2).copy()
        sample_started = time.monotonic()
        result = session.run(None, {input_name: tensor})[0].reshape(-1)
        if result.shape != (768,) or not np.isfinite(result).all():
            raise RuntimeError("invalid floating reference output")
        values.append(result.astype(np.float32))
        rows.append({"image": name, "run_ms": (time.monotonic() - sample_started) * 1000,
            "input_sha256": hashes[name]})
        if (index + 1) % 20 == 0:
            update("inference", count_completed=index + 1)
    np.asarray(values, np.float32).tofile(output / "embeddings.f32")
    (output / "manifest.json").write_text(json.dumps({"embedding_dim": 768, "samples": rows}, indent=2))
    if sha(source) != expected:
        raise RuntimeError("floating source changed during evaluation")
    update("completed", completed=True, count_completed=200)
except Exception as error:
    update("failed", error=repr(error))
    raise
