"""Fixed-sample PyTorch/ONNX/board differential loop; never changes deployment."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["torch", "onnx", "compare"], required=True)
parser.add_argument("--output", default="artifacts/diagnosis_20261010")
args = parser.parse_args()
root = Path(__file__).resolve().parent
destination = root / args.output
destination.mkdir(parents=True, exist_ok=True)
fp16_dir = root / "artifacts/quant_public_fp16_ieee"
manifest = json.loads((fp16_dir / "manifest.json").read_text())
indices = [0, 20, 40, 60, 80, 100]  # one fixed validation sample/class, no final-test selection
names = [manifest["samples"][i]["image"] for i in indices]
(destination / "inputs").mkdir(exist_ok=True)


def input_tensor(name):
    values = np.fromfile(fp16_dir / "inputs_fp16" / (name + ".bin"), dtype=np.float16)
    tensor = values.reshape(1, 224, 224, 3).astype(np.float32).transpose(0, 3, 1, 2).copy()
    np.save(destination / "inputs" / (name + ".npy"), tensor)
    return tensor


def metric(reference, candidate):
    a, b = np.asarray(reference, np.float64).ravel(), np.asarray(candidate, np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("nonfinite differential output")
    return {"cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "rmse": float(np.sqrt(np.mean((a-b)**2))), "max_abs_error": float(np.abs(a-b).max())}


if args.backend == "torch":
    import torch
    from clip.model import VisionTransformer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint = torch.jit.load(str(root / "assets/ViT-L-14.pt"), map_location="cpu").eval()
    state = {k[7:]: v for k, v in checkpoint.state_dict().items() if k.startswith("visual.")}
    model = VisionTransformer(224, 14, 1024, 24, 16, 768).float().eval()
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()
    outputs = []
    with torch.no_grad():
        for name in names:
            outputs.append(model(torch.from_numpy(input_tensor(name))).numpy().reshape(-1))
            print("REFERENCE torch " + name, flush=True)
    np.save(destination / "torch.f32.npy", np.asarray(outputs))
elif args.backend == "onnx":
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(root / "artifacts/clip_visual_reference_fp32.onnx"),
                                   sess_options=options, providers=["CPUExecutionProvider"])
    outputs = []
    for name in names:
        outputs.append(session.run(None, {"pixel_values": input_tensor(name)})[0].reshape(-1))
        print("REFERENCE onnx " + name, flush=True)
    np.save(destination / "onnx.f32.npy", np.asarray(outputs))
else:
    for name in names:
        input_tensor(name)
    reference = np.load(destination / "torch.f32.npy")
    onnx = np.load(destination / "onnx.f32.npy")
    board = {}
    for label, folder in [("fp16", "quant_public_fp16_ieee"), ("int8", "quant_public_int8_normal_ieee")]:
        location = root / "artifacts" / folder
        metadata = json.loads((location / "manifest.json").read_text())
        if [s["image"] for s in metadata["samples"]] != [s["image"] for s in manifest["samples"]]:
            raise RuntimeError("board sample order differs")
        board[label] = np.fromfile(location / "embeddings.f32", np.float32).reshape(-1, 768)[indices]
    rows = [{"image": name, "torch_to_onnx": metric(reference[i], onnx[i]),
             "torch_to_board_fp16": metric(reference[i], board["fp16"][i]),
             "torch_to_board_int8": metric(reference[i], board["int8"][i]),
             "board_fp16_to_int8": metric(board["fp16"][i], board["int8"][i])}
            for i, name in enumerate(names)]
    report = {"sample_policy": "first validation sample/class; six fixed samples", "rows": rows,
              "onnx_pass_cosine_threshold": 0.9999, "timestamp_unix": time.time()}
    (destination / "reference_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if any(r["torch_to_onnx"]["cosine"] < 0.9999 for r in rows):
        raise SystemExit("FAIL: FP32 exporter equivalence requires diagnosis before quantization")
