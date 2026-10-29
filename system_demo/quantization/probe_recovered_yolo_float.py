"""Read-only floating YOLO comparison using the recovered training/export artifacts."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

root = Path(__file__).resolve().parent
project = Path("G:/OrangePi/Yolo_clip")
weights = project / "outputs/train_single_class/yolo11n_waste_single_class2/weights"
output = root / "real_eval_200_20261011/yolo_float_contract.json"
if output.exists():
    raise RuntimeError("floating YOLO probe already exists; preserve it")
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
detector = YOLO(str(weights / "best.pt"))
detector.model.eval().to("cuda")
options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(str(weights / "best.onnx"), options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
rows = []
for category in ["cycle", "harmful", "kitchen", "others", "empty"]:
    for index in range(2):
        name = f"real_{category}_{index:03d}.jpg"
        image = cv2.imread(str(root / "real_eval_200_20261011/images" / name))
        side = max(image.shape[:2])
        canvas = np.full((side, side, 3), 114, np.uint8)
        canvas[:image.shape[0], :image.shape[1]] = image
        rgb = cv2.cvtColor(cv2.resize(canvas, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255).astype(np.float16).astype(np.float32).transpose(2, 0, 1)[None].copy()
        onnx_result = session.run(None, {input_name: tensor})[0]
        with torch.no_grad():
            prediction = detector.model(torch.from_numpy(tensor).cuda())
            torch_result = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
            torch_result = torch_result.cpu().numpy()
        for label, value in [("onnx_fp32", onnx_result), ("torch_fp32", torch_result)]:
            if value.shape != (1, 5, 8400) or not np.isfinite(value).all():
                raise RuntimeError("recovered YOLO output is not the deployed single-class format")
            best = int(value[0, 4].argmax())
            rows.append({"image": name, "mode": label, "max_confidence": float(value[0, 4, best]),
                "box_xywh": value[0, :4, best].astype(float).tolist()})
report = {"source_pt": str(weights / "best.pt"), "source_onnx": str(weights / "best.onnx"),
    "pt_sha256": hashlib.sha256((weights / "best.pt").read_bytes()).hexdigest(),
    "onnx_sha256": hashlib.sha256((weights / "best.onnx").read_bytes()).hexdigest(),
    "preprocessing": "same top-left letterbox640, RGB/255, IEEE-FP16 rounded values", "rows": rows}
with output.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(rows, indent=2), flush=True)
