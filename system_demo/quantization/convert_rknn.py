"""RKNN conversion variants from one reconstructed FP32 ONNX and one calibration split."""
import argparse
import json
import os
from pathlib import Path
import time
from rknn.api import RKNN

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["fp16", "fp16-batch3", "int8-normal", "int8-mmse", "int8-auto-hybrid", "int8-equalized", "int8-nhwc-control"], required=True)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
source = artifacts / ("clip_visual_equalized_fp32.onnx" if args.mode == "int8-equalized" else "clip_visual_reference_fp32.onnx")
calibration = artifacts / ("calibration_nhwc.txt" if args.mode == "int8-nhwc-control" else
                          "calibration_mmse.txt" if args.mode == "int8-mmse" else "calibration.txt")
if not source.exists(): raise FileNotFoundError("export the reference ONNX first")
quantized = not args.mode.startswith("fp16")
if quantized and not calibration.exists(): raise FileNotFoundError("prepare disjoint calibration data first")
work = artifacts / ("build_" + args.mode)
work.mkdir(exist_ok=True)
os.chdir(work)
model = RKNN(verbose=args.verbose)
start = time.monotonic()


def check(code, phase):
    if code != 0: raise RuntimeError(f"{phase} failed: {code}")
    print("DONE " + phase, flush=True)


try:
    check(model.config(target_platform="rk3588", quantized_dtype="w8a8", quantized_method="channel",
                       quantized_algorithm="mmse" if args.mode == "int8-mmse" else "normal",
                       mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], optimization_level=3,
                       auto_hybrid_cos_thresh=0.995), "configure")
    check(model.load_onnx(model=str(source)), "load ONNX")
    print("START RKNN build " + args.mode, flush=True)
    check(model.build(do_quantization=quantized, dataset=str(calibration) if quantized else None,
                      auto_hybrid=args.mode == "int8-auto-hybrid",
                      rknn_batch_size=3 if args.mode == "fp16-batch3" else None), "build")
    output = artifacts / ("clip_" + args.mode + ".rknn")
    check(model.export_rknn(str(output)), "export")
    (work / "conversion.json").write_text(json.dumps({"mode": args.mode, "source": str(source),
        "calibration": str(calibration) if quantized else None, "output": str(output),
        "seconds": time.monotonic() - start, "bytes": output.stat().st_size,
        "source_note": "same official reconstructed ONNX across variants; not historical deployed provenance"}, indent=2))
finally:
    model.release()
