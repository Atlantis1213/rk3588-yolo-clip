# RKNN 转换入口：从固定参考 ONNX 和校准协议生成 FP16、INT8 或混合精度实验产物。
# 构建成功仅代表可生成模型，是否可部署仍要检查板端输出、准确率及实际运行精度。

"""RKNN conversion variants from one reconstructed FP32 ONNX and one calibration split."""
import argparse
import json
import os
from pathlib import Path
import time
from rknn.api import RKNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=[
        "fp16",
        "fp16-batch3",
        "int8-normal",
        "int8-mmse",
        "int8-auto-hybrid",
        "int8-equalized",
        "int8-nhwc-control",
    ],
    required=True,
)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
# equalized 模式切换为重参数化 ONNX，其余使用共同浮点源；calibration 依据对照模式选取布局或子集。
source = artifacts / (
    "clip_visual_equalized_fp32.onnx"
    if args.mode == "int8-equalized"
    else "clip_visual_reference_fp32.onnx"
)
calibration = artifacts / (
    "calibration_nhwc.txt"
    if args.mode == "int8-nhwc-control"
    else "calibration_mmse.txt" if args.mode == "int8-mmse" else "calibration.txt"
)
if not source.exists():
    raise FileNotFoundError("export the reference ONNX first")
# FP16 模式不需要校准；其他模式必须先准备与验证/测试不重叠的校准输入。
quantized = not args.mode.startswith("fp16")
if quantized and not calibration.exists():
    raise FileNotFoundError("prepare disjoint calibration data first")
work = artifacts / ("build_" + args.mode)
# 各模式在独立构建目录运行，避免 SDK 临时 IR 和日志相互覆盖。
work.mkdir(exist_ok=True)
os.chdir(work)
model = RKNN(verbose=args.verbose)
start = time.monotonic()


def check(code, phase):
    if code != 0:
        raise RuntimeError(f"{phase} failed: {code}")
    print("DONE " + phase, flush=True)


try:
    # 校准输入已完成 CLIP 标准化，因此 SDK 的 mean=0/std=1，避免重复标准化。
    check(
        model.config(
            target_platform="rk3588",
            quantized_dtype="w8a8",
            quantized_method="channel",
            quantized_algorithm="mmse" if args.mode == "int8-mmse" else "normal",
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            optimization_level=3,
            auto_hybrid_cos_thresh=0.995,
        ),
        "configure",
    )
    check(model.load_onnx(model=str(source)), "load ONNX")
    print("START RKNN build " + args.mode, flush=True)
    # do_quantization 决定是否校准；auto_hybrid 由 SDK 自动挑选保留浮点的层。
    # batch3 是批量输入实验，与多个独立 context 的并发实验不同。
    check(
        model.build(
            do_quantization=quantized,
            dataset=str(calibration) if quantized else None,
            auto_hybrid=args.mode == "int8-auto-hybrid",
            rknn_batch_size=3 if args.mode == "fp16-batch3" else None,
        ),
        "build",
    )
    output = artifacts / ("clip_" + args.mode + ".rknn")
    check(model.export_rknn(str(output)), "export")
    # 保存源模型、校准清单和转换耗时，供后续结果追溯；不代表历史部署模型的来源。
    (work / "conversion.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "source": str(source),
                "calibration": str(calibration) if quantized else None,
                "output": str(output),
                "seconds": time.monotonic() - start,
                "bytes": output.stat().st_size,
                "source_note": "same official reconstructed ONNX across variants; not historical deployed provenance",
            },
            indent=2,
        )
    )
# 成功或失败都释放 Toolkit 资源，避免一次转换失败影响后续进程资源。
finally:
    model.release()
