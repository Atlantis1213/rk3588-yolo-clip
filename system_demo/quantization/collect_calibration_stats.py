# 校准统计的生产端：对固定 120 个输入运行浮点教师，收集 LN 和 GELU 激活分布。
# 输出逐通道范围/均值/方差、GELU 直方图、normal/KL 的近似 QDQ 误差和教师 embedding。

"""Full calibration distributions for LN and post-GELU; never inspect validation/test data."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--threads", type=int, default=2)
args = parser.parse_args()
root = Path(__file__).resolve().parent
output = root / "artifacts/method_sweep_20261011/calibration_stats120"
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(root / "assets"))
import numpy as np
import torch
import yaml
from clip_model_official import VisionTransformer
from record_ir_provenance import fingerprint

started = time.monotonic()
metadata = {
    "phase": "verify",
    "completed": False,
    "pid": os.getpid(),
    "device": args.device,
    "calibration_count": 120,
    "count_seen": 0,
    "scope": "original calibration inputs only; no validation/test activations",
}


def status(phase, **details):
    metadata.update(phase=phase, elapsed_seconds=time.monotonic() - started, **details)
    (output / "status.json").write_text(json.dumps(metadata, indent=2))
    print(
        json.dumps(
            {
                "phase": phase,
                "count_seen": metadata["count_seen"],
                "elapsed_seconds": metadata["elapsed_seconds"],
            }
        ),
        flush=True,
    )


try:
    status("verify")
    previous = json.loads(
        (root / "artifacts/manual_hybrid_20261010/ir_provenance.json").read_text()
    )
    if fingerprint(root) != previous:
        raise RuntimeError("source, original calibration or SDK changed")
    paths = [
        Path(line) for line in (root / "artifacts/calibration.txt").read_text().splitlines() if line
    ]
    if len(paths) != 120 or len(set(paths)) != 120:
        raise RuntimeError("expected 120 distinct calibration inputs")
    normal = yaml.safe_load(
        (
            root / "artifacts/manual_hybrid_20261010/clip_visual_reference_fp32.quantization.cfg"
        ).read_text()
    )["quantize_parameters"]
    kl = yaml.safe_load(
        (
            root
            / "artifacts/calibration_algorithms_20261010/kl_divergence/clip_visual_reference_fp32.quantization.cfg"
        ).read_text()
    )["quantize_parameters"]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute a different backend")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    status("load_original_weights")
    checkpoint = torch.jit.load(str(root / "assets/ViT-L-14.pt"), map_location="cpu").eval()
    state = {
        name[7:]: value
        for name, value in checkpoint.state_dict().items()
        if name.startswith("visual.")
    }
    model = VisionTransformer(224, 14, 1024, 24, 16, 768).float().eval()
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()
    model.to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # LN 输出宽度对应后继 Linear 的输入通道；GELU 的 4096 维输出对应 MLP 降维层输入。
    modules = {}
    for index, block in enumerate(model.transformer.resblocks):
        modules[f"block{index}.ln1"] = (block.ln_1, block.attn.in_proj_weight, None)
        modules[f"block{index}.ln2"] = (block.ln_2, block.mlp.c_fc.weight, None)
        modules[f"block{index}.gelu"] = (
            block.mlp.gelu,
            block.mlp.c_proj.weight,
            f"/transformer/resblocks/resblocks.{index}/mlp/gelu/Mul_1_output_0_tp_rs",
        )
    modules["ln_post"] = (model.ln_post, model.proj.T, None)
    statistics, arrays = {}, {}
    for name, (_, weight, tensor_name) in modules.items():
        width = weight.shape[1]
        statistics[name] = {
            "rows": 0,
            "min": torch.full((width,), float("inf"), device=args.device),
            "max": torch.full((width,), -float("inf"), device=args.device),
            "sum": torch.zeros(width, dtype=torch.float64, device=args.device),
            "sum_sq": torch.zeros(width, dtype=torch.float64, device=args.device),
            "cls_sum": torch.zeros(width, dtype=torch.float64, device=args.device),
            "cls_rows": 0,
        }
        # 权重为 [out,in]，沿 out 取最大值，得到每个输入通道的权重幅度，供 SmoothQuant 风格缩放。
        arrays[name + ".weight_input_absmax"] = weight.detach().abs().amax(dim=0).cpu().numpy()
        if tensor_name:
            lower = min(-0.25, float(normal[tensor_name]["min"][0]) - 0.01)
            upper = max(1.0, float(normal[tensor_name]["max"][0]) * 1.02 + 0.1)
            statistics[name].update(
                hist_min=lower,
                hist_max=upper,
                histogram=torch.zeros(65536, dtype=torch.float64, device=args.device),
                out_of_histogram=0,
            )
            for algorithm in ["normal", "kl"]:
                statistics[name][algorithm] = {
                    "zeros": 0,
                    "clipped": 0,
                    "squared_error": 0.0,
                    "mean_error_sum": torch.zeros(width, dtype=torch.float64, device=args.device),
                    "cls_error_sum": torch.zeros(width, dtype=torch.float64, device=args.device),
                }

    # 工厂函数把当前模块名绑定进 hook，避免循环闭包把所有模块统计写入最后一层。
    def collector(name, tensor_name):
        def hook(module, inputs, value):
            # 把 token 和 batch 维展平为行，最后一维保留通道；当前逐图推理的第 0 行是 CLS token。
            # 脱离计算图后统计，不存全量中间激活，降低内存消耗。
            value = value.detach().float().reshape(-1, value.shape[-1])
            if not torch.isfinite(value).all():
                raise RuntimeError("nonfinite calibration activation: " + name)
            record = statistics[name]
            record["rows"] += value.shape[0]
            record["min"] = torch.minimum(record["min"], value.amin(dim=0))
            record["max"] = torch.maximum(record["max"], value.amax(dim=0))
            record["sum"] += value.double().sum(dim=0)
            record["sum_sq"] += value.double().square().sum(dim=0)
            record["cls_sum"] += value[0].double()
            record["cls_rows"] += 1
            if tensor_name:
                record["histogram"] += torch.histc(
                    value, bins=65536, min=record["hist_min"], max=record["hist_max"]
                ).double()
                record["out_of_histogram"] += int(
                    ((value < record["hist_min"]) | (value > record["hist_max"])).sum()
                )
                for algorithm, parameters in [("normal", normal), ("kl", kl)]:
                    qp = parameters[tensor_name]
                    scale, zero = float(qp["scale"][0]), int(qp["zero_point"][0])
                    # 近似仿射量化 q=round(x/scale)+zero，再饱和到 [-128,127]。
                    # 记录量化为零点、超界饱和与反量化误差三类指标，不能把它们直接解释为分类错误率。
                    unbounded = torch.round(value / scale) + zero
                    quantized = unbounded.clamp(-128, 127)
                    error = ((quantized - zero) * scale - value).double()
                    qrecord = record[algorithm]
                    qrecord["zeros"] += int((quantized == zero).sum())
                    qrecord["clipped"] += int(((unbounded < -128) | (unbounded > 127)).sum())
                    qrecord["squared_error"] += float(error.square().sum())
                    qrecord["mean_error_sum"] += error.sum(dim=0)
                    qrecord["cls_error_sum"] += error[0]

        return hook

    handles = [
        module.register_forward_hook(collector(name, tensor_name))
        for name, (module, _, tensor_name) in modules.items()
    ]
    embeddings = []
    status("collect_calibration_forward")
    with torch.no_grad():
        for index, path in enumerate(paths):
            value = np.load(path, allow_pickle=False)
            if value.shape != (1, 3, 224, 224) or value.dtype != np.float32:
                raise RuntimeError("calibration tensor contract differs")
            result = model(torch.from_numpy(value).to(args.device)).cpu().numpy().reshape(-1)
            if result.size != 768 or not np.isfinite(result).all():
                raise RuntimeError("invalid calibration teacher output")
            embeddings.append(result)
            if (index + 1) % 10 == 0:
                status("collect_calibration_forward", count_seen=index + 1)
    # 统计完成后移除 hook，避免模型后续前向重复累加或保持对大数组的额外引用。
    for handle in handles:
        handle.remove()
    summaries = []
    for name, record in statistics.items():
        for key in ["min", "max"]:
            arrays[name + "." + key] = record[key].cpu().numpy()
        arrays[name + ".mean"] = (record["sum"] / record["rows"]).cpu().numpy()
        # 用 E[x²]-E[x]² 计算总体方差；浮点抵消可能出现极小负值，截断到零。
        arrays[name + ".variance"] = (
            (record["sum_sq"] / record["rows"] - (record["sum"] / record["rows"]) ** 2)
            .clamp_min(0)
            .cpu()
            .numpy()
        )
        arrays[name + ".cls_mean"] = (record["cls_sum"] / record["cls_rows"]).cpu().numpy()
        summary = {
            "module": name,
            "rows": record["rows"],
            "width": len(record["min"]),
            "min": float(record["min"].min()),
            "max": float(record["max"].max()),
        }
        if "histogram" in record:
            arrays[name + ".histogram"] = record["histogram"].cpu().numpy()
            arrays[name + ".histogram_bounds"] = np.array([record["hist_min"], record["hist_max"]])
            summary["out_of_histogram"] = record["out_of_histogram"]
            elements = record["rows"] * summary["width"]
            for algorithm in ["normal", "kl"]:
                qrecord = record[algorithm]
                summary[algorithm] = {
                    "zero_fraction": qrecord["zeros"] / elements,
                    "clipping_fraction": qrecord["clipped"] / elements,
                    "qdq_rmse": (qrecord["squared_error"] / elements) ** 0.5,
                }
                arrays[name + "." + algorithm + ".mean_error"] = (
                    (qrecord["mean_error_sum"] / record["rows"]).cpu().numpy()
                )
                arrays[name + "." + algorithm + ".cls_mean_error"] = (
                    (qrecord["cls_error_sum"] / record["cls_rows"]).cpu().numpy()
                )
        summaries.append(summary)
    np.savez_compressed(output / "channel_statistics.npz", **arrays)
    np.save(output / "teacher_embeddings.npy", np.asarray(embeddings, np.float32))
    (output / "statistics.json").write_text(
        json.dumps(
            {
                "scope": metadata["scope"],
                "calibration_count": 120,
                "input_names": [path.name for path in paths],
                "modules": summaries,
                "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "qdq_note": "float reference statistics and approximate affine QDQ, not bit-exact NPU internals",
            },
            indent=2,
        )
    )
    if fingerprint(root) != previous:
        raise RuntimeError("source/calibration changed during collection")
    status("completed", completed=True, count_seen=120, module_count=len(modules))
except Exception as error:
    status("failed", error=repr(error))
    raise
