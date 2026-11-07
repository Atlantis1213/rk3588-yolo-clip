# 仅用校准数据逐层重建 MLP down-projection：学习舍入，再估计均值偏置修正。
# 固定 SDK 的 KL120 量化参数；这是受限 AdaRound 风格实验，非完整 BRECQ 或硬件精确 QAT。

"""Bounded CUDA AdaRound/bias pilot on calibration only; no held-out task tuning."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from clip_model_official import VisionTransformer
from adaptive_rounding_core import AdaptiveRounding, fixed_quantize


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--run", default="adaround_down_kl120_20261011")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_tokens <= 0:
        raise ValueError("positive steps and batch required")
    root = Path(__file__).resolve().parent
    output = root / "runs" / args.run
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    info = {"completed": False, "pid": os.getpid(), "steps_per_block": args.steps,
        "scope": "fixed SDK KL120 scales, AdaRound-style MLP down-projection pilot; not full BRECQ or bit-exact NPU QAT",
        "development_or_locked_test_evaluated": False, "blocks": []}
    def update(phase, **values):
        info.update(phase=phase, elapsed_seconds=time.monotonic() - started, **values)
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(info, indent=2), encoding="utf-8")
        os.replace(temporary, output / "status.json")
        print(json.dumps({"phase": phase, "elapsed_seconds": info["elapsed_seconds"],
            "blocks_completed": len(info["blocks"])}), flush=True)
    try:
        update("verify")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; refuse silent CPU substitution")
        torch.set_num_threads(2)
        torch.manual_seed(20261011)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sha(root / "assets/ViT-L-14.pt") != "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836":
            raise RuntimeError("official weight hash differs")
        if sha(root / "assets/dataset-resized.zip") != "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2":
            raise RuntimeError("data archive hash differs")
        sources = json.loads((root / "dataset_provenance.json").read_text())["calibration_sources"]
        new_sources = [row["source"] for name in ["development", "locked_test"]
            for row in json.loads((root / ("v2_" + name + "_labels.json")).read_text())["samples"]]
        if len(sources) != 120 or len(set(sources)) != 120 or set(sources) & set(new_sources):
            raise RuntimeError("original calibration/held-out leakage gate failed")
        contract = json.loads((root / "reconstruction_contract_kl120.json").read_text())
        (output / "provenance.json").write_text(json.dumps({"args": vars(args), "sources": sources,
            "contract_sha": sha(root / "reconstruction_contract_kl120.json"), "trainer_sha": sha(Path(__file__)),
            "rounding_core_sha": sha(root / "adaptive_rounding_core.py"), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(), "tokens_per_image": 16,
            "selection": "all 24 down-projections; no selection on development labels"}, indent=2))
        update("load_teacher")
        checkpoint = torch.jit.load(str(root / "assets/ViT-L-14.pt"), map_location="cpu").eval()
        model = VisionTransformer(224, 14, 1024, 24, 16, 768).float().eval()
        model.load_state_dict({name[7:]: value for name, value in checkpoint.state_dict().items() if name.startswith("visual.")})
        del checkpoint
        gc.collect()
        model = model.cuda().requires_grad_(False)
        cache = [[] for _ in range(24)]
        # 每图缓存 CLS token 与均匀采样的 15 个 patch token，控制 24 层缓存的内存量。
        token_ids = torch.tensor([0] + np.linspace(1, 256, 15, dtype=int).tolist(), device="cuda")
        hooks = []
        for index, block in enumerate(model.transformer.resblocks):
            # 默认参数 index=index 固定当前层编号，避免 hook 闭包把所有层都写进最后一个缓存。
            def capture(module, inputs, value, index=index):
                cache[index].append(value.detach().reshape(257, 4096)[token_ids].half().cpu())
            hooks.append(block.mlp.gelu.register_forward_hook(capture))
        mean = np.array([.48145466, .4578275, .40821073], np.float32)
        std = np.array([.26862954, .26130258, .27577711], np.float32)
        update("calibration_cache")
        with zipfile.ZipFile(root / "assets/dataset-resized.zip") as archive, torch.no_grad():
            for index, name in enumerate(sources):
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cannot decode calibration image")
                rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
                value = ((rgb.astype(np.float32) / 255 - mean) / std).astype(np.float16).astype(np.float32)
                model(torch.from_numpy(value.transpose(2, 0, 1)[None].copy()).cuda())
                if (index + 1) % 20 == 0:
                    update("calibration_cache", images_seen=index + 1)
        for hook in hooks:
            hook.remove()
        for index, record in enumerate(contract["blocks"]):
            update("rounding_block_" + str(index))
            block = model.transformer.resblocks[index]
            weight = block.mlp.c_proj.weight.detach()
            bias = block.mlp.c_proj.bias.detach()
            values = torch.cat(cache[index]).float().cuda()
            cache[index] = []
            qp = record["weight"]
            scale = torch.tensor(qp["scale"], device="cuda")[:, None]
            zero = torch.tensor(qp["zero_point"], device="cuda")[:, None]
            ap = record["activation"]
            quantized_input = fixed_quantize(values, ap["scale"][0], ap["zero_point"][0])
            nearest = fixed_quantize(weight, scale, zero)
            with torch.no_grad():
                target = F.linear(values, weight, bias)
                nearest_output = F.linear(quantized_input, nearest, bias)
                nearest_error = F.mse_loss(nearest_output, target).item()
                nearest_bias = bias + (target - nearest_output).mean(0)
                nearest_bias_error = F.mse_loss(F.linear(quantized_input, nearest, nearest_bias), target).item()
            rounding = AdaptiveRounding(weight, qp["scale"], qp["zero_point"])
            optimizer = torch.optim.Adam([rounding.alpha], lr=1e-3)
            for step in range(args.steps):
                ids = torch.randperm(len(values), device="cuda")[:args.batch_tokens]
                optimizer.zero_grad(set_to_none=True)
                # 在量化输入上重建原浮点线性层输出，而不是只让权重自身逐元素误差最小。
                reconstruction = F.mse_loss(F.linear(quantized_input[ids], rounding(), bias), target[ids])
                # 前 20% 步数只拟合重建；之后逐步改变正则温度，推动软舍入趋向 0/1。
                progress = max(0., (step / args.steps - .2) / .8)
                temperature = 20 - 18 * progress
                regularizer = 1 - (2 * (rounding.soft_targets() - .5).abs()).pow(temperature)
                loss = reconstruction / max(nearest_error, 1e-8) + (.1 * regularizer.mean() if progress > 0 else 0.)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite reconstruction loss")
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                hard = rounding(hard=True)
                after = F.linear(quantized_input, hard, bias)
                # 逐输出通道取残差均值作为 bias 修正，不改变权重形状或新增部署算子。
                correction = (target - after).mean(0)
                hard_error = F.mse_loss(after, target).item()
                corrected_error = F.mse_loss(after + correction, target).item()
                changed = (hard != nearest).float().mean().item()
            artifact = output / (f"block{index:02d}.npz")
            np.savez(artifact, weight=hard.detach().cpu().numpy(), bias=(bias + correction).cpu().numpy(),
                nearest_corrected_bias=nearest_bias.cpu().numpy(), original_bias=bias.cpu().numpy())
            info["blocks"].append({"block": index, "nearest_mse": nearest_error,
                "nearest_bias_corrected_mse": nearest_bias_error, "adaround_mse": hard_error,
                "adaround_bias_corrected_mse": corrected_error, "rounding_changed_fraction": changed,
                "artifact_sha": sha(artifact), "rows": len(values)})
            update("block_completed")
            del values, quantized_input, target, nearest_output, nearest, rounding, optimizer, hard, after
            gc.collect()
            torch.cuda.empty_cache()
        update("completed", completed=True)
    except Exception as error:
        update("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
