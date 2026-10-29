"""Fold calibrated pre-LN channel scales into affine LN and its sole following linear.

No inference operators are added. This is a SmoothQuant/RepQ-style experimental adaptation,
not a claim to reproduce every component of either published method.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from clip.model import VisionTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()
if not 0 <= args.alpha <= 1: raise ValueError("alpha must be in [0,1]")
root = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
checkpoint = torch.jit.load(str(root / "assets/ViT-L-14.pt"), map_location="cpu").eval()
state = {key[7:]: value for key, value in checkpoint.state_dict().items() if key.startswith("visual.")}
visual = VisionTransformer(224, 14, 1024, 24, 16, 768).float().eval()
visual.load_state_dict(state, strict=True)
del state, checkpoint
gc.collect()

modules = {}
for index, block in enumerate(visual.transformer.resblocks):
    modules[f"block{index}.ln1"] = (block.ln_1, block.attn.in_proj_weight, False)
    modules[f"block{index}.ln2"] = (block.ln_2, block.mlp.c_fc.weight, False)
modules["ln_post"] = (visual.ln_post, visual.proj, True)
maxima = {key: torch.zeros(1024) for key in modules}


def collector(name):
    def hook(module, inputs, output):
        maxima[name] = torch.maximum(maxima[name], output.detach().abs().reshape(-1, 1024).amax(dim=0).cpu())
    return hook


handles = [module.register_forward_hook(collector(name)) for name, (module, _, _) in modules.items()]
paths = [Path(line) for line in (root / "artifacts/calibration.txt").read_text().splitlines() if line]
golden_inputs, golden_outputs = [], []
with torch.no_grad():
    for index, path in enumerate(paths):
        tensor = torch.from_numpy(np.load(path).astype(np.float32))
        embedding = visual(tensor)
        if index < 3: golden_inputs.append(tensor); golden_outputs.append(embedding.clone())
        if (index + 1) % 20 == 0: print(f"Activation calibration {index+1}/{len(paths)}", flush=True)
for handle in handles: handle.remove()

records = []
with torch.no_grad():
    for name, (layer_norm, weight, transposed) in modules.items():
        weight_max = weight.detach().abs().amax(dim=1 if transposed else 0).clamp_min(1e-5)
        activation_max = maxima[name].clamp_min(1e-5)
        scales = (activation_max.pow(args.alpha) / weight_max.pow(1 - args.alpha)).clamp(0.01, 100)
        layer_norm.weight.div_(scales); layer_norm.bias.div_(scales)
        if transposed: weight.mul_(scales[:, None])
        else: weight.mul_(scales[None, :])
        records.append({"module": name, "min_scale": float(scales.min()), "max_scale": float(scales.max())})
    differences, cosines = [], []
    for tensor, expected in zip(golden_inputs, golden_outputs):
        actual = visual(tensor)
        differences.append(float((actual - expected).abs().max()))
        cosines.append(float(torch.nn.functional.cosine_similarity(actual, expected).item()))
    if min(cosines) < 0.99999 or max(differences) > 0.001:
        raise RuntimeError("float equivalence check failed after folding scales")
    output = root / "artifacts/clip_visual_equalized_fp32.onnx"
    torch.onnx.export(visual, torch.zeros(1, 3, 224, 224), str(output), opset_version=17,
                      input_names=["pixel_values"], output_names=["image_embedding"], dynamic_axes=None,
                      do_constant_folding=True)
(root / "artifacts/equalization.json").write_text(json.dumps({"alpha": args.alpha,
    "calibration_count": len(paths), "modules": records, "float_cosines": cosines,
    "float_max_abs_errors": differences, "extra_inference_operators": 0,
    "note": "LN_pre untouched because it also feeds residual paths; only sole LN->linear consumers are folded."}, indent=2))
print("Equivalent channel equalization exported: " + str(output), flush=True)
