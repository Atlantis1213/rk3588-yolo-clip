"""Rebuild the official image tower, compare reference outputs and export fixed-shape ONNX."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from clip.model import VisionTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True, help="dump_clip_embeddings output, same input bits")
parser.add_argument("--skip-export", action="store_true")
parser.add_argument("--experimental-reference", action="store_true", help="use a separate official reference when historical RKNN differs")
args = parser.parse_args()
root = Path(__file__).resolve().parent
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

checkpoint = torch.jit.load(str(root / "assets/ViT-L-14.pt"), map_location="cpu").eval()
state = {key[len("visual."):]: value for key, value in checkpoint.state_dict().items() if key.startswith("visual.")}
visual = VisionTransformer(input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
visual.load_state_dict(state, strict=True)
visual = visual.float().eval()
del state, checkpoint
gc.collect()

baseline = Path(args.baseline)
manifest = json.loads((baseline / "manifest.json").read_text())
original = np.fromfile(baseline / "embeddings.f32", dtype=np.float32).reshape(-1, manifest["embedding_dim"])
reference, similarities = [], []
with torch.no_grad():
    for index, sample in enumerate(manifest["samples"]):
        data = np.fromfile(baseline / "inputs_fp16" / (sample["image"] + ".bin"), dtype=np.float16)
        data = data.reshape(1, 224, 224, 3).astype(np.float32).transpose(0, 3, 1, 2).copy()
        embedding = visual(torch.from_numpy(data)).numpy().reshape(-1)
        reference.append(embedding)
        cosine = float(embedding @ original[index] / (np.linalg.norm(embedding) * np.linalg.norm(original[index])))
        similarities.append({"image": sample["image"], "cosine_official_to_deployed": cosine})
        print(similarities[-1], flush=True)
root.joinpath("artifacts").mkdir(exist_ok=True)
np.save(root / "artifacts/reference_embeddings.npy", np.array(reference))
(root / "artifacts/reference_comparison.json").write_text(json.dumps(similarities, indent=2))
same_reference = min(s["cosine_official_to_deployed"] for s in similarities) >= 0.99
(root / "artifacts/reference_provenance.json").write_text(json.dumps({
    "official_checkpoint": "OpenAI ViT-L/14", "matches_deployed_threshold": same_reference,
    "experimental_reference": args.experimental_reference,
    "note": "Quantized candidates must be compared against a newly converted FP16 reference from this ONNX, not claimed to retain historical unknown weights."}, indent=2))
if not same_reference and not args.experimental_reference:
    raise RuntimeError("official reference materially differs from deployed model; do not label it same-weight quantization")
if not args.skip_export:
    output = root / "artifacts/clip_visual_reference_fp32.onnx"
    with torch.no_grad():
        torch.onnx.export(visual, torch.zeros(1, 3, 224, 224), str(output), opset_version=17,
                          input_names=["pixel_values"], output_names=["image_embedding"],
                          do_constant_folding=True, dynamic_axes=None)
    print("ONNX exported: " + str(output), flush=True)
