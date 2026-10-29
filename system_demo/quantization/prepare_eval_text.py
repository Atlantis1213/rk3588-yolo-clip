"""Fixed six-class public evaluation bank, separate from the deployment text bank."""
import json
from pathlib import Path
import numpy as np
import torch
import clip

root = Path(__file__).resolve().parent
classes = json.loads((root / "artifacts/eval_labels.json").read_text())["classes"]
torch.set_num_threads(4)
model, _ = clip.load(str(root / "assets/ViT-L-14.pt"), device="cpu", jit=False)
templates = ["a photo of {}.", "a photo of {} waste.", "a photo of a piece of {}."]
rows = []
with torch.no_grad():
    for label in classes:
        features = model.encode_text(clip.tokenize([t.format(label) for t in templates])).float()
        features = features / features.norm(dim=-1, keepdim=True)
        mean = features.mean(dim=0); mean = mean / mean.norm()
        rows.append(mean.numpy())
np.save(root / "artifacts/public_text_features.npy", np.array(rows, dtype=np.float32))
(root / "artifacts/public_prompt_bank.json").write_text(json.dumps({"classes": classes, "templates": templates,
    "note": "evaluation-only public six-class classifier; unchanged across FP16/INT8 variants"}, indent=2))
print("Public six-class text bank generated", flush=True)
