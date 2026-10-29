"""Paired classification flips and text-score errors; no tuning on the final split."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
texts = np.load(root / "artifacts/public_text_features.npy")
texts /= np.linalg.norm(texts, axis=1, keepdims=True)


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    names = [row["image"] for row in manifest["samples"]]
    embeddings = np.fromfile(directory / "embeddings.f32", np.float32).reshape(-1, manifest["embedding_dim"])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = embeddings @ texts.T
    predicted = scores.argmax(axis=1)
    ordered = np.sort(scores, axis=1)
    return names, scores, predicted, ordered[:, -1] - ordered[:, -2]


def compare(split, directories):
    labels = json.loads((root / "artifacts" / ("eval_labels.json" if split == "validation" else "final_test_labels.json")).read_text())
    truth = {row["image"]: labels["classes"].index(row["class"]) for row in labels["samples"]}
    names, teacher_scores, teacher_prediction, _ = load(directories["fp16"])
    current_names, kl_scores, kl_prediction, kl_margin = load(directories["kl"])
    selected_names, selected_scores, selected_prediction, selected_margin = load(directories["selected"])
    if names != current_names or names != selected_names or len(names) != 120:
        raise RuntimeError("paired sample order/count differs")
    actual = np.asarray([truth[name] for name in names])
    kl_correct = kl_prediction == actual
    selected_correct = selected_prediction == actual
    changed = kl_prediction != selected_prediction
    return {"split": split, "count": len(names), "kl_correct": int(kl_correct.sum()),
        "selected_correct": int(selected_correct.sum()),
        "corrected_over_kl": int((~kl_correct & selected_correct).sum()),
        "broken_relative_to_kl": int((kl_correct & ~selected_correct).sum()),
        "top1_changed_count": int(changed.sum()),
        "kl_text_score_rmse_to_fp16": float(np.sqrt(np.mean((kl_scores - teacher_scores) ** 2))),
        "selected_text_score_rmse_to_fp16": float(np.sqrt(np.mean((selected_scores - teacher_scores) ** 2))),
        "disagreements": [{"image": name, "truth": labels["classes"][actual[index]],
            "fp16": labels["classes"][teacher_prediction[index]],
            "kl": labels["classes"][kl_prediction[index]],
            "selected": labels["classes"][selected_prediction[index]],
            "kl_top1_margin": float(kl_margin[index]), "selected_top1_margin": float(selected_margin[index])}
            for index, name in enumerate(names) if changed[index]]}


artifacts = root / "artifacts"
records = [compare("validation", {"fp16": artifacts / "quant_public_fp16_ieee",
    "kl": artifacts / "quant_sdpa_safe_sensitive_kl_except_sdpa_ieee",
    "selected": artifacts / "quant_sdpa_safe_protect_kl_mlp_down_mid8_except_sdpa_ieee"}),
    compare("final_test", {name: artifacts / "partial_kl_final_20261011" / name for name in ["fp16", "kl", "selected"]})]
destination = artifacts / "paired_partial_kl_ranking_20261011.json"
if destination.exists():
    raise RuntimeError("paired evidence exists; do not overwrite")
destination.write_text(json.dumps({"scope": "descriptive post-evaluation analysis only; "
    "no candidate reselection or thresholds fitted on final data", "records": records}, indent=2))
print(json.dumps([{key: value for key, value in row.items() if key != "disagreements"} for row in records], indent=2))
