"""Score a predeclared candidate on frozen real ROIs; never tune or promote it."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quantization"))
from evaluate_real_200 import load, classify, task_metrics
from numerical_fidelity import summarize, unit_normalize


def evaluate(base, candidate_directory):
    selected = json.loads((base / "dataset/manifest.json").read_text(encoding="utf-8"))
    names = [row["image"] for row in selected["samples"]]
    bank = [line.strip().split("\t") for line in (base.parent / "board_labels_vitl14.tsv").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    texts = np.fromfile(base.parent / "board_text_features_vitl14.f32", np.float32).reshape(-1, 768).astype(np.float64)
    texts = unit_normalize(texts)
    categories = [row[1] for row in bank]
    truth = [row["class"] for row in selected["samples"]]
    candidate = load(candidate_directory, names)
    predicted, best, scores = classify(candidate, texts, categories, reject=.22)
    report = {"protocol": "200 frozen YOLO top1 proposal ROI inputs; known empty whole image; conditional classification, not end-to-end accuracy",
        "reject_threshold": .22, "no_threshold_prompt_or_parameter_tuning": True,
        "task": task_metrics(truth, predicted, selected["classes"]), "fidelity": {}}
    for variant in ["onnx_fp32", "official_fp16", "official_kl120"]:
        reference = load(base / variant, names)
        reference_prediction, reference_best, reference_scores = classify(reference, texts, categories, reject=.22)
        report["fidelity"][variant] = {
            "raw_output_contract": summarize(reference, candidate),
            "explicit_unit_output_contract": summarize(unit_normalize(reference), unit_normalize(candidate)),
            "coarse_decision_agreement": float(np.mean(np.array(predicted) == np.array(reference_prediction))),
            "fine_label_agreement": float(np.mean(best == reference_best)),
            "cosine_score_rmse": float(np.sqrt(np.mean((scores - reference_scores) ** 2))),
        }
    timing = [row["run_ms"] for row in json.loads((candidate_directory / "manifest.json").read_text())["samples"]]
    report["single_context_run_ms"] = {"mean": float(np.mean(timing)), "median": float(np.median(timing)),
        "p95": float(np.percentile(timing, 95)), "scope": "3-core single context rknn_run; not full pipeline or 3-context throughput"}
    with (candidate_directory / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
