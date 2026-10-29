"""Two independent standards: actual task accuracy and same-input numerical fidelity."""
import argparse
import json
from pathlib import Path
import numpy as np
from numerical_fidelity import summarize

CATEGORY = {"可回收物": "cycle", "有害垃圾": "harmful", "厨余垃圾": "kitchen", "其他垃圾": "others", "unknown": "empty"}


def task_metrics(truth, predicted, classes):
    actual = np.array([classes.index(value) for value in truth])
    predicted_ids = np.array([classes.index(value) for value in predicted])
    confusion = np.zeros((len(classes), len(classes)), np.int64)
    np.add.at(confusion, (actual, predicted_ids), 1)
    diagonal = confusion.diagonal()
    precision = np.divide(diagonal, confusion.sum(0), out=np.zeros(len(classes)), where=confusion.sum(0) != 0)
    recall = np.divide(diagonal, confusion.sum(1), out=np.zeros(len(classes)), where=confusion.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes)), where=precision + recall != 0)
    return {"count": len(truth), "correct": int((actual == predicted_ids).sum()),
        "accuracy": float((actual == predicted_ids).mean()), "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "confusion": confusion.tolist(),
        "recall_by_class": dict(zip(classes, recall.tolist()))}


def classify(embeddings, texts, categories, reject=.22):
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = normalized @ texts.T
    best = scores.argmax(axis=1)
    prediction = [CATEGORY[categories[index]] if scores[row, index] >= reject else "empty"
        for row, index in enumerate(best)]
    return prediction, best, scores


def load(directory, names):
    manifest = json.loads((directory / "manifest.json").read_text())
    if [row["image"] for row in manifest["samples"]] != names or manifest["embedding_dim"] != 768:
        raise ValueError("exact sample order and 768-dimensional outputs required")
    values = np.fromfile(directory / "embeddings.f32", np.float32)
    if values.size != len(names) * 768:
        raise ValueError("incomplete embedding file")
    values = values.reshape(len(names), 768).astype(np.float64)
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError("invalid embeddings")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/real_eval_200_20261011")
    parser.add_argument("--output", default="evaluation_two_standards.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = Path(args.data)
    if not data.is_absolute():
        data = root / data
    selected = json.loads((data / "dataset/manifest.json").read_text(encoding="utf-8"))
    board_status = json.loads((data / "board_status.json").read_text())
    auxiliary = board_status.get("confidence_threshold") == 0.0
    frozen = json.loads((data / "frozen/manifest.json").read_text())
    names = [row["image"] for row in selected["samples"]]
    if len(names) != 200 or names != [row["image"] for row in frozen["samples"]]:
        raise ValueError("frozen sample membership/order mismatch")
    bank_directory = data if (data / "board_labels_vitl14.tsv").exists() else data.parent
    bank = [line.strip().split("\t") for line in (bank_directory / "board_labels_vitl14.tsv").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if any(len(row) != 2 or row[1] not in CATEGORY for row in bank):
        raise ValueError("unexpected deployment label/category mapping")
    texts = np.fromfile(bank_directory / "board_text_features_vitl14.f32", np.float32).reshape(-1, 768).astype(np.float64)
    if len(texts) != len(bank) or not np.isfinite(texts).all() or (np.linalg.norm(texts, axis=1) == 0).any():
        raise ValueError("invalid fixed deployment text bank")
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    truth = [row["class"] for row in selected["samples"]]
    eligible = np.array([row["skip_yolo"] or row["detected"] for row in frozen["samples"]])
    nonempty = np.array([value != "empty" for value in truth])
    classes = selected["classes"]
    matrices, predictions, best_ids, scores = {}, {}, {}, {}
    task_rows = []
    variants = ["deployed_historical", "onnx_fp32", "official_fp16", "official_kl120"]
    for variant in variants:
        matrices[variant] = load(data / variant, names)
        prediction, best, current_scores = classify(matrices[variant], texts, [row[1] for row in bank])
        predictions[variant], best_ids[variant], scores[variant] = prediction, best, current_scores
        end_to_end = [value if eligible[index] else "empty" for index, value in enumerate(prediction)]
        report = task_metrics(truth, end_to_end, classes)
        report.update(variant=variant, yolo_detected_nonempty=int((eligible & nonempty).sum()),
            yolo_nonempty_count=int(nonempty.sum()), clip_eligible_count=int(eligible.sum()),
            diagnostic_full_frame_fallback_accuracy=float(np.mean(np.array(prediction) == np.array(truth))),
            empty_false_accept_rate=float(np.mean(np.array(prediction)[~nonempty] != "empty")))
        if eligible.any():
            report["clip_accuracy_on_eligible_inputs"] = float(np.mean(np.array(prediction)[eligible] == np.array(truth)[eligible]))
        task_rows.append(report)
    numeric = {}
    for first, second in [("onnx_fp32", "official_fp16"), ("official_fp16", "official_kl120"), ("onnx_fp32", "official_kl120")]:
        report = summarize(matrices[first], matrices[second])
        report.update(fine_label_top1_agreement=float((best_ids[first] == best_ids[second]).mean()),
            mapped_category_with_reject_agreement=float(np.mean(np.array(predictions[first]) == np.array(predictions[second]))),
            fixed_text_score_rmse=float(np.sqrt(np.mean((scores[first] - scores[second]) ** 2))),
            sample_names=names)
        if eligible.any():
            report["actual_pipeline_clip_inputs_only"] = summarize(matrices[first][eligible], matrices[second][eligible])
        numeric[first + "_to_" + second] = report
    output = data / args.output
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"dataset": "user real-scene 2025_7_25, 200 stratified images", "classes": classes,
            "reject_threshold": .22, "fixed_deployment_bank_labels": len(bank),
            "empty_bypass_note": "GT empty bypasses YOLO only, not CLIP prediction; this offline protocol is not an online empty-detection test",
            "detection_failures": "counted as unknown/empty in primary task; full-frame predictions shown only as diagnostics",
            "numeric_reference": "same official floating function, exact frozen ROI tensor values, real RK3588 outputs; historical weights excluded from same-weight error attribution",
            "sampling_limit": "exact duplicates removed, but not a held-out object-instance split and not a universal CLIP benchmark",
            "protocol": "auxiliary top1 proposal conditional classification, NOT online end-to-end accuracy" if auxiliary else "current board default detection threshold 0.05",
            "task_accuracy": task_rows, "quantization_fidelity": numeric}, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(task_rows, ensure_ascii=False, indent=2))
    print(json.dumps({name: {key: value for key, value in report.items()
        if key not in {"per_sample", "sample_names", "actual_pipeline_clip_inputs_only"}} for name, report in numeric.items()}, indent=2))


if __name__ == "__main__":
    main()
