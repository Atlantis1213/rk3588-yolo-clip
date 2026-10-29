"""No detection gating; score all frozen non-YOLO inputs, including empty predictions."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "quantization"))
from evaluate_real_200 import load, classify, task_metrics
from numerical_fidelity import summarize, unit_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true", help="report only completed references/validated retrieved outputs")
    args = parser.parse_args()
    data = root / "results/real_eval_200_20261011/nonyolo_reviewed_crop_v2"
    selected = json.loads((data / "dataset/manifest.json").read_text(encoding="utf-8"))
    crop = json.loads((data / "crop_manifest.json").read_text())
    names = [row["image"] for row in selected["samples"]]
    truth = np.array([row["class"] for row in selected["samples"]])
    if len(names) != 200 or crop["sample_exclusion_count"] or crop["yolo_used"]:
        raise RuntimeError("all 200 non-YOLO samples required")
    bank = [line.strip().split("\t") for line in (data / "board_labels_vitl14.tsv").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    texts = unit_normalize(np.fromfile(data / "board_text_features_vitl14.f32", np.float32).reshape(-1, 768))
    variants = ["onnx_fp32", "deployed_historical", "official_fp16", "official_kl120", "repq_all"]
    if args.partial:
        status = json.loads((data / "test_status.json").read_text())
        reference_status = json.loads((data / "onnx_fp32/status.json").read_text())
        if not reference_status["completed"]:
            raise RuntimeError("complete floating reference required")
        variants = [name for name in variants if name == "onnx_fp32" or name in status["model_outputs"]]
    matrices = {variant: load(data / variant, names) for variant in variants}
    fp = matrices["onnx_fp32"]
    rows = []
    for variant in variants:
        values = matrices[variant]
        predicted, best, scores = classify(values, texts, [row[1] for row in bank], reject=.22)
        predicted = np.array(predicted)
        metrics = task_metrics(truth.tolist(), predicted.tolist(), selected["classes"])
        row = {"variant": variant, "task": metrics,
            "nonempty_correct": int((predicted[truth != "empty"] == truth[truth != "empty"]).sum()),
            "nonempty_count": 160, "nonempty_accuracy": float(np.mean(predicted[truth != "empty"] == truth[truth != "empty"])),
            "empty_correct": int((predicted[truth == "empty"] == "empty").sum()), "empty_count": 40,
            "unit_contract_fp32_fidelity": summarize(unit_normalize(fp), unit_normalize(values)),
            "predictions": [{"image": name, "truth": str(truth[i]), "predicted": str(predicted[i]),
                "fine_label": bank[int(best[i])][0], "max_similarity": float(scores[i, best[i]])} for i, name in enumerate(names)]}
        if variant != "deployed_historical":
            row["raw_contract_fp32_fidelity"] = summarize(fp, values)
        rows.append(row)
    report = {"protocol": "CLIP-only on visually reviewed HSV or fixed conservative central crops; no YOLO",
        "count": 200, "nonempty_count": 160, "empty_count": 40, "reject_threshold": .22,
        "visual_review_corrections": crop["visual_review_correction_count"],
        "human_ground_truth_roi_annotations": False,
        "known_empty_branch": "test label selects full-frame input, classifier still predicts; not automatic empty routing",
        "test_set_used_for_training_calibration_or_score_tuning": False,
        "crop_protocol_adjusted_from_visual_quality_before_inference": True,
        "not_an_independent_new_blind_test_or_end_to_end_detection_score": True,
        "all_models_completed": not args.partial, "variants": rows}
    name = f"clip_only_partial_{len(variants)}.json" if args.partial else "clip_only_evaluation.json"
    with (data / name).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps([{key: value for key, value in row.items() if key in ["variant", "task", "nonempty_accuracy", "empty_correct"]}
        for row in rows], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
