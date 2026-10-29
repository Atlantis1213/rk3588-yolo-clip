"""Freeze one strategy using validation accuracy and short timing, before final evaluation."""
import hashlib
import json
from pathlib import Path
from partial_mlp_policy import PARTIAL_KL_DOWN

root = Path(__file__).resolve().parent
workspace = root.parent
metrics = json.loads((root / "artifacts/accuracy_partial_kl_groups_20261011.json").read_text())
indexed = {Path(row["directory"].replace("\\", "/")).name: row for row in metrics["variants"]}
baseline = indexed["quant_sdpa_safe_sensitive_kl_except_sdpa_ieee"]
labels = json.loads((root / "artifacts/eval_labels.json").read_text())
final = json.loads((root / "artifacts/final_test_labels.json").read_text())
calibration = json.loads((root / "artifacts/dataset_provenance.json").read_text())
validation_sources = {row["source"] for row in labels["samples"]}
final_sources = {row["source"] for row in final["samples"]}
calibration_sources = set(calibration["calibration_sources"])
if (len(final_sources) != 120 or final_sources & validation_sources or final_sources & calibration_sources
    or validation_sources & calibration_sources):
    raise RuntimeError("calibration/validation/final split overlap or missing final samples")

records = []
for variant, blocks in PARTIAL_KL_DOWN.items():
    row = indexed["quant_sdpa_safe_" + variant + "_ieee"]
    if row["count"] != 120:
        raise RuntimeError("incomplete validation: " + variant)
    label = {"early8": "kl_down_early8", "mid8": "kl_down_mid8", "late8": "kl_down_late8"}[
        next(name for name in ["early8", "mid8", "late8"] if name in variant)]
    speed_file = workspace / "results/sdpa_safe_full_20261010" / ("speed_" + label) / "speed_summary.json"
    mean_fps = None
    if speed_file.exists():
        timing = [entry for entry in json.loads(speed_file.read_text()) if entry["label"] == variant + "_3spread"]
        if len(timing) != 2 or {entry["repeat"] for entry in timing} != {0, 1}:
            raise RuntimeError("incomplete two-round timing: " + variant)
        if any(entry["report"]["workers"] != 3 or entry["report"]["core_mode"] != "spread" for entry in timing):
            raise RuntimeError("unexpected timing configuration")
        mean_fps = sum(entry["report"]["fps"] for entry in timing) / 2
    elif row["accuracy"] > baseline["accuracy"]:
        raise RuntimeError("accuracy-improving candidate not yet timed: " + variant)
    records.append({"variant": variant, "blocks": list(blocks), "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"], "embedding_cosine": row["embedding_cosine_mean"], "mean_fps": mean_fps})

eligible = [row for row in records if row["accuracy"] > baseline["accuracy"] and row["mean_fps"] is not None and row["mean_fps"] >= 5]
if not eligible:
    raise RuntimeError("no accuracy-improving partial candidate meets the short-benchmark 5 FPS threshold")
selected = max(eligible, key=lambda row: (row["accuracy"], row["macro_f1"], row["embedding_cosine"], row["mean_fps"]))
build = root / "artifacts/sdpa_safe_full_20261010" / selected["variant"]
status = json.loads((build / "status.json").read_text())
digest = hashlib.sha256()
with (build / "clip.rknn").open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if not status.get("completed") or digest.hexdigest() != status["model_sha"]:
    raise RuntimeError("selected canonical model incomplete or changed")
report = {"selected": selected, "model_sha": status["model_sha"], "candidates": records,
    "selection_rule": "validation accuracy improves over KL; two-round three-context CLIP-only mean FPS >=5; "
        "rank by validation accuracy, macro-F1, embedding cosine, then FPS",
    "final_test_used_for_selection": False, "source_splits_disjoint": True,
    "note": "final split has historical baseline evaluations; not used for this partial-group selection; default model unchanged"}
destination = root / "artifacts/partial_kl_selected_20261011.json"
if destination.exists():
    raise RuntimeError("selection already frozen; do not overwrite after final evaluation")
destination.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
