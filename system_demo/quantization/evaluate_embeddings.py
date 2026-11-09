# 早期公开六类评估：固定文本库对 embedding 分类，并与 FP16 基线比较数值和 top1 排名。
# 输入为每个 dump 的 manifest.json/embeddings.f32；新实验的完整划分检查另见 evaluate_protocol_v2.py。

"""Task metrics plus embedding/ranking preservation on the held-out public split."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--candidate", action="append", default=[])
parser.add_argument("--output", required=True)
parser.add_argument("--split", choices=["validation", "final_test"], default="validation")
args = parser.parse_args()
root = Path(__file__).resolve().parent
labels = json.loads(
    (
        root
        / "artifacts"
        / ("final_test_labels.json" if args.split == "final_test" else "eval_labels.json")
    ).read_text()
)
truth = {item["image"]: labels["classes"].index(item["class"]) for item in labels["samples"]}
texts = np.load(root / "artifacts/public_text_features.npy")
texts /= np.linalg.norm(texts, axis=1, keepdims=True)


# 返回原始特征、单位特征、文本得分和预测，使数值误差与分类误差能分别计算。
def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    values = np.fromfile(directory / "embeddings.f32", dtype=np.float32).reshape(
        -1, manifest["embedding_dim"]
    )
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise RuntimeError("nonfinite or empty embeddings: " + str(directory))
    normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
    # 矩阵 [样本,768] @ [768,类别] 得到每张图的余弦得分；argmax 是细类标签编号。
    scores = normalized @ texts.T
    predicted = scores.argmax(axis=1)
    y = np.array([truth[s["image"]] for s in manifest["samples"]])
    confusion = np.zeros((len(texts), len(texts)), dtype=np.int64)
    for actual, prediction in zip(y, predicted):
        confusion[actual, prediction] += 1
    # 混淆矩阵行是真实类别、列是预测类别；列和用于 precision，行和用于 recall。
    tp = confusion.diagonal()
    precision = np.divide(
        tp, confusion.sum(axis=0), out=np.zeros(len(tp)), where=confusion.sum(axis=0) != 0
    )
    recall = np.divide(
        tp, confusion.sum(axis=1), out=np.zeros(len(tp)), where=confusion.sum(axis=1) != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(tp)),
        where=(precision + recall) != 0,
    )
    result = {
        "directory": str(directory),
        "count": len(y),
        "accuracy": float((predicted == y).mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
        "mean_run_ms": float(np.mean([s["run_ms"] for s in manifest["samples"]])),
    }
    return manifest, values, normalized, scores, predicted, result


base_manifest, base_values, base_normalized, base_scores, base_predicted, base_result = load(
    args.baseline
)
results = [base_result]
for directory in args.candidate:
    manifest, values, normalized, scores, predicted, result = load(directory)
    if [s["image"] for s in manifest["samples"]] != [s["image"] for s in base_manifest["samples"]]:
        raise RuntimeError("evaluation sample order mismatch")
    cosine = (normalized * base_normalized).sum(axis=1)
    # accuracy_delta_pp 的单位是百分点；top1_agreement 是对基线预测的一致率，并非真实正确率。
    # speed_ratio 只来自串行 dump 平均耗时，不能代替相同条件下的流水线吞吐测试。
    result.update(
        embedding_cosine_mean=float(cosine.mean()),
        embedding_cosine_min=float(cosine.min()),
        raw_embedding_rmse=float(np.sqrt(np.mean((values - base_values) ** 2))),
        top1_agreement=float((predicted == base_predicted).mean()),
        accuracy_delta_pp=float((result["accuracy"] - base_result["accuracy"]) * 100),
        speed_ratio=base_result["mean_run_ms"] / result["mean_run_ms"],
    )
    results.append(result)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(
        {
            "dataset": "TrashNet " + args.split + " 120 samples, fixed six-class zero-shot bank",
            "classes": labels["classes"],
            "variants": results,
        },
        indent=2,
    )
)
print(json.dumps(results, indent=2), flush=True)
