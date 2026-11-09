# 严格完整数据集评估：标签、文本特征、模型输出按同一类别顺序配对。
# 同时计算任务准确率、macro-F1、embedding 偏差及相对于 FP16 的配对得失。

"""Strict complete-split task metrics and paired preservation against FP16."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


# 先验证样本集合、标签、输出字节数和文本矩阵尺寸，再归一化分类。返回值顺序供 compare 成对使用。
def load_dump(directory, labels, texts):
    directory = Path(directory)
    classes = labels["classes"]
    truth = {row["image"]: classes.index(row["class"]) for row in labels["samples"]}
    if len(truth) != len(labels["samples"]):
        raise ValueError("duplicate label images")
    manifest = json.loads((directory / "manifest.json").read_text())
    names = [row["image"] for row in manifest["samples"]]
    # 不允许漏样本、重复样本或额外样本，防止仅在成功子集上报告过高准确率。
    if len(set(names)) != len(names) or set(names) != set(truth):
        raise ValueError("dump does not contain the complete split exactly once")
    dim = manifest["embedding_dim"]
    if texts.shape != (len(classes), dim):
        raise ValueError("text bank shape/class count mismatch")
    values = np.fromfile(directory / "embeddings.f32", np.float32)
    if values.size != len(names) * dim:
        raise ValueError("embedding byte count mismatch")
    values = values.reshape(len(names), dim).astype(np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError("invalid embeddings")
    normalized = values / norms
    scores = normalized @ texts.T
    prediction = scores.argmax(axis=1)
    actual = np.array([truth[name] for name in names])
    confusion = np.zeros((len(classes), len(classes)), np.int64)
    # 混淆矩阵行是真实类别、列是预测类别；macro-F1 让每类等权。
    np.add.at(confusion, (actual, prediction), 1)
    diagonal = confusion.diagonal()
    precision = np.divide(
        diagonal, confusion.sum(0), out=np.zeros(len(classes)), where=confusion.sum(0) != 0
    )
    recall = np.divide(
        diagonal, confusion.sum(1), out=np.zeros(len(classes)), where=confusion.sum(1) != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(classes)),
        where=precision + recall != 0,
    )
    stats = {
        "directory": str(directory),
        "count": len(names),
        "correct": int((actual == prediction).sum()),
        "accuracy": float((actual == prediction).mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
        "recall_by_class": dict(zip(classes, recall.tolist())),
        "serial_dump_run_ms_mean": float(np.mean([row["run_ms"] for row in manifest["samples"]])),
    }
    return names, actual, prediction, values, normalized, scores, stats


def compare(base, candidate):
    names, truth, prediction, values, normalized, scores, stats = candidate
    # 矩阵逐行比较前要求文件顺序一致，不能仅验证集合相同。
    if names != base[0]:
        raise ValueError("paired dump sample order mismatch")
    cosine = (base[4] * normalized).sum(1)
    base_right = base[1] == base[2]
    right = truth == prediction
    # wins 是基线错而候选对，losses 是基线对而候选错；两者均对和均错的样本不进入该检验。
    wins = int((right & ~base_right).sum())
    losses = int((~right & base_right).sum())
    # McNemar 精确检验只统计基线和候选一对一的胜/负样本，观察改进是否有配对证据。
    discordant = wins + losses
    # 在胜负等可能的零假设下计算双侧精确二项尾概率，对应配对 McNemar 检验。
    p_value = (
        min(
            1.0,
            2
            * sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    stats.update(
        accuracy_delta_pp=(stats["accuracy"] - base[6]["accuracy"]) * 100,
        embedding_cosine_mean=float(cosine.mean()),
        embedding_cosine_min=float(cosine.min()),
        raw_embedding_rmse=float(np.sqrt(np.mean((values - base[3]) ** 2))),
        text_score_rmse=float(np.sqrt(np.mean((scores - base[5]) ** 2))),
        top1_agreement=float((prediction == base[2]).mean()),
        paired_wins=wins,
        paired_losses=losses,
        exact_mcnemar_p=p_value,
    )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol = root / "artifacts/method_sweep_20261011/protocol_v2"
    labels = json.loads((protocol / "development_labels.json").read_text())
    old_labels = json.loads((root / "artifacts/eval_labels.json").read_text())
    if labels["classes"] != old_labels["classes"]:
        raise ValueError("new class order differs from fixed text bank")
    texts = np.load(root / "artifacts/public_text_features.npy").astype(np.float64)
    if not np.isfinite(texts).all() or (np.linalg.norm(texts, axis=1) == 0).any():
        raise ValueError("invalid text bank")
    texts /= np.linalg.norm(texts, axis=1, keepdims=True)
    base = load_dump(args.baseline, labels, texts)
    variants = [base[6]] + [
        compare(base, load_dump(directory, labels, texts)) for directory in args.candidate
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(
            {
                "dataset": "TrashNet development v2, 144 images, fixed six-class zero-shot bank",
                "classes": labels["classes"],
                "locked_test_evaluated": False,
                "timing_note": "serial dump timings are descriptive only; use same-round concurrency benchmarks for speed",
                "variants": variants,
            },
            handle,
            indent=2,
        )
    print(json.dumps(variants, indent=2), flush=True)


if __name__ == "__main__":
    main()
