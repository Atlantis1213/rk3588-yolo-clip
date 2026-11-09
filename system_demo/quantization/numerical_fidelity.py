# 在样本及输入严格对齐的前提下衡量 embedding 数值保真度；与分类正确率分别报告。
# raw 指标保留输出幅度差异，normalized 指标只比较方向，二者不能混用。

"""Matched-input numerical fidelity metrics; distinct from task recognition accuracy."""
import numpy as np


def unit_normalize(embeddings):
    """Explicitly match a unit-output contract; never silently alter raw metrics."""
    values = np.asarray(embeddings, np.float64)
    if values.ndim != 2 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError("finite nonempty embedding matrix required")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("zero-norm embedding")
    return values / norms


# 两矩阵必须以完全相同的样本顺序排列；本函数只验证形状/数值，样本身份由调用方核实。
def summarize(reference, candidate):
    reference = np.asarray(reference, np.float64)
    candidate = np.asarray(candidate, np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] == 0:
        raise ValueError("nonempty matched embedding matrices required")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite embedding")
    norms = np.linalg.norm(reference, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    if (norms == 0).any() or (candidate_norms == 0).any():
        raise ValueError("zero-norm embedding")
    difference = candidate - reference
    # 余弦忽略模长；若候选只整体缩放，余弦仍可能为 1，但 raw RMSE 会揭示差异。
    cosine = np.clip(np.sum(reference * candidate, axis=1) / (norms * candidate_norms), -1.0, 1.0)
    error_energy = np.sum(difference**2, axis=1)
    signal_energy = np.sum(reference**2, axis=1)
    # 每张图的误差 L2 除参考 L2，保留幅度差；normalized_l2 则先各自单位化，只反映方向。
    relative_l2 = np.sqrt(error_energy / signal_energy)
    normalized_l2 = np.linalg.norm(
        reference / norms[:, None] - candidate / candidate_norms[:, None], axis=1
    )
    # 全局能量比按所有元素累积；与每样本相对误差的算术平均权重不同。
    squared_error = float(error_energy.sum())
    # 全局 SQNR 使用总信号能量/总误差能量；完全相同输出以字符串 inf 记录，便于 JSON 保存。
    sqnr = float(10 * np.log10(signal_energy.sum() / squared_error)) if squared_error else "inf"
    return {
        "count": len(reference),
        "embedding_dim": reference.shape[1],
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "raw_mae": float(np.abs(difference).mean()),
        "raw_rmse": float(np.sqrt(np.mean(difference**2))),
        "raw_max_abs_error": float(np.abs(difference).max()),
        "nrmse_l2_global": float(np.sqrt(error_energy.sum() / signal_energy.sum())),
        "relative_l2_mean": float(relative_l2.mean()),
        "relative_l2_p95": float(np.percentile(relative_l2, 95)),
        "normalized_embedding_l2_mean": float(normalized_l2.mean()),
        "sqnr_db_global": sqnr,
        "norm_ratio_mean": float((candidate_norms / norms).mean()),
        "per_sample": [
            {
                "cosine": float(cosine[i]),
                "raw_rmse": float(np.sqrt(np.mean(difference[i] ** 2))),
                "relative_l2": float(relative_l2[i]),
                "normalized_l2": float(normalized_l2[i]),
            }
            for i in range(len(reference))
        ],
    }
