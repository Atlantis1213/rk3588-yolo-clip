# 用校准激活直方图提出 INT8 范围候选：在截断误差与舍入误差之间取舍。
# 这是直方图中心点的 QDQ 误差近似，不是融合 NPU 算子的逐位模拟，也不使用测试集标签。

"""Calibration-only affine INT8 range proposals, not bit-exact fused NPU QDQ."""
import numpy as np


def histogram_values(counts, bounds):
    counts = np.asarray(counts, np.float64)
    bounds = np.asarray(bounds, np.float64)
    if (counts.ndim != 1 or counts.size < 2 or bounds.shape != (2,) or not np.isfinite(counts).all()
            or not np.isfinite(bounds).all() or (counts < 0).any() or counts.sum() <= 0 or bounds[1] <= bounds[0]):
        raise ValueError("finite nonnegative nonempty histogram and increasing bounds required")
    step = (bounds[1] - bounds[0]) / len(counts)
    # 以等宽 bin 中心代表该区间数值，counts 作为出现次数加权。
    return bounds[0] + (np.arange(len(counts)) + .5) * step, counts


def affine_parameters(lower, upper):
    lower, upper = min(float(lower), 0.), max(float(upper), 0.)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("nonzero finite quantization range required")
    scale = (upper - lower) / 255.
    zero = int(np.clip(np.rint(-128. - lower / scale), -128, 127))
    return {"min": lower, "max": upper, "scale": scale, "zero_point": zero}


def histogram_error(counts, bounds, parameters):
    values, weights = histogram_values(counts, bounds)
    scale, zero = parameters["scale"], parameters["zero_point"]
    if scale <= 0 or not np.isfinite(scale) or not -128 <= zero <= 127:
        raise ValueError("invalid INT8 scale/zero point")
    unbounded = np.rint(values / scale) + zero
    quantized = np.clip(unbounded, -128, 127)
    # 饱和到 INT8 后反量化，与原 bin 中心的差同时包含 clipping 和 rounding 误差。
    difference = (quantized - zero) * scale - values
    return {"histogram_qdq_mse": float(np.sum(weights * difference ** 2) / weights.sum()),
        "clipping_fraction": float(weights[(unbounded < -128) | (unbounded > 127)].sum() / weights.sum()),
        "zero_fraction": float(weights[quantized == zero].sum() / weights.sum())}


def percentile_upper(counts, bounds, percentile):
    _, weights = histogram_values(counts, bounds)
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0,1]")
    index = min(int(np.searchsorted(np.cumsum(weights), weights.sum() * percentile)), len(weights) - 1)
    return float(bounds[0] + (index + 1) * (bounds[1] - bounds[0]) / len(weights))


def mse_search(counts, bounds, lower, upper, baseline):
    start = max(percentile_upper(counts, bounds, .99), 1e-6)
    end = max(float(upper), start)
    # 保留原始范围作为候选，避免搜索结果比基线的校准 MSE 更差。
    candidates = [baseline] + [affine_parameters(lower, value) for value in np.linspace(start, end, 256)]
    # Also retain conventional percentile ranges, selected by calibration MSE only.
    candidates += [affine_parameters(lower, percentile_upper(counts, bounds, p)) for p in [.999, .9999, .99999]]
    scored = [(histogram_error(counts, bounds, parameters), parameters) for parameters in candidates]
    error, parameters = min(scored, key=lambda row: row[0]["histogram_qdq_mse"])
    return dict(parameters, **error, search_candidate_count=len(candidates))
