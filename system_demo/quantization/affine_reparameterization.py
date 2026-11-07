# LayerNorm → Linear 的等价参数折叠；此处矩阵方向为 [input,output]。
# 目的在于改变中间激活分布以便量化，同时在浮点下保持整个组合的输出。

"""Same-operator LN-to-linear affine folds; matrix orientation is [input,output]."""
import numpy as np


def validate(gamma, beta, weight, bias):
    if gamma.ndim != 1 or beta.shape != gamma.shape or weight.ndim != 2 or weight.shape[0] != len(gamma):
        raise ValueError("LayerNorm/linear dimensions differ")
    if bias.shape != (weight.shape[1],):
        raise ValueError("fold requires the existing linear bias")
    if not all(np.isfinite(value).all() for value in [gamma, beta, weight, bias]):
        raise ValueError("nonfinite original parameters")


# LN 输出除以各通道 scales，Linear 对应输入行乘 scales，二者在矩阵乘法中抵消。
def smooth_fold(gamma, beta, weight, bias, scales):
    validate(gamma, beta, weight, bias)
    scales = np.asarray(scales, np.float64)
    if scales.shape != gamma.shape or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("positive finite channel scales required")
    return ((gamma / scales).astype(gamma.dtype), (beta / scales).astype(beta.dtype),
        (weight * scales[:, None]).astype(weight.dtype), bias.copy())


def affine_fold(gamma, beta, weight, bias, channel_scale, channel_zero, common_scale, common_zero):
    validate(gamma, beta, weight, bias)
    channel_scale = np.asarray(channel_scale, np.float64)
    channel_zero = np.asarray(channel_zero, np.float64)
    if (channel_scale.shape != gamma.shape or channel_zero.shape != gamma.shape
        or not np.isfinite(channel_scale).all() or (channel_scale <= 0).any()
        or not np.isfinite(channel_zero).all() or not np.isfinite(common_scale) or common_scale <= 0):
        raise ValueError("invalid channel/common quantization parameters")
    if (channel_zero != np.rint(channel_zero)).any() or not np.isfinite(common_zero) or common_zero != round(common_zero):
        raise ValueError("zero points must be finite integers")
    # ratio 改变通道尺度，shift 改变通道偏移；新的 Linear bias 需扣除 shift@W。
    ratio = channel_scale / common_scale
    shift = channel_scale * (channel_zero - common_zero)
    correction = shift @ weight.astype(np.float64)
    return ((gamma / ratio).astype(gamma.dtype), ((beta + shift) / ratio).astype(beta.dtype),
        (weight * ratio[:, None]).astype(weight.dtype), (bias - correction).astype(bias.dtype))


# 范围先包含零，再映射到 [-128,127]；最小 scale 防止常量通道出现除零。
def channel_parameters(low, high):
    low = np.minimum(np.asarray(low, np.float64), 0.)
    high = np.maximum(np.asarray(high, np.float64), 0.)
    if low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all() or (high < low).any():
        raise ValueError("invalid observed channel ranges")
    scale = np.maximum((high - low) / 255, 1e-7)
    zero = np.clip(np.rint(-128 - low / scale), -128, 127).astype(np.int64)
    return scale, zero
