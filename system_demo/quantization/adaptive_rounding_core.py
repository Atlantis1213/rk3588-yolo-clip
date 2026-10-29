"""AdaRound-style hard-sigmoid rounding with fixed, explicit SDK channel scales."""
import torch
from torch import nn


class AdaptiveRounding(nn.Module):
    def __init__(self, weight, scale, zero):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=weight.dtype, device=weight.device).reshape(-1, 1)
        zero = torch.as_tensor(zero, dtype=weight.dtype, device=weight.device).reshape(-1, 1)
        if weight.ndim != 2 or scale.shape[0] != weight.shape[0] or zero.shape != scale.shape:
            raise ValueError("per-output channel dimensions differ")
        if not torch.isfinite(weight).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("finite weights and positive scales required")
        if not torch.isfinite(zero).all() or (zero != zero.round()).any():
            raise ValueError("finite integer zero points required")
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)
        value = weight / scale
        self.register_buffer("floor", value.floor())
        remainder = value - value.floor()
        fraction = ((remainder + .1) / 1.2).clamp(1e-6, 1 - 1e-6)
        self.alpha = nn.Parameter(torch.logit(fraction))

    def soft_targets(self):
        return (self.alpha.sigmoid() * 1.2 - .1).clamp(0, 1)

    def forward(self, hard=False):
        fraction = (self.alpha >= 0).to(self.floor.dtype) if hard else self.soft_targets()
        integers = (self.floor + fraction + self.zero).clamp(-128, 127)
        return (integers - self.zero) * self.scale


def fixed_quantize(value, scale, zero):
    return ((value / scale).round() + zero).clamp(-128, 127).sub(zero).mul(scale)
