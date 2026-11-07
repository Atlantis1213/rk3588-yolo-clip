# 自适应舍入核心：固定 SDK 的每输出通道 scale/zero，只训练权重向下或向上取整的选择。
# weight 为 [out,in]；训练返回反量化浮点代理，最终 hard=True 固化为 INT8 网格。

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
        # floor 和量化参数不训练；alpha 通过可导的 hard-sigmoid 控制小数部分。
        self.register_buffer("floor", value.floor())
        remainder = value - value.floor()
        # 将权重原有小数部分反解为 sigmoid 的输入，使初始软舍入尽量还原浮点权重。
        fraction = ((remainder + .1) / 1.2).clamp(1e-6, 1 - 1e-6)
        self.alpha = nn.Parameter(torch.logit(fraction))

    # 把 sigmoid 拉伸到 [-0.1,1.1] 后截断到 [0,1]，允许训练结果到达两个端点。
    def soft_targets(self):
        return (self.alpha.sigmoid() * 1.2 - .1).clamp(0, 1)

    # 训练用连续小数，导出用 alpha 的符号二值化；加零点后限制到有符号 8 位范围。
    def forward(self, hard=False):
        fraction = (self.alpha >= 0).to(self.floor.dtype) if hard else self.soft_targets()
        integers = (self.floor + fraction + self.zero).clamp(-128, 127)
        return (integers - self.zero) * self.scale


def fixed_quantize(value, scale, zero):
    return ((value / scale).round() + zero).clamp(-128, 127).sub(zero).mul(scale)
