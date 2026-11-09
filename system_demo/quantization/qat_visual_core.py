# QAT 图像塔代理：权重按输出通道 fake quant，激活按 SDK 参数或校准范围 fake quant。
# fake quant 前向模拟 INT8 网格，反向提供训练梯度；它不是 RKNN 硬件数值的精确模拟。

"""Same-size CLIP visual tower with backend-inspired W8A8 fake quantization.

This training proxy is not claimed to be a bit-exact RKNN simulator. Canonical float
parameters can be exported without changing deployment graph size or architecture.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from clip_model_official import VisionTransformer


# 将卷积/线性权重展平到 [输出通道,其余维] 后独立求范围，scale 不参与梯度求导。
def fake_weight(weight):
    flat = weight.detach().reshape(weight.shape[0], -1)
    low = torch.minimum(flat.amin(dim=1), torch.zeros(weight.shape[0], device=weight.device))
    high = torch.maximum(flat.amax(dim=1), torch.zeros_like(low))
    scale = ((high - low) / 255).clamp_min(1e-8)
    zero = (-128 - torch.round(low / scale)).clamp(-128, 127).to(torch.int32)
    return torch.fake_quantize_per_channel_affine(weight, scale, zero, 0, -128, 127)


# 激活状态机：enabled 控制是否启用代理；calibrating 只统计范围；ready 表示已有冻结量化参数。
class ActivationQuant(nn.Module):
    def __init__(self, parameters, name):
        super().__init__()
        self.name = name
        values = parameters.get(name, {})
        scales = values.get('scale', [])
        zeros = values.get('zero_point', [])
        if len(scales) > 1:
            raise ValueError('activation proxy expects per-layer scale: ' + name)
        self.enabled = False
        self.ready = len(scales) == 1 and len(zeros) == 1
        self.calibrating = False
        self.observed_low = 0.0
        self.observed_high = 0.0
        self.scale = float(scales[0]) if self.ready else None
        self.zero = int(zeros[0]) if self.ready else None
        if self.ready and (self.scale <= 0 or not math.isfinite(self.scale)):
            raise ValueError('invalid activation scale: ' + name)

    def forward(self, value):
        if not self.enabled:
            return value
        # 校准阶段只累积观测范围并原样返回，完成后再冻结量化参数。
        if self.calibrating:
            data = value.detach().float()
            self.observed_low = min(self.observed_low, float(data.min()))
            self.observed_high = max(self.observed_high, float(data.max()))
            return value
        # 未给定 SDK 参数的站点可由首批输入初始化；正式训练入口会先遍历训练集校准缺失站点。
        # 验证时应保持 ready，不应借验证样本重新估计尺度。
        if not self.ready:
            data = value.detach().float()
            low = min(float(data.min()), 0.0)
            high = max(float(data.max()), 0.0)
            self.scale = max((high - low) / 255, 1e-8)
            self.zero = max(-128, min(127, -128 - round(low / self.scale)))
            self.ready = True
        return torch.fake_quantize_per_tensor_affine(
            value.float(), self.scale, self.zero, -128, 127
        )

    # 校准观测始终包含零，映射到 255 个间隔；常量范围以最小 scale 防止除零。
    def finish_calibration(self):
        self.scale = max((self.observed_high - self.observed_low) / 255, 1e-8)
        self.zero = max(-128, min(127, -128 - round(self.observed_low / self.scale)))
        self.ready = True
        self.calibrating = False


# 沿用原始权重和偏置；冻结权重可预先缓存，训练权重则要在前向重新 fake quant。
class QuantLinear(nn.Linear):
    def __init__(self, source, parameters, input_name, output_name):
        super().__init__(source.in_features, source.out_features, source.bias is not None)
        self.weight = source.weight
        self.bias = source.bias
        self.input_quant = ActivationQuant(parameters, input_name)
        self.output_quant = ActivationQuant(parameters, output_name)
        self.enabled = False
        # 缓存 detached 权重会切断该权重的梯度，只适合冻结层；当前入口仅训练 LN 与可选末端投影。
        self.register_buffer('cached_weight', torch.empty(0), persistent=False)

    def cache_weight(self):
        self.cached_weight = fake_weight(self.weight).detach()

    def forward(self, value):
        if not self.enabled:
            return F.linear(value, self.weight, self.bias)
        weight = self.cached_weight if self.cached_weight.numel() else fake_weight(self.weight)
        return self.output_quant(F.linear(self.input_quant(value), weight, self.bias))


# 只支持图像塔的无 mask 自注意力；按多头拆分 Q/K/V 后手动执行 scaled dot-product。
class QuantAttention(nn.MultiheadAttention):
    def __init__(self, source, parameters, prefix):
        super().__init__(source.embed_dim, source.num_heads, dropout=0.0)
        self.in_proj_weight = source.in_proj_weight
        self.in_proj_bias = source.in_proj_bias
        self.out_proj.weight = source.out_proj.weight
        self.out_proj.bias = source.out_proj.bias
        self.input_quant = ActivationQuant(
            parameters, prefix + '/ln_1/LayerNormalization_output_0_tp_rs'
        )
        a = prefix + '/attn/'
        self.qkv_quant = ActivationQuant(parameters, a + 'MatMul_output_0_rs_sw_gather')
        self.q_quant = ActivationQuant(parameters, a + 'Gather_2sl')
        self.k_quant = ActivationQuant(parameters, a + 'Reshape_7_output_0')
        self.v_quant = ActivationQuant(parameters, a + 'Reshape_8_output_0')
        self.projection_input_quant = ActivationQuant(parameters, a + 'Transpose_5_output_0_sw')
        self.output_quant = ActivationQuant(parameters, a + 'Gemm_output_0_mm')
        self.enabled = False
        self.register_buffer('cached_qkv', torch.empty(0), persistent=False)
        self.register_buffer('cached_projection', torch.empty(0), persistent=False)

    def cache_weight(self):
        self.cached_qkv = fake_weight(self.in_proj_weight).detach()
        self.cached_projection = fake_weight(self.out_proj.weight).detach()

    def forward(self, query, key, value, need_weights=False, attn_mask=None, **kwargs):
        if not self.enabled:
            return super().forward(
                query, key, value, need_weights=need_weights, attn_mask=attn_mask, **kwargs
            )
        if attn_mask is not None or query is not key or key is not value:
            raise ValueError('proxy supports only the unmasked visual self-attention')
        # Attention 输入采用 [token,batch,width]；拆头后为 [batch,head,token,head_dim]。
        length, batch, width = query.shape
        qkv = self.qkv_quant(F.linear(self.input_quant(query), self.cached_qkv, self.in_proj_bias))
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = self.q_quant(q), self.k_quant(k), self.v_quant(v)

        def split(x):
            return x.reshape(length, batch, self.num_heads, width // self.num_heads).permute(
                1, 2, 0, 3
            )

        q, k, v = split(q), split(k), split(v)
        # 每头维度 d=width/num_heads，QKᵀ 除 sqrt(d) 后 softmax，再乘 V。
        probabilities = F.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(width // self.num_heads), dim=-1
        )
        # 多头结果从 [B,H,L,D] 转回 [L,B,W]，再做输出投影；这里 softmax 本身未实现硬件整数模拟。
        attention = (probabilities @ v).permute(2, 0, 1, 3).reshape(length, batch, width)
        output = self.output_quant(
            F.linear(
                self.projection_input_quant(attention), self.cached_projection, self.out_proj.bias
            )
        )
        return output, probabilities.mean(dim=1) if need_weights else None


# 复用原模型模块和 Parameter，逐层替换为量化代理；输入 NCHW，输出未归一化的 [batch,768]。
class QuantVisual(VisionTransformer):
    def __init__(self, original, parameters):
        # Do not allocate a second full tower: transfer its existing modules and parameters.
        nn.Module.__init__(self)
        self.input_resolution = original.input_resolution
        self.output_dim = original.output_dim
        self.conv1 = original.conv1
        self.class_embedding = original.class_embedding
        self.positional_embedding = original.positional_embedding
        self.ln_pre = original.ln_pre
        self.transformer = original.transformer
        self.ln_post = original.ln_post
        self.proj = original.proj
        self.input_quant = ActivationQuant(parameters, 'pixel_values_int8')
        self.conv_quant = ActivationQuant(parameters, '/conv1/Conv_output_0')
        self.concat_quant = ActivationQuant(parameters, '/Transpose_output_0_sw-rs')
        self.cls_quant = ActivationQuant(parameters, '/Gather_2sl-rs_tp')
        self.proj_input_quant = ActivationQuant(parameters, '/Gather_output_0_tp_rs_sw')
        self.output_quant = ActivationQuant(parameters, 'image_embedding_int8')
        self.enabled = False
        self.checkpoint_blocks = False
        self.register_buffer('cached_conv', torch.empty(0), persistent=False)
        self.register_buffer('cached_proj', torch.empty(0), persistent=False)
        for i, block in enumerate(self.transformer.resblocks):
            prefix = '/transformer/resblocks/resblocks.' + str(i)
            block.attn = QuantAttention(block.attn, parameters, prefix)
            block.mlp.c_fc = QuantLinear(
                block.mlp.c_fc,
                parameters,
                prefix + '/ln_2/LayerNormalization_output_0_tp_rs',
                prefix + '/mlp/c_fc/MatMul_output_0_rs_sw',
            )
            block.mlp.c_proj = QuantLinear(
                block.mlp.c_proj,
                parameters,
                prefix + '/mlp/gelu/Mul_1_output_0_tp_rs',
                prefix + '/mlp/c_proj/Add_output_0-rs',
            )

    def enable_quantization(self, enabled):
        self.enabled = enabled
        for module in self.modules():
            if isinstance(module, (ActivationQuant, QuantLinear, QuantAttention)):
                module.enabled = enabled

    # 冻结矩阵预先量化一次；可训练末端 proj 不缓存，每次前向重新 fake quant 保留梯度路径。
    def cache_frozen_weights(self):
        self.cached_conv = fake_weight(self.conv1.weight).detach()
        self.cached_proj = (
            torch.empty(0, device=self.proj.device)
            if self.proj.requires_grad
            else fake_weight(self.proj.T.contiguous()).T.contiguous().detach()
        )
        for module in self.modules():
            if isinstance(module, (QuantLinear, QuantAttention)):
                module.cache_weight()

    # 仅导出标准浮点参数，排除代理量化缓存，保持部署图架构与模型尺寸。
    def canonical_state(self):
        return {name: parameter.detach().cpu() for name, parameter in self.named_parameters()}

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)
        x = self.conv_quant(
            F.conv2d(self.input_quant(x), self.cached_conv, stride=self.conv1.stride)
        )
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        # 16×16 个 patch token 前拼接一个 CLS，加入位置编码后共 257 个 token 进入 Transformer。
        cls = self.class_embedding.to(x.dtype)[None, None, :].expand(x.shape[0], 1, -1)
        x = self.concat_quant(torch.cat([cls, x], dim=1))
        x = self.ln_pre(x + self.positional_embedding).permute(1, 0, 2)
        for block in self.transformer.resblocks:
            if self.checkpoint_blocks and self.training:
                # 梯度检查点用重算换取训练显存；b=block 固定当前循环层，避免闭包捕获后续层。
                x = checkpoint(lambda value, b=block: b(value), x, use_reentrant=False)
            else:
                x = block(x)
        # 只抽取 CLS token，经最终 LN 和投影产生图像 embedding；余弦归一化由上层损失/分类完成。
        x = self.cls_quant(x.permute(1, 0, 2)[:, 0, :])
        x = self.proj_input_quant(self.ln_post(x))
        projection = (
            fake_weight(self.proj.T.contiguous()).T.contiguous()
            if self.proj.requires_grad
            else self.cached_proj
        )
        return self.output_quant(x @ projection)
