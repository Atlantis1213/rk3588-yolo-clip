# 把已选混合精度策略应用到新校准 IR：只复制哪些 tensor 保留 FP16。
# 新校准的 scale、zero_point、范围必须保留，不能从旧 IR 一起覆盖。

"""Copy precision choices, never calibration parameters, to a fresh SDK IR."""
from copy import deepcopy


def reapply_precision_policy(fresh, reference):
    parameters = fresh["quantize_parameters"]
    reference_names = set(reference["quantize_parameters"])
    # 图中张量集合必须一致，否则旧策略名可能映射到错误或缺失的算子。
    if set(parameters) != reference_names:
        missing = sorted(reference_names - set(parameters))
        added = sorted(set(parameters) - reference_names)
        raise ValueError(f"calibration IR tensor names differ: missing={missing[:8]}, added={added[:8]}")
    choices = reference["custom_quantize_layers"]
    if not choices or any(dtype != "float16" for dtype in choices.values()):
        raise ValueError("expected a nonempty, exclusively FP16 reference precision policy")
    if any("sdpa" in name.lower() for name in choices):
        raise ValueError("reference policy protects the unsafe mixed-FP16 SDPA path")
    if not set(choices).issubset(parameters):
        raise ValueError("reference precision choices contain unmapped tensors")
    # 本实验要求 24 个融合 SDPA 输出仍为 INT8，防止误进入已诊断异常的混合 FP16 路径。
    sdpa = [name for name in parameters if name.endswith("_sdpa")]
    if len(sdpa) != 24 or not all(parameters[name]["dtype"] == "int8" for name in sdpa):
        raise ValueError("fresh IR must contain 24 INT8 fused SDPA outputs")
    result = deepcopy(fresh)
    result["custom_quantize_layers"] = dict(choices)
    return result
