# SDK IR 改动范围验证：图结构必须完全一致，只有白名单 initializer 的数值允许变化。
# SDK 的自定义算子并不自动符合标准 ONNX checker；本模块验证结构保留，不验证算子执行语义。

"""Validate copied RKNN vendor IR without pretending it is standard ONNX.

This checks change scope, NOT vendor operator semantics. The real SDK step2,
simulator smoke test and board fidelity evaluation remain separate gates.
"""
import hashlib

import numpy as np
import onnx
from onnx import numpy_helper


# 返回结构 SHA、initializer 顺序，以及每个参数的形状/dtype/数值 SHA，供后续候选比对。
def snapshot_ir(model):
    # Keep every protobuf field (including attrs, metadata, opsets and signatures)
    # except initializer payloads. External-data representation may change on save.
    structure = onnx.ModelProto()
    # 按 protobuf 字段类型复制，保留元数据、opset、节点属性等信息，而不仅比较节点数量。
    for descriptor, value in model.ListFields():
        if descriptor.name != "graph":
            if descriptor.label == descriptor.LABEL_REPEATED:
                getattr(structure, descriptor.name).extend(value)
            elif descriptor.message_type is not None:
                getattr(structure, descriptor.name).CopyFrom(value)
            else:
                setattr(structure, descriptor.name, value)
    for descriptor, value in model.graph.ListFields():
        if descriptor.name != "initializer":
            if descriptor.label == descriptor.LABEL_REPEATED:
                getattr(structure.graph, descriptor.name).extend(value)
            elif descriptor.message_type is not None:
                getattr(structure.graph, descriptor.name).CopyFrom(value)
            else:
                setattr(structure.graph, descriptor.name, value)
    tensors = {}
    order = []
    for tensor in model.graph.initializer:
        if tensor.name in tensors:
            raise ValueError("duplicate initializer name")
        # 先解析参数实际数值，再计算哈希，避免内嵌数据与外部 data 文件的表示差异造成误报。
        array = numpy_helper.to_array(tensor)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError("unsupported or nonfinite initializer: " + tensor.name)
        tensors[tensor.name] = {
            "dims": tuple(tensor.dims),
            "dtype": tensor.data_type,
            "sha256": hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest(),
        }
        order.append(tensor.name)
    return {
        "structure_sha256": hashlib.sha256(
            structure.SerializeToString(deterministic=True)
        ).hexdigest(),
        "initializer_order": order,
        "initializers": tensors,
    }


# allowed_initializers 是允许修改的参数名集合；不要求每个白名单参数都必须改变。
def validate_ir(baseline, model, allowed_initializers):
    current = snapshot_ir(model)
    if current["structure_sha256"] != baseline["structure_sha256"]:
        raise ValueError("SDK IR structure/attributes/output contract changed")
    if current["initializer_order"] != baseline["initializer_order"]:
        raise ValueError("initializer membership/order changed")
    allowed = set(allowed_initializers)
    if not allowed <= set(baseline["initializers"]):
        raise ValueError("unknown allowed initializer")
    changed = []
    for name, before in baseline["initializers"].items():
        after = current["initializers"][name]
        # 形状和 dtype 即使在白名单中也不能变化；只允许指定参数的内容更新。
        if (after["dims"], after["dtype"]) != (before["dims"], before["dtype"]):
            raise ValueError("initializer shape/dtype changed: " + name)
        if after["sha256"] != before["sha256"]:
            if name not in allowed:
                raise ValueError("unauthorized initializer change: " + name)
            changed.append(name)
    return {
        "structure_sha256": current["structure_sha256"],
        "initializer_count": len(current["initializers"]),
        "allowed_initializers": sorted(allowed),
        "changed_initializers": changed,
        "vendor_semantics_validated": False,
        "requires_sdk_step2": True,
    }
