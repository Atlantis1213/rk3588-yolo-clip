"""Validate copied RKNN vendor IR without pretending it is standard ONNX.

This checks change scope, NOT vendor operator semantics. The real SDK step2,
simulator smoke test and board fidelity evaluation remain separate gates.
"""
import hashlib

import numpy as np
import onnx
from onnx import numpy_helper


def snapshot_ir(model):
    # Keep every protobuf field (including attrs, metadata, opsets and signatures)
    # except initializer payloads. External-data representation may change on save.
    structure = onnx.ModelProto()
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
        array = numpy_helper.to_array(tensor)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError("unsupported or nonfinite initializer: " + tensor.name)
        tensors[tensor.name] = {
            "dims": tuple(tensor.dims), "dtype": tensor.data_type,
            "sha256": hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest(),
        }
        order.append(tensor.name)
    return {"structure_sha256": hashlib.sha256(structure.SerializeToString(deterministic=True)).hexdigest(),
        "initializer_order": order, "initializers": tensors}


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
        if (after["dims"], after["dtype"]) != (before["dims"], before["dtype"]):
            raise ValueError("initializer shape/dtype changed: " + name)
        if after["sha256"] != before["sha256"]:
            if name not in allowed:
                raise ValueError("unauthorized initializer change: " + name)
            changed.append(name)
    return {"structure_sha256": current["structure_sha256"], "initializer_count": len(current["initializers"]),
        "allowed_initializers": sorted(allowed), "changed_initializers": changed,
        "vendor_semantics_validated": False, "requires_sdk_step2": True}
