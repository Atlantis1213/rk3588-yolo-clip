"""Fixed, equal-sized layer groups and strict checks of actual matrix precision."""
import re

PARTIAL_KL_DOWN = {
    "protect_kl_mlp_down_early8_except_sdpa": tuple(range(0, 8)),
    "protect_kl_mlp_down_mid8_except_sdpa": tuple(range(8, 16)),
    "protect_kl_mlp_down_late8_except_sdpa": tuple(range(16, 24)),
}


def checked_blocks(blocks):
    blocks = tuple(blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(type(i) is not int or not 0 <= i < 24 for i in blocks):
        raise ValueError("blocks must be nonempty unique integers in [0,23]")
    return tuple(sorted(blocks))


def block_id(name):
    match = re.search(r"/resblocks\.(\d+)/", name)
    if match is None:
        raise ValueError("missing block index: " + name)
    return int(match.group(1))


def select_down_tensors(layers, blocks):
    selected = checked_blocks(blocks)
    indexed = {}
    for row in layers:
        if row["op"] == "Conv" and "/mlp/c_proj/" in row["tensor"]:
            index = block_id(row["tensor"])
            if index in indexed:
                raise ValueError("duplicate down-projection tensor for block " + str(index))
            indexed[index] = row["tensor"]
    if set(indexed) != set(range(24)):
        raise ValueError("expected exactly 24 complete down-projection block mappings")
    return [indexed[index] for index in selected]


def verify_partial_down_inventory(report, blocks):
    selected = set(checked_blocks(blocks))
    groups = {name: {} for name in ["down", "up", "qkv", "attn_proj"]}
    for line in report.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit() or not re.fullmatch(r"Conv\w*", fields[1]):
            continue
        full_name = fields[-1]
        if "/mlp/c_proj/" in full_name:
            group = "down"
        elif "/mlp/c_fc/" in full_name:
            group = "up"
        elif re.search(r"/attn/MatMul(?:#|_output)", full_name):
            group = "qkv"
        elif "/attn/Gemm" in full_name:
            group = "attn_proj"
        else:
            continue
        index = block_id(full_name)
        if index in groups[group]:
            raise ValueError(f"duplicate actual {group} matrix in block {index}")
        groups[group][index] = (fields[2], fields[3])
    for group, actual in groups.items():
        if set(actual) != set(range(24)):
            raise ValueError("actual matrix block coverage incomplete: " + group)
        for index, precision in actual.items():
            expected = ("FLOAT16", "NPU") if group == "down" and index in selected else ("INT8", "NPU")
            if precision != expected:
                raise ValueError(f"actual {group} block {index}: {precision}, expected {expected}")
    return {"selected_down_fp16_npu": len(selected), "other_down_int8_npu": 24 - len(selected),
        "up_int8_npu": 24, "qkv_int8_npu": 24, "attn_proj_int8_npu": 24}
