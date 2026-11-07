# 验证公开校准集扩展：120 原始 + 120 新增，六类均衡，保持来源不交叉且文件哈希匹配。
# 返回严格排序的 240 个路径，保证不同转换实验使用完全相同的输入清单。

"""Validate the existing public 240-image calibration expansion before conversion."""
from collections import Counter
import json
from pathlib import Path


def expanded_calibration(root, original_paths, digest):
    protocol = root / "artifacts/method_sweep_20261011/protocol_v2"
    audit = json.loads((protocol / "verification.json").read_text())
    if (not audit["source_split_disjoint"] or not audit["generated_file_hashes_valid"]
            or audit["exact_resized_pixel_duplicate_count"]):
        raise ValueError("public calibration protocol verification failed")
    record = json.loads((protocol / "protocol.json").read_text())
    extra = json.loads((protocol / "extra_calibration_labels.json").read_text())
    if len(original_paths) != 120 or len(extra["samples"]) != 120:
        raise ValueError("120 original + 120 extra calibration required")
    counts = Counter(row["class"] for row in extra["samples"])
    if len(extra["classes"]) != 6 or counts != dict.fromkeys(extra["classes"], 20):
        raise ValueError("balanced extra public calibration required")
    # 按原始来源去重，避免不同派生文件实际来自同一张验证/测试图片。
    excluded = set(record["excluded_previous_sources"])
    sources = [row["source"] for row in extra["samples"]]
    if len(set(sources)) != 120 or set(sources) & excluded:
        raise ValueError("extra calibration source overlap")
    expected = []
    for row in extra["samples"]:
        if Path(row["image"]).name != row["image"]:
            raise ValueError("simple calibration basenames required")
        path = protocol / "extra_calibration" / row["image"]
        if digest(path) != row["file_sha256"]:
            raise ValueError("extra calibration SHA mismatch")
        expected.append(path)
    paths = [Path(line) for line in (protocol / "calibration_augmented_240.txt").read_text().splitlines() if line]
    # 不仅验证集合，还验证原始清单和新增清单的顺序，防止输入协议悄悄变化。
    if paths != list(original_paths) + expected or len(set(paths)) != 240:
        raise ValueError("expanded calibration order/membership differs")
    return paths
