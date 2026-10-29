"""Guard against silently protecting the wrong blocks or unintended matrices."""
import unittest
from partial_mlp_policy import PARTIAL_KL_DOWN, checked_blocks, select_down_tensors, verify_partial_down_inventory


def layer_rows():
    return [{"op": "Conv", "tensor": f"/transformer/resblocks/resblocks.{index}/mlp/c_proj/Add_output_0-rs"}
            for index in range(24)]


def inventory(selected):
    lines = []
    for index in range(24):
        prefix = f"Conv:/transformer/resblocks/resblocks.{index}"
        entries = [("ConvAdd" if index in selected else "Conv", "FLOAT16" if index in selected else "INT8", "/mlp/c_proj/MatMul#2"),
                   ("Conv", "INT8", "/mlp/c_fc/MatMul#2"),
                   ("Conv", "INT8", "/attn/MatMul#3"), ("Conv", "INT8", "/attn/Gemm#2")]
        for op, dtype, name in entries:
            lines.append(f"{len(lines)+1} {op} {dtype} NPU shapes timing {prefix}{name}")
    return "\n".join(lines)


class PartialMlpPolicyTest(unittest.TestCase):
    def test_groups_partition_24_blocks(self):
        self.assertEqual(sorted(i for group in PARTIAL_KL_DOWN.values() for i in group), list(range(24)))
        self.assertTrue(all(len(group) == 8 for group in PARTIAL_KL_DOWN.values()))

    def test_exact_selection_in_all_three_groups(self):
        for blocks in PARTIAL_KL_DOWN.values():
            selected = select_down_tensors(layer_rows(), blocks)
            self.assertEqual(len(selected), 8)
            for name, index in zip(selected, blocks):
                self.assertIn(f"/resblocks.{index}/", name)
            result = verify_partial_down_inventory(inventory(blocks), blocks)
            self.assertEqual(result["selected_down_fp16_npu"], 8)
            self.assertEqual(result["other_down_int8_npu"], 16)

    def test_invalid_selection_rejected(self):
        for blocks in [[], [0, 0], [-1], [24], [True]]:
            with self.assertRaises(ValueError):
                checked_blocks(blocks)

    def test_missing_mapping_rejected(self):
        with self.assertRaises(ValueError):
            select_down_tensors(layer_rows()[:-1], range(8))

    def test_wrong_protected_block_rejected(self):
        with self.assertRaises(ValueError):
            verify_partial_down_inventory(inventory(range(1, 9)), range(8))

    def test_unintended_up_precision_rejected(self):
        altered = inventory(range(8)).replace("2 Conv INT8 NPU", "2 Conv FLOAT16 NPU")
        with self.assertRaises(ValueError):
            verify_partial_down_inventory(altered, range(8))

    def test_cpu_target_rejected(self):
        altered = inventory(range(8)).replace("3 Conv INT8 NPU", "3 Conv INT8 CPU")
        with self.assertRaises(ValueError):
            verify_partial_down_inventory(altered, range(8))

    def test_duplicate_actual_matrix_rejected(self):
        report = inventory(range(8))
        with self.assertRaises(ValueError):
            verify_partial_down_inventory(report + "\n" + report.splitlines()[0], range(8))


if __name__ == "__main__":
    unittest.main()
