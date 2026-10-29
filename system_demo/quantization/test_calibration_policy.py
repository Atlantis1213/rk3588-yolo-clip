"""Regression checks for single-variable calibration policy transfer."""
from copy import deepcopy
import unittest
from calibration_policy import reapply_precision_policy


class CalibrationPolicyTest(unittest.TestCase):
    def setUp(self):
        parameters = {f"block{index}_sdpa": {"dtype": "int8", "scale": [0.5]} for index in range(24)}
        parameters["norm"] = {"dtype": "int8", "scale": [0.1]}
        parameters["projection"] = {"dtype": "int8", "scale": [0.2]}
        self.reference = {"quantize_parameters": parameters, "custom_quantize_layers": {"norm": "float16"}}
        self.fresh = deepcopy(self.reference)
        self.fresh["custom_quantize_layers"] = {}
        self.fresh["quantize_parameters"]["projection"]["scale"] = [0.01]

    def test_fresh_scales_retained_and_precision_choices_identical(self):
        before = deepcopy(self.fresh)
        result = reapply_precision_policy(self.fresh, self.reference)
        self.assertEqual(result["quantize_parameters"], before["quantize_parameters"])
        self.assertEqual(result["custom_quantize_layers"], self.reference["custom_quantize_layers"])
        self.assertNotEqual(result["quantize_parameters"], self.reference["quantize_parameters"])
        self.assertEqual(self.fresh, before)

    def test_tensor_name_change_rejected(self):
        del self.fresh["quantize_parameters"]["projection"]
        with self.assertRaisesRegex(ValueError, "tensor names differ"):
            reapply_precision_policy(self.fresh, self.reference)

    def test_fp16_sdpa_override_rejected(self):
        self.reference["custom_quantize_layers"]["block0_sdpa"] = "float16"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            reapply_precision_policy(self.fresh, self.reference)

    def test_missing_sdpa_rejected(self):
        for cfg in [self.fresh, self.reference]:
            del cfg["quantize_parameters"]["block23_sdpa"]
        with self.assertRaisesRegex(ValueError, "24 INT8"):
            reapply_precision_policy(self.fresh, self.reference)

    def test_non_int8_sdpa_rejected(self):
        self.fresh["quantize_parameters"]["block0_sdpa"]["dtype"] = "float16"
        with self.assertRaisesRegex(ValueError, "24 INT8"):
            reapply_precision_policy(self.fresh, self.reference)


if __name__ == "__main__":
    unittest.main()
