import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from sdk_ir_validation import snapshot_ir, validate_ir


def fixture():
    graph = helper.make_graph([
        helper.make_node("exDataConvert", ["x", "weight"], ["y"], name="vendor", mode=1),
    ], "sdk_ir", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(np.array([1.], np.float32), name="weight"),
         numpy_helper.from_array(np.array([2.], np.float32), name="untouched")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


class SdkIrValidationTest(unittest.TestCase):
    def setUp(self):
        self.source = fixture()
        self.baseline = snapshot_ir(self.source)

    def test_vendor_operator_repro_and_unchanged_passes(self):
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(self.source)
        self.assertEqual(validate_ir(self.baseline, self.source, {"weight"})["changed_initializers"], [])

    def test_only_allowed_value_change_passes_after_external_roundtrip(self):
        model = copy.deepcopy(self.source)
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.array([3.], np.float32), "weight"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vendor.model"
            onnx.save_model(model, path, save_as_external_data=True,
                all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
            result = validate_ir(self.baseline, onnx.load(path), {"weight"})
        self.assertEqual(result["changed_initializers"], ["weight"])

    def test_unrelated_value_change_rejected(self):
        self.source.graph.initializer[1].CopyFrom(numpy_helper.from_array(np.array([3.], np.float32), "untouched"))
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            validate_ir(self.baseline, self.source, {"weight"})

    def test_attributes_topology_opset_and_output_contract_rejected(self):
        for modification in ("attribute", "input", "opset", "output", "domain"):
            with self.subTest(modification=modification):
                model = copy.deepcopy(self.source)
                if modification == "attribute":
                    model.graph.node[0].attribute[0].i = 2
                elif modification == "input":
                    model.graph.node[0].input[1] = "untouched"
                elif modification == "opset":
                    model.opset_import[0].version = 18
                elif modification == "output":
                    model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 2
                else:
                    model.graph.node[0].domain = "changed"
                with self.assertRaisesRegex(ValueError, "structure"):
                    validate_ir(self.baseline, model, {"weight"})

    def test_shape_dtype_and_nonfinite_rejected(self):
        for values in (np.array([1., 2.], np.float32), np.array([1.], np.float16), np.array([np.nan], np.float32)):
            model = copy.deepcopy(self.source)
            model.graph.initializer[0].CopyFrom(numpy_helper.from_array(values, "weight"))
            with self.assertRaises(ValueError):
                validate_ir(self.baseline, model, {"weight"})

    def test_missing_or_unknown_allowed_tensor_rejected(self):
        with self.assertRaises(ValueError):
            validate_ir(self.baseline, self.source, {"does_not_exist"})
        del self.source.graph.initializer[1]
        with self.assertRaises(ValueError):
            validate_ir(self.baseline, self.source, {"weight"})


if __name__ == "__main__":
    unittest.main()
