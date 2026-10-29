import unittest
import numpy as np
from numerical_fidelity import summarize, unit_normalize


class NumericalFidelityTest(unittest.TestCase):
    def test_unit_output_contract_requires_explicit_normalization(self):
        original = [[.6, .8]]
        raw = [[3., 4.]]
        self.assertEqual(summarize(original, raw)["nrmse_l2_global"], 4.)
        self.assertEqual(summarize(unit_normalize(original), unit_normalize(raw))["raw_rmse"], 0.)

    def test_unit_contract_invalid_values_rejected(self):
        for values in ([[0., 0.]], [[np.nan, 1.]], [], [1., 2.]):
            with self.assertRaises(ValueError):
                unit_normalize(values)

    def test_identical_outputs(self):
        report = summarize([[1, 2], [3, 4]], [[1, 2], [3, 4]])
        self.assertEqual(report["raw_rmse"], 0.)
        self.assertEqual(report["sqnr_db_global"], "inf")
        self.assertAlmostEqual(report["cosine_mean"], 1.)

    def test_scale_error_is_not_hidden_by_cosine(self):
        report = summarize([[3, 4]], [[6, 8]])
        self.assertAlmostEqual(report["cosine_mean"], 1.)
        self.assertEqual(report["nrmse_l2_global"], 1.)
        self.assertEqual(report["sqnr_db_global"], 0.)
        self.assertEqual(report["normalized_embedding_l2_mean"], 0.)

    def test_orthogonal_values(self):
        report = summarize([[1, 0]], [[0, 1]])
        self.assertEqual(report["cosine_mean"], 0.)
        self.assertEqual(report["raw_rmse"], 1.)
        self.assertAlmostEqual(report["sqnr_db_global"], -3.01029995664)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            summarize([[1, 2]], [[1, 2], [3, 4]])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            summarize([[1, 2]], [[np.nan, 2]])

    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError):
            summarize([[1, 2]], [[0, 0]])


if __name__ == "__main__":
    unittest.main()
