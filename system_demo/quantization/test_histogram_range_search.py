import unittest
import numpy as np
from histogram_range_search import affine_parameters, histogram_error, percentile_upper, mse_search


class HistogramRangeTest(unittest.TestCase):
    def test_asymmetric_parameters_include_zero_and_are_signed_int8(self):
        qp = affine_parameters(-1., 3.)
        self.assertAlmostEqual(qp["scale"], 4. / 255)
        self.assertEqual(qp["zero_point"], -64)
        self.assertEqual(affine_parameters(1., 3.)["min"], 0.)

    def test_percentile_counts_not_bin_index(self):
        self.assertEqual(percentile_upper([9, 1, 0, 0], [0., 4.], .9), 1.)
        self.assertEqual(percentile_upper([9, 1, 0, 0], [0., 4.], 1.), 2.)

    def test_mse_cannot_worsen_retained_baseline(self):
        counts = np.zeros(1024)
        counts[10:110] = 1000
        counts[-1] = 1
        baseline = affine_parameters(0., 10.)
        before = histogram_error(counts, [0., 10.], baseline)
        after = mse_search(counts, [0., 10.], 0., 10., baseline)
        self.assertLessEqual(after["histogram_qdq_mse"], before["histogram_qdq_mse"])

    def test_invalid_histograms_and_parameters_rejected(self):
        for counts, bounds in [([0, 0], [0, 1]), ([1, -1], [0, 1]), ([1, 1], [1, 0]), ([1, np.nan], [0, 1])]:
            with self.assertRaises(ValueError):
                histogram_error(counts, bounds, affine_parameters(0, 1))
        with self.assertRaises(ValueError):
            affine_parameters(0, 0)


if __name__ == "__main__":
    unittest.main()
