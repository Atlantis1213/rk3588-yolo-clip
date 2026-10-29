import unittest
import numpy as np
from affine_reparameterization import smooth_fold, affine_fold, channel_parameters


class AffineReparameterizationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.normalized = rng.normal(size=(20, 8))
        self.gamma = rng.uniform(.2, 2, 8)
        self.beta = rng.normal(size=8)
        self.weight = rng.normal(size=(8, 12))
        self.bias = rng.normal(size=12)

    def evaluate(self, parameters):
        gamma, beta, weight, bias = parameters
        return (self.normalized * gamma + beta) @ weight + bias

    def test_smooth_float_equivalence(self):
        expected = self.evaluate((self.gamma, self.beta, self.weight, self.bias))
        actual = self.evaluate(smooth_fold(self.gamma, self.beta, self.weight, self.bias, np.linspace(.1, 10, 8)))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_affine_float_equivalence(self):
        expected = self.evaluate((self.gamma, self.beta, self.weight, self.bias))
        actual = self.evaluate(affine_fold(self.gamma, self.beta, self.weight, self.bias,
            np.linspace(.01, .5, 8), np.arange(8) - 4, .08, -2))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_channel_quantizer_maps_to_one_layer_quantizer(self):
        x = self.normalized * self.gamma + self.beta
        scale, zero = channel_parameters(x.min(axis=0), x.max(axis=0))
        common_scale, common_zero = float(scale.mean()), int(np.rint(zero.mean()))
        ratio = scale / common_scale
        shift = scale * (zero - common_zero)
        transformed = (x + shift) / ratio
        q_channel = np.clip(np.rint(x / scale) + zero, -128, 127)
        q_layer = np.clip(np.rint(transformed / common_scale) + common_zero, -128, 127)
        np.testing.assert_array_equal(q_layer, q_channel)
        restored = ((q_layer - common_zero) * common_scale) * ratio - shift
        np.testing.assert_allclose(restored, (q_channel - zero) * scale, rtol=1e-12, atol=1e-12)

    def test_no_parameter_shape_change(self):
        original = (self.gamma, self.beta, self.weight, self.bias)
        result = smooth_fold(*original, np.ones(8))
        self.assertEqual([value.shape for value in result], [value.shape for value in original])

    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            smooth_fold(self.gamma, self.beta, self.weight, self.bias, np.zeros(8))


if __name__ == "__main__":
    unittest.main()
