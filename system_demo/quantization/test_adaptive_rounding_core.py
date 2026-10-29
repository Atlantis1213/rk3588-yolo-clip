import unittest
import torch
from adaptive_rounding_core import AdaptiveRounding, fixed_quantize


class RoundingTest(unittest.TestCase):
    def test_soft_initialization_reconstructs_in_range_weight(self):
        weight = torch.tensor([[.013, -.049, .122], [.41, -.55, .77]])
        rounding = AdaptiveRounding(weight, [.01, .1], [0, 0])
        torch.testing.assert_close(rounding(), weight, atol=1e-6, rtol=1e-5)

    def test_hard_weights_are_on_fixed_channel_grid(self):
        weight = torch.tensor([[.013, -.049, .122], [.41, -.55, .77]])
        rounding = AdaptiveRounding(weight, [.01, .1], [0, 0])
        values = rounding(hard=True) / rounding.scale
        torch.testing.assert_close(values, values.round(), atol=1e-6, rtol=0.)

    def test_gradient_reaches_rounding_logits(self):
        rounding = AdaptiveRounding(torch.tensor([[.014, .037]]), [.01], [0])
        rounding().square().sum().backward()
        self.assertGreater(rounding.alpha.grad.abs().sum().item(), 0.)

    def test_invalid_scales_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveRounding(torch.ones(2, 2), [0, 1], [0, 0])

    def test_clipping_is_signed_int8(self):
        torch.testing.assert_close(fixed_quantize(torch.tensor([-999., 999.]), 1., 0), torch.tensor([-128., 127.]))


if __name__ == "__main__":
    unittest.main()
