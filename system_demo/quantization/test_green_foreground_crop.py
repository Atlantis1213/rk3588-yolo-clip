import sys
from pathlib import Path
import unittest
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from green_foreground_crop import foreground_crop


class ForegroundCropTest(unittest.TestCase):
    def test_green_cast_neutral_object_is_not_discarded_by_saturation_gate(self):
        image = self.green()
        cv2.rectangle(image, (60, 30), (100, 60), (100, 180, 110), -1)
        _, old = foreground_crop(image)
        box, new = foreground_crop(image, mode="saturated_green")
        self.assertTrue(old["fallback"])
        self.assertTrue(new["foreground_found"])
        self.assertLess(box[2] * box[3], image.shape[0] * image.shape[1])

    def green(self):
        image = np.zeros((100, 200, 3), np.uint8)
        image[:, :, 1] = 200
        return image

    def test_object_box_with_padding_contains_whole_object(self):
        image = self.green()
        cv2.rectangle(image, (60, 30), (100, 60), (0, 0, 255), -1)
        box, status = foreground_crop(image)
        x, y, width, height = box
        self.assertTrue(status["foreground_found"])
        self.assertLessEqual(x, 60)
        self.assertLessEqual(y, 30)
        self.assertGreaterEqual(x + width, 101)
        self.assertGreaterEqual(y + height, 61)
        self.assertLess(width * height, 20000)

    def test_no_foreground_preserves_full_frame(self):
        box, status = foreground_crop(self.green())
        self.assertEqual(box, [0, 0, 200, 100])
        self.assertTrue(status["fallback"])

    def test_empty_is_full_frame_not_assigned_prediction(self):
        box, status = foreground_crop(self.green(), empty=True)
        self.assertEqual(box, [0, 0, 200, 100])
        self.assertEqual(status["method"], "known_empty_full_frame")

    def test_largest_component_and_clipped_padding(self):
        image = self.green()
        cv2.rectangle(image, (0, 0), (45, 40), (255, 0, 0), -1)
        cv2.rectangle(image, (150, 60), (155, 65), (0, 0, 255), -1)
        box, _ = foreground_crop(image)
        self.assertEqual(box[:2], [0, 0])
        self.assertLess(box[2], 100)


if __name__ == "__main__":
    unittest.main()
