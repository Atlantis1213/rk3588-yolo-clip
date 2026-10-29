import unittest
import numpy as np
from evaluate_real_200 import task_metrics, classify


class RealTaskMetricsTest(unittest.TestCase):
    def test_five_class_confusion(self):
        classes = ["cycle", "harmful", "kitchen", "others", "empty"]
        report = task_metrics(classes, ["cycle", "harmful", "kitchen", "empty", "empty"], classes)
        self.assertEqual(report["correct"], 4)
        self.assertEqual(report["accuracy"], .8)
        self.assertEqual(report["confusion"][3][4], 1)

    def test_empty_is_not_assigned_from_ground_truth(self):
        prediction, _, _ = classify(np.array([[1., 0.]]), np.eye(2), ["可回收物", "unknown"])
        self.assertEqual(prediction, ["cycle"])

    def test_reject_threshold_applied_to_similarity(self):
        prediction, _, _ = classify(np.array([[0., 1.]]), np.array([[1., 0.]]), ["可回收物"])
        self.assertEqual(prediction, ["empty"])


if __name__ == "__main__":
    unittest.main()
