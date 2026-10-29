import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate_protocol_v2 import load_dump, compare


class ProtocolMetricTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.labels = {"classes": ["a", "b"], "samples": [
            {"image": "a.png", "class": "a"}, {"image": "b.png", "class": "b"}]}
        self.texts = np.eye(2)

    def tearDown(self):
        self.temporary.cleanup()

    def dump(self, names, values):
        (self.root / "manifest.json").write_text(json.dumps({"embedding_dim": 2,
            "samples": [{"image": name, "run_ms": 1} for name in names]}))
        np.asarray(values, np.float32).tofile(self.root / "embeddings.f32")
        return load_dump(self.root, self.labels, self.texts)

    def test_known_accuracy_and_paired_loss(self):
        base = self.dump(["a.png", "b.png"], np.eye(2))
        other = self.dump(["a.png", "b.png"], [[1, 0], [1, 0]])
        report = compare(base, other)
        self.assertEqual(base[6]["accuracy"], 1.)
        self.assertEqual(report["accuracy_delta_pp"], -50.)
        self.assertEqual(report["paired_losses"], 1)

    def test_incomplete_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete split"):
            self.dump(["a.png"], [[1, 0]])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete split"):
            self.dump(["a.png", "a.png"], np.eye(2))

    def test_byte_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "byte count"):
            self.dump(["a.png", "b.png"], [[1, 0]])

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid embeddings"):
            self.dump(["a.png", "b.png"], [[np.nan, 0], [0, 1]])

    def test_pair_order_rejected(self):
        base = self.dump(["a.png", "b.png"], np.eye(2))
        other = self.dump(["b.png", "a.png"], np.eye(2))
        with self.assertRaisesRegex(ValueError, "order"):
            compare(base, other)


if __name__ == "__main__":
    unittest.main()
