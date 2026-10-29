import json
from pathlib import Path
import tempfile
import unittest

from calibration_selection import expanded_calibration


class CalibrationSelectionTest(unittest.TestCase):
    def fixture(self, root):
        protocol = root / "artifacts/method_sweep_20261011/protocol_v2"
        protocol.mkdir(parents=True)
        classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        rows = [{"class": category, "source": f"source/{category}/{i}.jpg",
            "image": f"v2_{category}_{i:03d}.npy", "file_sha256": "verified"}
            for category in classes for i in range(20)]
        original = [root / f"old_{i}.npy" for i in range(120)]
        (protocol / "verification.json").write_text(json.dumps({"source_split_disjoint": True,
            "generated_file_hashes_valid": True, "exact_resized_pixel_duplicate_count": 0}))
        (protocol / "protocol.json").write_text(json.dumps({"excluded_previous_sources": ["old.jpg"]}))
        (protocol / "extra_calibration_labels.json").write_text(json.dumps({"classes": classes, "samples": rows}))
        all_paths = original + [protocol / "extra_calibration" / row["image"] for row in rows]
        (protocol / "calibration_augmented_240.txt").write_text("\n".join(map(str, all_paths)))
        return protocol, original, rows, classes

    def test_valid_expansion_and_modified_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, original, _, _ = self.fixture(root)
            self.assertEqual(len(expanded_calibration(root, original, lambda _: "verified")), 240)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                expanded_calibration(root, original, lambda _: "modified")
            with (protocol / "calibration_augmented_240.txt").open("a") as handle:
                handle.write("\nextra.npy")
            with self.assertRaisesRegex(ValueError, "order/membership"):
                expanded_calibration(root, original, lambda _: "verified")

    def test_source_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, original, rows, classes = self.fixture(root)
            rows[0]["source"] = "old.jpg"
            (protocol / "extra_calibration_labels.json").write_text(json.dumps({"classes": classes, "samples": rows}))
            with self.assertRaisesRegex(ValueError, "source overlap"):
                expanded_calibration(root, original, lambda _: "verified")

    def test_failed_deduplication_audit_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, original, _, _ = self.fixture(root)
            audit = json.loads((protocol / "verification.json").read_text())
            audit["exact_resized_pixel_duplicate_count"] = 1
            (protocol / "verification.json").write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "verification failed"):
                expanded_calibration(root, original, lambda _: "verified")


if __name__ == "__main__":
    unittest.main()
