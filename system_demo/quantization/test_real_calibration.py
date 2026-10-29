import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from prepare_real_calibration import select_samples


class RealCalibrationTest(unittest.TestCase):
    def test_eval_sources_file_and_pixels_excluded_and_seed_is_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classes = ["cycle", "harmful", "kitchen", "others", "empty"]
            generator = np.random.default_rng(1)
            rows = []
            for category in classes:
                (root / category).mkdir()
                for i in range(4):
                    image = generator.integers(0, 256, (8, 8, 3), dtype=np.uint8)
                    path = root / category / f"{i}.png"
                    ok, encoded = cv2.imencode(".png", image)
                    self.assertTrue(ok)
                    path.write_bytes(encoded.tobytes())
                    if i == 0:
                        rows.append({"source": f"{category}/0.png", "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "original_pixels_sha256": hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()})
                        # Different container bytes, exactly same decoded pixels.
                        ok, bmp = cv2.imencode(".bmp", image)
                        (root / category / "copy.bmp").write_bytes(bmp.tobytes())
                (root / category / "byte_copy.png").write_bytes((root / category / "0.png").read_bytes())
            exclusion = {"source_root": str(root), "classes": classes, "samples": rows * 40}
            chosen = select_samples(root, exclusion, per_class=2)
            self.assertEqual(chosen, select_samples(root, exclusion, per_class=2))
            self.assertEqual(len(chosen), 10)
            self.assertTrue(all(row["source"].endswith(("1.png", "2.png", "3.png")) for row in chosen))
            with self.assertRaisesRegex(ValueError, "insufficient"):
                select_samples(root, exclusion, per_class=4)

    def test_invalid_exclusion_manifest_rejected(self):
        with self.assertRaises(ValueError):
            select_samples(Path("missing"), {"samples": []})


if __name__ == "__main__":
    unittest.main()
