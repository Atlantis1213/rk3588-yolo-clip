import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_real_candidate import evaluate


class RealCandidateEvaluationTest(unittest.TestCase):
    def test_fixed_bank_reject_empty_and_explicit_contract_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "proposal_diagnostic"
            (base / "dataset").mkdir(parents=True)
            classes = ["cycle", "harmful", "kitchen", "others", "empty"]
            samples = [{"image": f"sample{i:03d}.jpg", "class": classes[i // 40]} for i in range(200)]
            (base / "dataset/manifest.json").write_text(json.dumps({"samples": samples, "classes": classes}))
            labels = "a\t可回收物\nb\t有害垃圾\nc\t厨余垃圾\nd\t其他垃圾\n"
            (root / "board_labels_vitl14.tsv").write_text(labels, encoding="utf-8")
            texts = np.eye(768, dtype=np.float32)[:4]
            texts.tofile(root / "board_text_features_vitl14.f32")
            unit = np.eye(768, dtype=np.float32)[np.repeat(np.arange(5), 40)]
            candidate = base / "candidate"
            for name in ["onnx_fp32", "official_fp16", "official_kl120", "candidate"]:
                output = base / name
                output.mkdir()
                (output / "manifest.json").write_text(json.dumps({"embedding_dim": 768,
                    "samples": [{"image": row["image"], "run_ms": 300.} for row in samples]}))
                (unit if name == "candidate" else unit * 10).tofile(output / "embeddings.f32")
            report = evaluate(base, candidate)
            self.assertEqual(report["task"]["correct"], 200)
            self.assertEqual(report["task"]["recall_by_class"]["empty"], 1.)
            self.assertGreater(report["fidelity"]["onnx_fp32"]["raw_output_contract"]["raw_rmse"], 0.)
            self.assertEqual(report["fidelity"]["onnx_fp32"]["explicit_unit_output_contract"]["raw_rmse"], 0.)
            self.assertEqual(report["fidelity"]["onnx_fp32"]["coarse_decision_agreement"], 1.)
            self.assertEqual(report["single_context_run_ms"]["mean"], 300.)
            with self.assertRaises(FileExistsError):
                evaluate(base, candidate)


if __name__ == "__main__":
    unittest.main()
