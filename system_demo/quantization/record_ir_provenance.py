"""Fingerprint source/calibration for safe reuse of this experiment's SDK step1 IR."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(root, algorithm="normal"):
    if algorithm not in {"normal", "kl_divergence", "mmse"}:
        raise ValueError("unsupported calibration algorithm: " + algorithm)
    source = root / "artifacts/clip_visual_reference_fp32.onnx"
    dataset = root / "artifacts/calibration.txt"
    inputs = [Path(line) for line in dataset.read_text().splitlines() if line]
    return {"source_sha256": digest(source), "dataset_list_sha256": digest(dataset),
            "calibration_inputs": [{"name": path.name, "sha256": digest(path)} for path in inputs],
            "toolkit_version": version("rknn-toolkit2"),
            "target_platform": "rk3588", "algorithm": algorithm, "quantized_method": "channel",
            "optimization_level": 3, "mean": [0, 0, 0], "std": [1, 1, 1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/manual_hybrid_20261010/ir_provenance.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / args.output
    current = fingerprint(root)
    if output.exists():
        if json.loads(output.read_text()) != current:
            raise SystemExit("FAIL: existing IR source/calibration fingerprint differs")
        print("PASS: IR source/calibration fingerprint unchanged")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(current, indent=2))
        print("IR source/calibration fingerprint recorded")
