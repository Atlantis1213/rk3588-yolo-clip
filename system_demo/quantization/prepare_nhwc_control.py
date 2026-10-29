"""One-variable calibration-layout control: identical values, NHWC file layout."""
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "artifacts"
destination = root / "calibration_nhwc"; destination.mkdir(exist_ok=True)
outputs = []
for line in (root / "calibration.txt").read_text().splitlines():
    if not line: continue
    path = Path(line)
    original = np.load(path)
    target = destination / path.name
    np.save(target, original.transpose(0, 2, 3, 1).copy())
    outputs.append(str(target))
(root / "calibration_nhwc.txt").write_text("\n".join(outputs) + "\n")
print("NHWC control files prepared; source values/calibration split unchanged")
