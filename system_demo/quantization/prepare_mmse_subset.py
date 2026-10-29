from pathlib import Path

root = Path(__file__).resolve().parent / "artifacts"
groups = {}
for line in (root / "calibration.txt").read_text().splitlines():
    if line:
        label = Path(line).stem.split("_")[0]
        groups.setdefault(label, []).append(line)
selected = [path for label in sorted(groups) for path in groups[label][:5]]
(root / "calibration_mmse.txt").write_text("\n".join(selected) + "\n")
print(f"MMSE calibration subset: {len(selected)} samples, no validation/test inputs")
