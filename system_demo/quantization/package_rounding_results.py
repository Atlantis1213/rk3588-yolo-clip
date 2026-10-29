"""Archive completed GPU pilot weights after checking every block digest."""
import json
from pathlib import Path
import zipfile
from reconstruct_rounding import sha

root = Path(__file__).resolve().parent
run = root / "runs/adaround_down_kl120_20261011"
status = json.loads((run / "status.json").read_text())
if not status["completed"] or status["phase"] != "completed" or len(status["blocks"]) != 24:
    raise RuntimeError("complete 24-block pilot required")
for index, row in enumerate(status["blocks"]):
    if row["block"] != index or sha(run / f"block{index:02d}.npz") != row["artifact_sha"]:
        raise RuntimeError("pilot block SHA mismatch")
output = run / "reconstructed_weights.zip"
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for filename in ["status.json", "provenance.json"] + [f"block{index:02d}.npz" for index in range(24)]:
        archive.write(run / filename, filename)
print(json.dumps({"archive": str(output), "sha256": sha(output), "bytes": output.stat().st_size,
    "nearest_mse_mean": sum(row["nearest_mse"] for row in status["blocks"]) / 24,
    "adaround_mse_mean": sum(row["adaround_mse"] for row in status["blocks"]) / 24,
    "adaround_bias_mse_mean": sum(row["adaround_bias_corrected_mse"] for row in status["blocks"]) / 24}, indent=2))
