"""Independent timing, same SDK 2.3.2 and same host submission for every model."""
import json
import os
from pathlib import Path
import subprocess

output = Path("results/quant_speed_20261010"); output.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ, LD_LIBRARY_PATH=str(Path("quant_runtime232").resolve()))
variants = [("fp16", "clip_reference_fp16_232.rknn"), ("int8", "clip_int8_normal_232.rknn"),
            ("auto", "clip_auto_hybrid_232.rknn"), ("equalized", "clip_equalized_int8_232.rknn")]
input_tensor = "results/quant_public_fp16_ieee/inputs_fp16/cardboard_020.png.bin"
rows = []
for repeat in range(2):
    for name, model in (variants if repeat == 0 else list(reversed(variants))):
        command = ["./build/bench_clip_concurrency", model, input_tensor, "1", "all", "10", "none", "host"]
        print("START " + name, flush=True)
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
        if result.returncode: raise RuntimeError(result.stderr)
        report = json.loads(result.stdout[result.stdout.find("{"):])
        item = {"name": name, "repeat": repeat, "model": model, "report": report}
        rows.append(item)
        (output / f"{repeat}_{name}.json").write_text(json.dumps(item, indent=2))
        print(f"DONE {name}: {report['fps']:.3f}FPS", flush=True)
(output / "summary.json").write_text(json.dumps(rows, indent=2))
