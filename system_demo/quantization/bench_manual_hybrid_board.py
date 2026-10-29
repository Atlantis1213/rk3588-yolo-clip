"""Board validation then independent timing, never concurrent NPU workloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="clip_manual_sensitive_232.rknn")
parser.add_argument("--skip-validation", action="store_true")
args = parser.parse_args()
environment = dict(os.environ, LD_LIBRARY_PATH=str(Path("quant_runtime232").resolve()))
output = Path("results/manual_hybrid_20261010")
output.mkdir(parents=True, exist_ok=True)
if not args.skip_validation:
    print("START manual hybrid 120-image validation", flush=True)
    result = subprocess.run(["./build/dump_clip_embeddings", args.model, "eval_png",
                             "results/quant_manual_sensitive_ieee", "host"], env=environment,
                            capture_output=True, text=True, timeout=240)
    (output / "validation.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    print("DONE manual hybrid validation", flush=True)
input_tensor = "results/quant_public_fp16_ieee/inputs_fp16/cardboard_020.png.bin"
configs = [("fp16_all", "clip_reference_fp16_232.rknn", 1, "all"),
           ("manual_all", args.model, 1, "all"), ("manual_3spread", args.model, 3, "spread")]
rows = []
for repeat in range(2):
    for name, model, workers, mode in (configs if repeat == 0 else list(reversed(configs))):
        print("START " + name, flush=True)
        command = ["./build/bench_clip_concurrency", model, input_tensor, str(workers), mode, "10", "none", "host"]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
        item = {"name": name, "repeat": repeat, "model": model, "command": command,
                "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        if result.returncode == 0:
            item["report"] = json.loads(result.stdout[result.stdout.find("{"):])
        (output / f"{repeat}_{name}.json").write_text(json.dumps(item, indent=2))
        result.check_returncode()
        rows.append(item)
        print(f"DONE {name}: {item['report']['fps']:.3f}FPS", flush=True)
(output / "speed_summary.json").write_text(json.dumps(rows, indent=2))
