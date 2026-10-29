import argparse
import json
from pathlib import Path
from sweep_npu_scheduling import execute, pipeline_command

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results/npu_adaptive_20261009")
parser.add_argument("--duration", type=int, default=20)
parser.add_argument("--rounds", type=int, default=3)
args = parser.parse_args()
if args.duration <= 0 or args.rounds <= 0:
    parser.error("duration and rounds must be positive")
output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
configs = [
    ("baseline", 2, "legacy", 4, "round_robin", "none"),
    ("adaptive_1y3c", 1, "adaptive", 3, "spread", "per_core"),
    ("adaptive_2y3c", 2, "adaptive", 3, "spread", "per_core"),
]
items = []
for repeat in range(args.rounds):
    offset = repeat % len(configs)
    order = configs[offset:] + configs[:offset]
    for config in order:
        items.append({"config": config, "report": execute(pipeline_command(config, args.duration), output,
                     f"{repeat}_{config[0]}")["report"]})
summary = []
for config in configs:
    selected = [item["report"] for item in items if item["config"][0] == config[0]]
    summary.append({"config": config, "means": {key: sum(r[key] for r in selected) / len(selected) for key in
         ("completed_fps", "latency_ms_p95", "clip_run_ms_mean", "yolo_run_ms_mean", "yolo_npu_gate_wait_ms_mean")}})
(output / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
