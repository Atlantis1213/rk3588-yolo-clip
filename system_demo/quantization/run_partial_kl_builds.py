"""Three predefined equal-cost precision ablations; serialized SDK workers."""
import argparse
from pathlib import Path
import subprocess
import sys
from partial_mlp_policy import PARTIAL_KL_DOWN

parser = argparse.ArgumentParser()
parser.add_argument("--variants", nargs="+", choices=list(PARTIAL_KL_DOWN), default=list(PARTIAL_KL_DOWN))
args = parser.parse_args()
if len(set(args.variants)) != len(args.variants):
    raise ValueError("duplicate build variant")
root = Path(__file__).resolve().parent
for variant in args.variants:
    subprocess.run([sys.executable, str(root / "build_sdpa_safe_full.py"), "--variant", variant], check=True)
