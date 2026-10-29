"""Run tiny boundary diagnostics serially without competing native compiler workers."""
from pathlib import Path
import subprocess
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--segments',nargs='+',default=['qkv','q','k','v','core'])
args=parser.parse_args()
root=Path(__file__).resolve().parent
for segment in args.segments:
    print('BOUNDARY '+segment,flush=True)
    subprocess.run([sys.executable,str(root/'run_prefix_control.py'),'--segment',segment],check=True)
