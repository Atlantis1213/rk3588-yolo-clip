"""Isolate native compiler output in per-phase logs; deterministic bounded experiment."""
from pathlib import Path
import subprocess
import sys
import json
import argparse
from prefix_specs import SPECS,directory_name

parser=argparse.ArgumentParser()
parser.add_argument('--segment',choices=list(SPECS),default='stem')
parser.add_argument('--cpu-attention',action='store_true')
parser.add_argument('--optimization-level',type=int,choices=[0,3],default=3)
args=parser.parse_args()

root=Path(__file__).resolve().parent
directory=directory_name(args.segment,args.cpu_attention,args.optimization_level)
output=root/'artifacts'/directory
output.mkdir(exist_ok=False)
for phase in ['extract','plain','step1','hybrid']:
    print('PREFIX START '+phase,flush=True)
    with (output/(phase+'.log')).open('w') as stream:
        try:
            subprocess.run([sys.executable,str(root/'build_prefix_control.py'),'--phase',phase,'--segment',args.segment,
                            '--optimization-level',str(args.optimization_level),
                            *(['--cpu-attention'] if args.cpu_attention else [])],
                           stdout=stream,stderr=subprocess.STDOUT,timeout=240,check=True)
        except Exception as error:
            (output/'status.json').write_text(json.dumps({'phase':'failed','stage':phase,'error':repr(error)}))
            raise
    print('PREFIX DONE '+phase,flush=True)
(output/'status.json').write_text(json.dumps({'phase':'completed'}))
