"""Serial hardware differential for two diagnostic prefix routes."""
from pathlib import Path
import os
import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--segment',choices=['stem','block0','ln1','attn0','qkv','q','k','v','core','projection'],default='stem')
parser.add_argument('--cpu-attention',action='store_true')
parser.add_argument('--optimization-level',type=int,choices=[0,3],default=3)
args=parser.parse_args()

root=Path(__file__).resolve().parents[1]
directory='prefix_control_20261010' if args.segment=='stem' else 'prefix_'+args.segment+'_control_20261010'
if args.cpu_attention: directory=directory.replace('_control_','_sdpa_cpu_control_')
if args.optimization_level==0: directory=directory.replace('_control_','_opt0_control_')
models=root/'quantization/artifacts'/directory
output=root/'results'/directory
output.mkdir(exist_ok=False)
inputs=[root/'results/quant_public_fp16_ieee/inputs_fp16'/(name+'_020.png.bin')
        for name in ['cardboard','glass','metal','paper','plastic','trash']]
environment=dict(os.environ,LD_LIBRARY_PATH=str(root/'quant_runtime232'))
for name in ['plain_fp16','hybrid_all_fp16']:
    with (output/(name+'.log')).open('w') as stream:
        subprocess.run([str(root/'build/dump_prefix_outputs'),str(models/(name+'.rknn')),
                        str(output/name),*[str(p) for p in inputs]],
                       cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=90,check=True)
    print('DONE '+name,flush=True)
