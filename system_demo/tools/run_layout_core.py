"""Serial fixed-input hardware probe; one model/core variable per comparison."""
import os
import argparse
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--runtime',choices=['232','system'],default='232')
parser.add_argument('--output',default='layout_core_v2_20261010')
parser.add_argument('--cores',nargs='+',choices=['all','0'],default=['all','0'])
args = parser.parse_args()
destination = root/'results'/args.output
destination.mkdir(parents=True,exist_ok=False)
inputs = [root/'results/quant_public_fp16_ieee/inputs_fp16'/(name+'_020.png.bin')
          for name in ['cardboard','glass','metal','paper','plastic','trash']]
if not all(p.exists() for p in inputs):
    raise RuntimeError('missing fixed inputs')
environment = dict(os.environ)
environment['LD_LIBRARY_PATH'] = str(root/'quant_runtime232') if args.runtime=='232' else '/usr/lib'
for name,model in [('baseline','clip_reference_fp16_232.rknn'),
                   ('all_fp16_io','clip_control_all_fp16_io_232.rknn'),
                   ('sensitive_gelu_io','clip_sensitive_gelu_io_232.rknn')]:
    for core in args.cores:
        label = name+'_'+core
        print('START '+label,flush=True)
        with (destination/(label+'.log')).open('w') as log:
            subprocess.run([str(root/'build/probe_input_layout'),str(root/model),core,
                            str(destination/label),*[str(p) for p in inputs]],
                           cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,
                           timeout=180,check=True)
        print('DONE '+label,flush=True)
