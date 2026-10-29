"""Verify validation tensors are exactly the saved ordinary-FP16 baseline inputs."""
from pathlib import Path
import json
import hashlib
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--variants',nargs='+',default=['fp16_except_sdpa','sensitive_except_sdpa'])
parser.add_argument('--report-name',default='input_equality.json')
args=parser.parse_args()
if Path(args.report_name).name!=args.report_name or not args.report_name.endswith('.json'):
    raise ValueError('report name must be a plain JSON filename')

root=Path(__file__).resolve().parents[1]
base=root/'results/quant_public_fp16_ieee'
base_manifest=json.loads((base/'manifest.json').read_text())
names=[r['image'] for r in base_manifest['samples']]
output=root/'results/sdpa_safe_full_20261010'
results=[]
for variant in args.variants:
    directory=root/'results'/('quant_sdpa_safe_'+variant+'_ieee')
    manifest=json.loads((directory/'manifest.json').read_text())
    if [r['image'] for r in manifest['samples']]!=names:raise RuntimeError('sample order differs')
    for name in names:
        a=(base/'inputs_fp16'/(name+'.bin')).read_bytes()
        b=(directory/'inputs_fp16'/(name+'.bin')).read_bytes()
        if a!=b:raise RuntimeError('input bits differ: '+variant+' '+name)
    results.append({'variant':variant,'count':len(names),'input_bits_equal_to_baseline':True})
(output/args.report_name).write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
