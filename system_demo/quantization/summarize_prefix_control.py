"""Shape-checked intermediate differential, not an end-to-end precision claim."""
import json
from pathlib import Path
import numpy as np
import argparse
from prefix_specs import SPECS,directory_name

parser=argparse.ArgumentParser()
parser.add_argument('--segment',choices=list(SPECS),default='stem')
parser.add_argument('--cpu-attention',action='store_true')
parser.add_argument('--optimization-level',type=int,choices=[0,3],default=3)
args=parser.parse_args()

directory=directory_name(args.segment,args.cpu_attention,args.optimization_level)
root=Path(__file__).resolve().parent/'artifacts'/directory
golden=np.load(root/'onnx_reference.npy')
names=[name+'_020.png.bin' for name in ['cardboard','glass','metal','paper','plastic','trash']]
def metric(a,b):
    a,b=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('invalid differential shape/output')
    return {'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            'rmse':float(np.sqrt(np.mean((a-b)**2))),'max_abs_error':float(np.abs(a-b).max())}

variants={}
for model in ['plain_fp16','hybrid_all_fp16']:
    directory=root/'board'/model
    metadata=json.loads((directory/'manifest.json').read_text())
    if metadata['dims']!=list(golden.shape[1:]):
        raise RuntimeError('unhandled runtime output layout: '+str(metadata))
    variants[model]={r['image']:np.fromfile(directory/r['file'],np.float32) for r in metadata['rows']}
plain_sim=np.load(root/'plain_simulator.npy')
hybrid_sim=np.load(root/'hybrid_simulator.npy')
rows=[]
for i,name in enumerate(names):
    rows.append({'image':name,'board_plain_to_onnx':metric(variants['plain_fp16'][name],golden[i]),
        'board_hybrid_to_onnx':metric(variants['hybrid_all_fp16'][name],golden[i]),
        'board_hybrid_to_plain':metric(variants['hybrid_all_fp16'][name],variants['plain_fp16'][name]),
        'sim_plain_to_onnx':metric(plain_sim[i],golden[i]),
        'sim_hybrid_to_onnx':metric(hybrid_sim[i],golden[i])})
keys=[key for key in rows[0] if key!='image']
report={'endpoint':json.loads((root/'provenance.json').read_text())['endpoint'],'samples':rows,
        'mean_cosine':{key:float(np.mean([r[key]['cosine'] for r in rows])) for key in keys}}
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['mean_cosine'],indent=2))
